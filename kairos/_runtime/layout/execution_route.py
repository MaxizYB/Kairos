"""Kairos execution route implementation."""
from kairos._runtime.layout.plan import query_segments


def execution_route(begin, prompt, gpu_pages, cpu_chunks, planned_loads,
                    page=16, chunk=256, slots=2, costs=None, dependency_step=0.,
                    query_budget=2048):
    if begin >= prompt:
        return dict(route='prefix', native_queries=0, segmented_queries=0,
                    native_frontier=begin)
    frontier = begin
    native_loads = 0
    while frontier // chunk in cpu_chunks:
        native_loads += 1
        frontier = (frontier // chunk + 1) * chunk
    frontier = min(frontier, prompt - 1)
    valid = set(gpu_pages) | {j for i in planned_loads
        for j in range(i * (chunk // page), (i + 1) * (chunk // page))}
    segments = query_segments(begin, prompt, valid, page=page, slots=slots)
    queries = sum(b - a for a, b in segments)
    native_queries = prompt - frontier
    result = dict(route='segmented' if queries < native_queries else 'prefix',
                native_queries=native_queries, segmented_queries=queries,
                native_frontier=frontier)
    if costs is not None:
        load = costs.load_token * chunk + costs.load_call
        native = native_loads * load + costs.compute([(frontier, prompt)])
        # A submitted exact segment must settle before another true batch.
        # Price that explicit extra dependency with the shared fixed-step
        # coefficient, not a future tool/output predictor or a new threshold.
        steps = max(1, (queries + query_budget - 1) // query_budget)
        segmented = (len(planned_loads) * load + costs.compute(segments)
                     + steps * dependency_step)
        result.update(native_recovery_s=native, segmented_recovery_s=segmented,
                      dependency_s=steps * dependency_step,
                      cost_rejected=result['route']=='segmented' and segmented >= native)
        if segmented >= native:
            result['route'] = 'prefix'
    return result
