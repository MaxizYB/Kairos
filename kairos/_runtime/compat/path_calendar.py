"""Kairos path calendar implementation."""
from copy import deepcopy
import math
import time


def gate_open(gate, completed):
    return all(completed.get(p, 0) >= n for p, n in gate.items())


def call_pages(program, stage, page=16):
    turn = program.turns[stage]
    return math.ceil((turn.prompt + turn.output) / page)


def simulate_calendar(programs, order, beta, decode_steps, *, gates=None,
                      capacity_pages=3071, page_size=16, headroom_pages=32,
                      max_seqs=8, query_budget=2048, mixed_floor=0.,
                      host_step=0., strict_order=True, static_peak=False,
                      record=False, initial=None, new_peak=False, minimum_pages=None, **unused):
    """Native-like batching with complete closed-loop Program continuations.

    A call reserves its entire known prompt+output, a tool phase retains the
    preceding history, and a returned call must fund its growth before it
    starts. No policy preemption, eviction, free return capacity, or future
    arrivals injected into the running cohort. Unknown future arrivals are
    used only by the explicitly enabled strict-order oracle variant.
    """
    by = {p.name: p for p in programs}
    if len(order) != len(by) or set(order) != set(by):
        raise ValueError('Order must contain every Program exactly once')
    if gates and (set(gates) - set(by) or any(
            p not in by or n < 0 or n > len(by[p].turns)
            for gate in gates.values() for p, n in gate.items())):
        raise ValueError('Invalid completion milestone')
    rank = {p: i for i, p in enumerate(order)}
    minimum_pages = minimum_pages or {}
    peaks = {p: max(by[p].peak_pages(page_size), minimum_pages.get(p,0)) for p in by}
    def target_pages(p, stage):
        return max(call_pages(by[p],stage,page_size),minimum_pages.get(p,0) if stage==0 else 0)
    limit = capacity_pages - headroom_pages
    if limit <= 0 or any(v > limit for v in peaks.values()):
        raise ValueError('A Program cannot fit the declared envelope')
    pending = set(by); rows = {}; done = {}; completed = {p: 0 for p in by}
    events = []; t = 0.; area = 0.; engine = 0.; queries = 0; steps = 0
    peak_reserved = 0; reserved_area = 0.; decode_hist = {}

    def log(event, **kw):
        if record: events.append(dict(time=t, event=event, **kw))

    def used():
        return sum(row['pages'] for row in rows.values())

    def new_row(p, stage, ready, held):
        turn = by[p].turns[stage]
        keep = min(turn.keep // page_size * page_size, max(0, turn.prompt - 1)) if stage else 0
        return dict(stage=stage, ready=ready, computed=keep, generated=0,
                    pages=peaks[p] if static_peak else held, started=False)

    # Online residual rollouts start with actual admitted work. They never
    # restart its prefill, make its current references disappear, or pretend
    # an outstanding tool has already returned. Default oracle behavior above
    # is unchanged. These are estimates of future execution, not native state.
    for p, state in (initial or {}).items():
        if p not in by: raise ValueError('Unknown initial Program')
        row = new_row(p, 0, float(state.get('ready', 0)), int(state['pages']))
        row.update({k: state[k] for k in ('computed', 'generated', 'started') if k in state})
        if row['pages'] < 0 or row['computed'] < 0 or row['generated'] < 0:
            raise ValueError('Negative initial resource/work')
        if row['generated'] >= by[p].turns[0].output:
            raise ValueError('Initial call must have unfinished output')
        rows[p] = row; pending.remove(p)
    if used() > limit:
        return dict(feasible=False, reason='initial overcommit', events=[])

    def advance(end):
        nonlocal t, area, reserved_area
        area += sum(max(0., end - max(t, p.arrival)) for p in programs
                    if p.name not in done and p.arrival < end)
        reserved_area += used() * (end - t)
        t = end

    while len(done) < len(by):
        # Preserve existing running work; admit returned calls before fresh
        # Programs, and fund complete-call growth instead of one prefill chunk.
        for p in sorted(rows, key=rank.get):
            row = rows[p]
            if row['ready'] > t or row['started']: continue
            target = peaks[p] if static_peak else target_pages(p, row['stage'])
            if new_peak: target = max(target,row['pages'])
            if used() - row['pages'] + target <= limit:
                row.update(pages=target, started=True)
                log('start_call', program=p, stage=row['stage'], reserved_pages=used())
        while len(rows) < max_seqs:
            candidates = sorted((p for p in pending if by[p].arrival <= t), key=rank.get)
            if strict_order and pending:
                first = min(pending, key=rank.get)
                candidates = [first] if by[first].arrival <= t else []
            if not candidates: break
            p = candidates[0]
            target = peaks[p] if static_peak or new_peak else target_pages(p, 0)
            if not gate_open((gates or {}).get(p, {}), completed) or used() + target > limit:
                break
            pending.remove(p)
            rows[p] = new_row(p, 0, t, target)
            rows[p]['started'] = True
            log('admit', program=p, reserved_pages=used(), completed=dict(completed))
        peak_reserved = max(peak_reserved, used())
        ready = sorted((p for p, row in rows.items()
                        if row['started'] and row['ready'] <= t), key=rank.get)
        if not ready:
            future = [by[p].arrival for p in pending if by[p].arrival > t]
            future += [r['ready'] for r in rows.values() if r['ready'] > t]
            if not future:
                return dict(feasible=False, reason='future growth deadlock', time=t,
                            completed=completed, pending=sorted(pending), held=deepcopy(rows),
                            events=events if record else None)
            advance(min(future)); continue
        left = query_budget; Q = A = prefill = 0; members = []
        decoders = [p for p in ready if rows[p]['computed'] >= by[p].turns[rows[p]['stage']].prompt]
        for p in decoders + [p for p in ready if p not in decoders]:
            if not left: break
            row = rows[p]; turn = by[p].turns[row['stage']]
            if row['computed'] < turn.prompt:
                n = min(left, turn.prompt - row['computed'])
                begin = row['computed'] + 1
                A += n * (2 * begin + n - 1) // 2
                row['computed'] += n; prefill += n
                if row['computed'] == turn.prompt: row['generated'] += 1
            else:
                n = 1; A += turn.prompt + row['generated']; row['generated'] += 1
            Q += n; left -= n; members.append(p)
        N = len(members)
        ds = float(decode_steps.get(str(N), decode_steps.get(N, .02)))
        duration = (max(float(mixed_floor), float(beta[0]) + float(beta[1]) * Q +
                    float(beta[2]) * A + float(beta[3]) * N, ds) if prefill else ds) + host_step
        if not duration > 0: raise ValueError('Nonpositive service duration')
        # Exact aggregation within this model while the pure-decode batch and
        # every event predicate are constant. Completion/arrival boundaries
        # force a normal iteration; no approximation of partial output work.
        repeat = 1
        if not prefill:
            until_finish = min(by[p].turns[rows[p]['stage']].output - rows[p]['generated'] for p in members)
            future = [by[p].arrival for p in pending if by[p].arrival > t]
            future += [r['ready'] for r in rows.values() if r['ready'] > t]
            until_event = max(1, math.ceil((min(future) - t) / duration - 1e-10)) if future else until_finish + 1
            repeat = min(until_finish + 1, until_event)
            for p in members: rows[p]['generated'] += repeat - 1
            decode_hist[N] = decode_hist.get(N, 0) + repeat
        advance(t + duration * repeat)
        engine += duration * repeat; queries += prefill; steps += repeat
        for p in members:
            row = rows[p]; turn = by[p].turns[row['stage']]
            if row['generated'] < turn.output: continue
            stage = row['stage']; completed[p] = stage + 1
            log('turn_finish', program=p, stage=stage)
            if stage + 1 == len(by[p].turns):
                done[p] = t; del rows[p]; log('finish', program=p)
            else:
                held = call_pages(by[p], stage, page_size)
                rows[p] = new_row(p, stage + 1, t + turn.gap,
                                  max(held, row['pages']) if new_peak else held)
    total = sum(done[p.name] - p.arrival for p in programs)
    if not math.isclose(total, area, rel_tol=1e-8, abs_tol=1e-6):
        raise AssertionError('Completion sum and unfinished area disagree')
    return dict(feasible=True, mean_jct_s=total / len(by), sum_jct_s=total,
                completion_s=done, makespan_s=t, prefill_queries=queries,
                engine_s=engine, steps=steps, peak_reserved_pages=peak_reserved,
                reserved_page_seconds=reserved_area, decode_iterations_by_N=decode_hist,
                events=events if record else None)


def admission_gates(events):
    return {e['program']: {p: n for p, n in e['completed'].items() if n}
            for e in events if e['event'] == 'admit'}


def compile_funding_steps(events, programs, page_size=16):
    """Compile the ENTIRE model resource schedule, not just first admissions.

    Every growing envelope is funded in the planned order after all prior
    completion milestones. Funding may precede the actual tool return; the
    reservation then cannot be consumed by another, faster-moving Program.
    This first implementation requires nondecreasing per-Program footprints.
    """
    by = {p.name: p for p in programs}; completed = {}; steps = []
    for p in programs:
        pages = [call_pages(p, k, page_size) for k in range(len(p.turns))]
        if any(a > b for a, b in zip(pages, pages[1:])):
            raise ValueError('Funding compiler currently requires monotone call footprints')
    for e in events:
        if e['event'] == 'turn_finish': completed[e['program']] = e['stage'] + 1
        elif e['event'] in ['admit', 'start_call']:
            p = e['program']; stage = e.get('stage', 0)
            steps.append(dict(program=p, stage=stage, pages=call_pages(by[p], stage, page_size),
                              after=dict(completed)))
    if len(steps) != sum(len(p.turns) for p in programs):
        raise ValueError('Incomplete resource schedule')
    return steps


def improve_calendar(programs, order, beta, decode_steps, *, passes=2, **settings):
    """Coordinate search of milestone guards; no claim of global optimality.

    All preceding completion-prefix milestones in the current feasible plan
    are considered for each admission. Invalid/deadlocked full continuations
    are rejected, rather than assigned an optimistic finite terminal value.
    """
    started = time.perf_counter(); evaluations = 0; rejected = 0
    initial = simulate_calendar(programs, order, beta, decode_steps,
                                static_peak=True, record=True, **settings)
    if not initial['feasible']: raise ValueError('Reference schedule is infeasible')
    gates = admission_gates(initial['events'])
    best = simulate_calendar(programs, order, beta, decode_steps, gates=gates, record=True, **settings)
    if not best['feasible']: raise AssertionError('Peak-feasible guards became infeasible')
    history = []; converged = False
    for iteration in range(passes):
        improved = False
        for p in order:
            prefixes = [{}]; completed = {}; seen = {()}
            for e in best['events']:
                if e['event'] == 'admit' and e['program'] == p: break
                if e['event'] == 'turn_finish':
                    completed[e['program']] = e['stage'] + 1
                    key = tuple(sorted(completed.items()))
                    if key not in seen: prefixes.append(dict(completed)); seen.add(key)
            choice = gates[p]; value = best
            for gate in prefixes:
                candidate = dict(gates, **{p: gate})
                result = simulate_calendar(programs, order, beta, decode_steps,
                                             gates=candidate, record=True, **settings)
                evaluations += 1
                if not result['feasible']: rejected += 1; continue
                if result['sum_jct_s'] < value['sum_jct_s'] - 1e-8:
                    choice, value = gate, result
            if choice != gates[p]:
                gates[p] = choice; best = value; improved = True
        history.append(dict(iteration=iteration, mean_jct_s=best['mean_jct_s']))
        if not improved: converged = True; break
    return dict(order=list(order), gates=gates, simulation=best, initial_peak=initial,
                evaluations=evaluations, infeasible_candidates=rejected, passes=history,
                converged=converged, planning_wall_s=time.perf_counter()-started,
                optimality='best complete-path feasible milestone plan; no global optimality certificate')
