"""Kairos execution capacity implementation."""
import re


def replay_arm(name):
    """Keep registered capacity labels, preserving legacy group stripping."""
    return '~'.join(part for i, part in enumerate(name.split('~'))
                    if i == 0 or re.fullmatch(r'cap\d+', part)
                    or part in ('holdlinked', 'holdsplit', 'finishlogical'))


def replay_repeat(name):
    match = re.search(r'(?:^|~)(r\d+)(?=~|$)', name)
    if match is None:
        raise ValueError('Missing replay repeat label')
    return match.group(1)


def registered_caps(value, runner_capacity):
    if not value:
        return ()
    caps = tuple(int(part) for part in value.split(','))
    if len(set(caps)) != len(caps) or any(cap < 1 or cap > runner_capacity for cap in caps):
        raise ValueError('Real request caps must be unique and fit the model runner')
    return caps


def experiment_cap(name, caps, runner_capacity):
    if not caps:
        return runner_capacity
    labels = re.findall(r'(?:^|~)cap(\d+)(?=~|$)', name)
    if len(labels) > 1:
        raise ValueError('Multiple real request caps in an experiment name')
    cap = int(labels[0]) if labels else caps[0]
    if cap not in caps:
        raise ValueError('Unregistered real request cap')
    return cap


def audit_waiting_capacity(scheduler, num_running):
    """Bounded read-only gate samples; no connector lookup or page touches.

    Prefix hashes and native capacity accounting are observed without adopting
    any page. Native step reservation, connector readiness and future growth
    are not implied by this diagnostic. Real admission remains authoritative.
    """
    import time
    now = time.monotonic()
    if now < getattr(scheduler, '_next_execution_capacity_audit', 0.0):
        return
    scheduler._next_execution_capacity_audit = now + 0.5
    began = time.perf_counter()
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    candidates = []
    for queue_name in ('waiting', 'skipped_waiting'):
        for request in getattr(scheduler, queue_name):
            if len(candidates) >= 4:
                break
            status = request.status.name
            item = dict(request=request.request_id, program=scheduler.program(request),
                        status=status, prompt_tokens=request.num_prompt_tokens,
                        stale_outputs=request.num_stale_output_tokens,
                        finished_receiving=request.request_id in scheduler.finished_recving_kv_req_ids)
            # Read-only full-block prefix match. Do not call connector lookup,
            # native promotion or get_computed_blocks (which can emit events).
            hits = []
            if request.num_computed_tokens == 0 and status in ('WAITING', 'PREEMPTED'):
                for block_hash in request.block_hashes[:(request.num_prompt_tokens - 1) // scheduler.block_size]:
                    found = pool.get_cached_block(block_hash, [0])
                    if found is None:
                        break
                    hits.append(found[0])
            blocks = manager.create_kv_cache_blocks([hits])
            demand = scheduler._required_capacity(request, 1, dict(
                num_new_computed_tokens=len(hits) * scheduler.block_size,
                new_computed_blocks=blocks,
                num_lookahead_tokens=scheduler.num_lookahead_tokens,
                full_sequence_must_fit=bool(scheduler.scheduler_reserve_full_isl),
                has_scheduled_reqs=bool(scheduler.running),
            ))
            item.update(demand, local_prefix_tokens=len(hits) * scheduler.block_size,
                        fits_without_step_reservation=demand['D'] == 0)
            candidates.append(item)
    scheduler._emit('execution_capacity_gate', running=num_running,
        real_request_cap=scheduler.max_num_running_reqs,
        runner_descriptors=scheduler.gapped_runner_capacity,
        reserved_segment_slots=scheduler._gapped_reserved_slots(),
        pending_batches=len(getattr(scheduler, '_pipeline_inflight', {})),
        candidates=candidates, audit_s=time.perf_counter() - began,
        read_only=True, native_step_reservation_included=False,
        external_matches_queried=False, future_decode_growth_included=False)
