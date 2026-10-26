"""Kairos gpu implementation."""
import math
import os


def install(scheduler):
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    if len(managers) != 1 or not isinstance(managers[0], FullAttentionManager):
        raise ValueError("Gapped prototype requires one full-attention group")
    manager = managers[0]
    if manager.block_size != 16:
        raise ValueError("Gapped prototype requires 16-token GPU pages")
    pool = manager.block_pool
    native_allocate = manager.allocate_new_blocks
    native_cache = pool.cache_full_blocks
    import inspect
    cache_signature=inspect.signature(native_cache)

    def allocate(request_id, num_tokens, num_tokens_main_model):
        if (not scheduler.gapped_enabled or
                (getattr(scheduler.connector, 'uses_gapped', None) is not None
                 and not scheduler.connector.uses_gapped(request_id))):
            return native_allocate(request_id, num_tokens, num_tokens_main_model)
        request = scheduler.requests[request_id]
        blocks = manager.req_to_blocks[request_id]
        partial = manager._partial_hit_reqs.get(request_id)
        stop = math.ceil(num_tokens / 16)
        if stop <= len(blocks) and partial is None:
            return []
        loaded = scheduler.connector.load_chunks(request_id)
        adopt_load_hits = os.environ.get('CONTINUUM_GAPPED_ADOPT_LOAD_HITS', '0') == '1'
        adopted, holes = {}, []
        for index in range(len(blocks), stop):
            found = (pool.get_cached_block(request.block_hashes[index], [0])
                     if index < (request.num_prompt_tokens - 1) // 16
                     and (adopt_load_hits or index // 16 not in loaded) else None)
            # Segment-budget merging can recompute a hit. Only adopt pages
            # which no other request owns, so such writes cannot race with
            # another request reading or extending a shared cache page.
            # A whole CPU chunk may include exact GPU hits. With exclusive
            # request ownership, restoring the SAME hashed history into
            # those pages is safe and needs no replacement allocation. The
            # source CPU session and destination GPU references still stay
            # live until the connector's normal consumption fence.
            if found is not None and found[0].ref_cnt == 0:
                adopted[index] = found[0]
            else:
                holes.append(index)
        # Protect hits BEFORE drawing any free pages; otherwise native LRU
        # could recycle a later hit while filling an earlier hole.
        pool.touch(list(adopted.values()))
        fresh = pool.get_new_blocks(len(holes) + int(partial is not None))
        cow = []
        if partial is not None:
            index, source = manager._partial_hit_reqs.pop(request_id)
            private, fresh = fresh[0], fresh[1:]
            # Use the engine's existing copy queue and source-reference
            # lifetime. The worker fences this copy before model execution.
            manager._apply_cow(request_id, index, source, private)
            manager.new_block_ids.append(private.block_id)
            cow.append(private)
        missing = dict(zip(holes, fresh, strict=True))
        appended = [adopted.get(i, missing.get(i)) for i in range(len(blocks), stop)]
        if any(b is None or b.is_null for b in appended):
            raise RuntimeError("Incomplete gapped physical block table")
        blocks.extend(appended)
        scheduler.gapped_gpu_pages.setdefault(request_id, set()).update(adopted)
        if manager._record_new_block_ids:
            manager.new_block_ids.extend(b.block_id for b in fresh)
        return cow + appended

    def cache(*args, **kwargs):
        if not scheduler.gapped_enabled:
            return native_cache(*args, **kwargs)
        # The native pool promotes partial hashes but asserts on already-full
        # pages. Those are precisely our adopted pages: skip re-registering
        # them, while retaining native hash publication for computed holes.
        bound = cache_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if (getattr(scheduler.connector, 'uses_gapped', None) is not None
                and not scheduler.connector.uses_gapped(values['request'].request_id)):
            return native_cache(*args, **kwargs)
        start, stop = values['num_cached_blocks'], values['num_full_blocks']
        mask = values['block_mask'] or [True] * (stop - start)
        values['block_mask'] = [keep and (b.block_hash is None or
            b.block_hash_num_tokens < (start + i + 1) * values['block_size'])
            for i, (keep, b) in enumerate(zip(mask,
                values['blocks'][start:stop], strict=True))]
        return native_cache(**values)

    manager.allocate_new_blocks = allocate
    pool.cache_full_blocks = cache
