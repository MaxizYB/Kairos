"""Kairos range helper implementation."""
import copy
from functools import wraps

from kairos._runtime.compat.replica import replica_model_supported


class RangeHasher:
    def __init__(self, base, start, end):
        self.base, self.start, self.end = base, start, end

    def __getattr__(self, name):
        return getattr(self.base, name)

    def compute_chunk_hashes(self, token_ids, **kwargs):
        if kwargs:
            raise ValueError("Unexpected nested range hash request")
        return self.base.compute_chunk_hashes(token_ids, start=self.start, end=self.end)


class RangeContext:
    def __init__(self, base, hasher):
        self.base = base
        self.token_hasher = hasher

    def __getattr__(self, name):
        return getattr(self.base, name)


def install_range_lookup(lookup_class):
    original_lookup = lookup_class.lookup
    original_query = lookup_class.query_prefetch_status

    @wraps(original_lookup)
    def lookup(module, key, tp_size):
        if not key.start:
            return original_lookup(module, key, tp_size)
        if (not replica_model_supported(key.model_name) or key.world_size != 1
                or key.start % module._ctx.chunk_size or key.end % module._ctx.chunk_size):
            raise ValueError("Range lookup requires the validated Qwen3-8B single-GPU layout")
        # A call-private module/context view shares the original job locks,
        # layout, sessions and native storage manager, but never mutates the
        # context/hasher seen by concurrent STORE/RETRIEVE/ordinary LOOKUP.
        view = copy.copy(module)
        view._ctx = RangeContext(module._ctx,
                                RangeHasher(module._ctx.token_hasher, key.start, key.end))
        return original_lookup(view, key, tp_size)

    @wraps(original_query)
    def query(module, request_id):
        result = original_query(module, request_id)
        if result is not None:
            session = module._ctx.session_manager.get_or_create(request_id)
            key = session.lookup_ipc_key
            if key is not None and key.start:
                # Native resolve_prefetched_obj_keys intersects an absolute
                # hit endpoint with key.start/end, not a relative count.
                session.record_prefetch_result(
                    result + key.start//module._ctx.chunk_size,
                    session.prefetch_locked_gids)
        return result

    lookup_class.lookup = lookup
    lookup_class.query_prefetch_status = query
