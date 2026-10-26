"""Kairos range lookup implementation."""
from __future__ import annotations


class RangeLookupAdapter:
    def __init__(self, base, send, lookup_request_type):
        self.base = base
        self.send = send
        self.lookup_request_type = lookup_request_type
        self.starts = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def maybe_submit_lookup_request(self, request_id, token_ids, cache_salt="", request_configs=None):
        start = self.starts.get(request_id, 0)
        if not start:
            return self.base.maybe_submit_lookup_request(
                request_id, token_ids, cache_salt, request_configs)
        base = self.base
        if base.tp_size != 1 or len(base._server_urls) != 1:
            raise ValueError("Range lookup prototype requires TP=1 and one LMCache server")
        base._ensure_heartbeat_started()
        if not base.is_healthy or request_id in base._pending_lookups:
            return
        end = len(token_ids)//base.lmcache_tokens_per_chunk*base.lmcache_tokens_per_chunk
        if start >= end:
            base._pending_lookups.add(request_id)
            base._lookup_params[request_id] = (token_ids, cache_salt, request_configs)
            base._finished_lookup_results[request_id] = 0
            return
        key = base._create_key(token_ids, start=start, end=end, request_id=request_id,
                               cache_salt=cache_salt, request_configs=request_configs).no_worker_id_version()
        for url in base._server_urls:
            future = self.send(base.mq_clients[url], self.lookup_request_type, [key, base.tp_size])
            # This waits for job submission, as in the native adapter. It
            # does not wait for GPU restore or claim unlocked objects.
            future.result(timeout=base._mq_timeout)
        base._pending_lookups.add(request_id)
        base._lookup_params[request_id] = (token_ids, cache_salt, request_configs)

    def check_lookup_result(self, request_id):
        result = self.base.check_lookup_result(request_id)
        # Native result is a relative token count for the requested range.
        # Only a positive CPU hit extends the available GPU prefix.
        return result + self.starts.get(request_id, 0) if result else result

    def free_lookup_locks(self, token_ids, start, end, request_id, **kwargs):
        start = max(start, self.starts.get(request_id, 0))
        if end > start:
            return self.base.free_lookup_locks(token_ids, start, end, request_id, **kwargs)

    def cleanup_lookup_result(self, request_id):
        # update_state_after_alloc cleans the job before releasing overlap
        # locks. Retain the range until end_session has released the session.
        return self.base.cleanup_lookup_result(request_id)

    def end_session(self, request_id):
        result = self.base.end_session(request_id)
        self.base.cleanup_lookup_result(request_id)
        self.starts.pop(request_id, None)
        return result
