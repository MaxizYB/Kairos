"""Kairos decode graph policy implementation."""


def set_decode_graph_policy(dispatcher,exact):
    capture=list(range(1,9)) if exact else [1,2,4,8]
    config=dispatcher.compilation_config
    original=config.cudagraph_capture_sizes
    try:
        config.cudagraph_capture_sizes=capture
        for keys in dispatcher.cudagraph_keys.values():keys.clear()
        dispatcher.initialize_cudagraph_keys(dispatcher.cudagraph_mode,
            uniform_decode_query_len=dispatcher.uniform_decode_query_len)
    finally:
        config.cudagraph_capture_sizes=original
    return dict(exact=bool(exact),active_capture_sizes=capture,
        bs_to_padded_graph_size=list(dispatcher._bs_to_padded_graph_size),
        capture_is_common=True)
