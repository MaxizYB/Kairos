# EfficientAgent request-admission source

Repository: https://github.com/KunmingSHAO/efficientagent_release

Commit: `b21b12a07a3ccf473c82acf00820b1aa2073044b`.

`connector.py`: upstream `efficientagent/kvtier/connector.py`, Git blob
`ba0cd04263ba43c488ead58b916dd7985105af5c`; byte-for-byte verified.

The MP adapter executes the unchanged `_pressure`,
`get_num_new_matched_tokens` and `set_skip_save` functions extracted from
this file. Construction, lookup bridge, telemetry transport and STORE gate
are project adapters. It does not deploy the original LMCacheConnectorV1
or reproduce the entire EfficientAgent system. Native MP reservation skips
already-present destinations, supplying the equivalent copy dedup path.

Comparison keeps the common 256-token chunk, Qwen3-8B KV footprint and
2 GiB host tier; official kappa=8, theta=.95, window=60 s remain unchanged.
Upstream's default chunk is 1024: these are not identical physical write
thresholds. The matrix also includes conditioned kappa=32, matching the
upstream default's 8192-token limit. A full chunk-size/threshold sensitivity
is still required for a paper-level comparison. Full original deployment
remains unmeasured.

The independent 2026-10-10 threshold matrix additionally tested conditioned
kappa=8/16/24 with eager and official lazy-h10, without changing the upstream
functions. All 640 requests completed and all first-lookup decisions matched
the source rule. This finite sweep is not a full threshold/chunk-size tuning
or a reproduction of the original backend; results are recorded in
`V21_Strong_Baseline_Findings_20261010.md`, section 9.
