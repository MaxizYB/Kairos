# HiCache source audit snapshot

Repository: https://github.com/sgl-project/sglang

Commit: `3dc91366ac04b314f856839cb1a76e9a66aac9e7` (PR #33777).

`full_component.py` is an unmodified snapshot of
`python/sglang/srt/mem_cache/unified_cache/components/full_component.py`.
Its Git blob SHA-1 is `3ec838c382a6407b7be6434252c0b05682833310` (checked locally on 2026-10-10).
It establishes device/host leaf eviction, session coverage, host anchoring locks,
and the GPU-prefix/host-suffix load-back path.

`unified_tree_core.py` is an unmodified snapshot of
`python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py`.
Its Git blob SHA-1 is `d1acd88b1e0e15cb0359da8b0977b3445616ccd0`
(downloaded using the GitHub blob API and checked locally on 2026-10-10).
It establishes the two-pass duplicate reclaim rule, imminent-demotion
exclusion, native lock/in-flight checks, write-back before device eviction,
and host/device tree-dependency maintenance.

These two files are source-audit snapshots, not a complete HiCache deployment.
The other engine/tree/component dependencies have not been installed.

The mechanism comparison added on 2026-10-10 executes verified functions
from these snapshots via AST extraction in `continuum_compat/hicache_policy.py`.
Both blobs are checked again before execution. The executed functions are
the two-pass duplicate sweep, its eligibility predicate, the device/host
leaf predicates, and the official host/device eviction walks.

The adapter maps unshared Program paths to native vLLM pages and LMCache
objects. The backend retains its own allocation, locks and transfer events.
Demotion-before-reuse uses real D2H and holds source references until host
commit. This is a source-verified mechanism port, not a complete SGLang
HiCache deployment; session caching, shared radix paths, auxiliary cache
components and the SGLang execution/transfer backend are not reproduced.
Measured results must be labelled accordingly.

The 2026-10-11 supplemental comparator also composes the same verified
device-leaf walk with official EfficientAgent admission, the verified
HiCache host sweep, and the common eager STORE path. It changes only free
GPU page selection; it neither pins additional pages nor initiates
write-back. This composition isolates selection from demotion-copy overhead
and is explicitly constructed by this study, not a HiCache default mode.
