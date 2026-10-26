# Official LMCache lazy-offload snapshot

Repository: https://github.com/LMCache/LMCache

Commit: `68a9c6b5667ff6d9493530984bd2e40fb3941451`

Source directory: `lmcache/integration/vllm/`.

All eight Python files match the Git blobs in the commit's recursive tree (verified 2026-10-09). Apache-2.0 SPDX headers are preserved. The six lazy policy/manager/state files are loaded unmodified by the private experiment adapter. The full connector and metadata files are read-only references; their complete newer dependencies are not installed.

| File | Upstream Git blob SHA-1 |
|---|---|
| `lazy_offload_manager.py` | `4d8bbd9f3475bfed0c1b394d186552072c2a2f7f` |
| `lazy_offload_policy/__init__.py` | `e7a45afa043ebc4db203b806a02cb43f7467fac8` |
| `lazy_offload_policy/base.py` | `376432fc259d50a016dff900e4e3fe3674ed4f73` |
| `lazy_offload_policy/eviction_aware.py` | `54ca54b1174238237e690debf2d8e955a92bbe34` |
| `lazy_offload_policy/fifo.py` | `a09cc99107614e11ffd0ba1f1c62d61a9dc95ed0` |
| `lazy_offload_state.py` | `4a374907bf7c5eb688a9212baf3af73e28700c62` |
| `lmcache_mp_connector.py` | `b4a9275e9ef4b36df71d84ba6af6747ffe31e5d5` |
| `lmcache_mp_metadata.py` | `aeca3b3ac9c2da8ea6ce4169b024338eda795fe5` |
