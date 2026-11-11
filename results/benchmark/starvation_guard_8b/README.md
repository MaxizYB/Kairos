# Kairos bounded-aging guard: 8B comparison

This archive records one forward fixed-work Replay for each arm on Qwen3-8B.
The unguarded arm is the existing Kairos scheduler; the guarded arm adds the
queue-local bounded-aging policy described in the method document.

| Workload | Arm | Mean JCT (s) | P95 JCT (s) | Replay wall (s) | Mean TTFT (s) | P95 TTFT (s) |
|---|---|---:|---:|---:|---:|---:|
| 16 programs / 178 calls | Kairos | 91.43 | 142.53 | 145.13 | 3.53 | 17.90 |
| 16 programs / 178 calls | Kairos + guard | **88.77** | **137.93** | 145.56 | **3.32** | **11.64** |
| 32 programs / 420 calls | Kairos | 120.52 | 219.14 | 247.57 | 6.08 | 36.93 |
| 32 programs / 420 calls | Kairos + guard | 122.43 | 226.47 | 254.93 | 6.15 | **36.79** |

The 32-program workload is not a 36-program trace. Both workloads use the
fixed-work-private-stop-v1 protocol, completed all calls, had zero failures,
and passed the post-run audit. The guard registry was emitted in guarded runs,
but activation count was zero in both workloads because no request crossed the
default four-second waiting threshold. Therefore this archive is an
installation/no-regression comparison, not evidence that starvation relief has
already been exercised end to end.

The JSON file contains the exact floating-point summaries and SHA-256 digests
of the source summary/audit files retained in the experiment workspace.
