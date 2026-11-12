# Kairos: continuous joint prediction and geometric J

These are archived Qwen3-8B fixed-work Replay measurements, each measured once.
The current release packages the accepted continuous predictor, geometric rule,
exact-state caches, reservation settlement and queue aging. Packaging itself did
not add an end-to-end GPU measurement. Each group below keeps its own GPU,
workload and chronology; rows from different groups are not paired samples.

| Group | Method | Mean JCT (s) | P95 JCT (s) | Makespan (s) |
|---|---|---:|---:|---:|
| common16 / GPU7 | Forest baseline | 87.179 | 140.525 | 143.480 |
| common16 / GPU7 | Continuous joint + geometric J | 88.010 | 136.007 | 139.556 |
| common16 / GPU7 | Continuum | 103.767 | 158.680 | 165.415 |
| baseline32 / GPU2 | Forest baseline | 120.679 | 208.027 | 222.170 |
| baseline32 / GPU2 | Earlier accelerated continuous + geometric J | 119.733 | 214.015 | 241.032 |
| baseline32 / GPU2 | Continuum | 152.397 | 260.451 | 269.080 |
| remaining-work32 / GPU1 | Predicted remaining-work first | 122.982 | 216.781 | 245.282 |
| remaining-work32 / GPU1 | Continuous joint + geometric J | 117.710 | 214.216 | 235.745 |
| cleanup32 / GPU1 | Final continuous joint + geometric J | 116.426 | 209.500 | 232.832 |

The 16-program workload has 178 calls and 26,164 output tokens; its original
arrivals and tool gaps are preserved. It is a historical, intentionally selected
workload, not a new random holdout. Forest and joint arms were run in that order;
Continuum was measured subsequently on the same GPU.

The 32-program workload has 420 calls and 43,928 output tokens, simultaneous
arrivals and zero tool waits. These are complete natural-end trajectories in the
16K domain, without token scaling, but service-isolation Replay cannot establish
tool latency or TTL performance. They had been examined during prior research.
GPU1/7 reuse the recorded GPU2 physical calibration. The GPU2 three-method group
reuses the preselected forest and earlier accelerated continuous runs and adds
Continuum. GPU1 cleanup adds one run against its preselected earlier geometric
reference; no contemporaneous baseline rerun is implied.

The 16-program joint arm improves P95 and makespan over the forest but slightly
increases mean JCT. The GPU2 joint arm improves mean JCT but increases P95 and
makespan. These mixed results are retained, without a universal improvement or
statistical significance claim.

The guard16/guard32 groups retain the measured no-guard/guard pairs:

| Group | Arm | Mean JCT (s) | P95 JCT (s) | Replay wall (s) |
|---|---|---:|---:|---:|
| 16 programs / GPU3 | No guard | 91.434 | 142.530 | 145.135 |
| 16 programs / GPU3 | Historical guard | 88.768 | 137.928 | 145.564 |
| 32 programs / GPU3 | No guard | 120.525 | 219.136 | 247.574 |
| 32 programs / GPU3 | Historical guard | 122.426 | 226.472 | 254.926 |

Both historical guard runs registered the policy but recorded zero activations.
Source review found monotonic time was subtracted from epoch-based arrival time,
so zero activation cannot establish that waits were shorter than four seconds.
The release corrects that clock mismatch and has explicit clock/queue tests.
These historical timings are not proof of starvation relief or measurements of
the corrected guard.

summary.json records all 14 cells from six comparison groups and hashes of the
copied source artifacts. Per-program/request records and audits are included when
present in the local archive. The duplicate geometric reference in two groups is
the same measured run, not an independent repetition. All completion audits use
private fixed-work stopping, public label-independent output limits and
completed-current-action visibility. No new oracle scheduling experiment was run.
