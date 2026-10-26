# Kairos 与 Continuum：已归档实验

单卡 RTX 4090、Qwen3-8B；16 programs、178 calls。每方法一次正向回放。

| 方法 | Mean JCT/s ↓ | P95 JCT/s ↓ | Wall/s ↓ |
|---|---:|---:|---:|
| Continuum | 106.472 | 150.118 | 161.167 |
| Kairos | 93.597 | 136.181 | 150.258 |

改善幅度：mean_jct_s: 12.09%, p95_jct_s: 9.28%, wall_s: 6.77%

逐程序完整数据见 program_jct.csv；详细配置、指标口径与局限见 ../docs/EXPERIMENTS.md。
这是归档测量结果，不是对重命名后代码新做的 GPU 验证。
