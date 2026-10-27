# Kairos：在正确的时机推进 Agent 程序

**10.27 注：当前版本有巨大设计缺陷，正在修复中**

Kairos 面向单 GPU 上交替执行 LLM 调用与工具等待的 agent program。它用因果后缀分布衡量当前调用的预期推进比例，以实际 GPU/CPU KV 前缀报价执行成本，并在完整调用容量承诺内安排成熟续轮。

核心指标是：

$$
B_i=\mathbb E\!\left[\frac{s_i}{s_i+R_i^+}\right],\qquad
J_i=\frac{t_i}{B_i}\quad\text{（越小越先）}.
$$

其中 s 是当前标准工作量，R⁺ 是当前轮之后的标准服务量分布，t 是当前布局下的恢复、重算及输出成本。它是局部推进指标，不是整个程序 JCT 的预测或最优性保证。

## 从哪里开始

- [完整方法](docs/METHOD.md)：文字流程、公式、森林训练、调度、容量承诺、TTL 与两层 KV。
- [架构与代码导读](docs/ARCHITECTURE.md)：每个模块做什么，以及哪里改、哪里不改。
- [复现指南](docs/REPRODUCE.md)：CPU 核验、训练重建、固定模型与完整 GPU 回放命令。
- [实验设置、结果和局限](docs/EXPERIMENTS.md)：只报告 Kairos 与 Continuum。
- [第三方来源与发布范围](THIRD_PARTY.md)。

## 已归档的端到端结果

Qwen3-8B / 单卡 RTX 4090 24 GiB / CPU KV 8 GiB；16 programs、178 calls、26164 output tokens。每方法一次正向回放，均 178/178 完成、0 失败。

| 方法 | Mean program JCT/s ↓ | P95 program JCT/s ↓ | 总回放时间/s ↓ |
|---|---:|---:|---:|
| Continuum | 106.472 | 150.118 | 161.167 |
| Kairos | 93.597 | 136.181 | 150.258 |
| 降低 | 12.09% | 9.28% | 6.77% |

单请求 P95 TTFT 和 P95 E2E 并非全面改善，详见实验文档。本结果是已测实现的归档证据，不是重命名独立包后重新进行的 GPU 测量；没有多次回放置信区间，也不是未参与开发的盲测。

## 无 GPU 也能先验证

从仓库根目录运行，只需 Python 3.12+ 标准库：

~~~bash
python -m kairos.verify
python -m unittest discover -s tests -v
python -m kairos.audit
python -m kairos.evaluate
python scripts/report.py
python -m kairos.run --method kairos --model /path/to/checkpoint --output runs/check --dry-run
~~~

训练重建需要轻量 NumPy/scikit-learn；GPU 回放需要独立的服务器环境。不要在无 GPU 的开发机上安装服务端依赖。模型权重不包含在仓库中。

## 仓库布局

~~~text
kairos/              易读的评分、分布预测、训练、保护与复现入口
  _runtime/          冻结的执行/引用/传输适配层，非第二套研究项目
assets/              冻结森林、标准单位、部署和恢复成本、工具先验
configs/             两方法完整命令与环境参数模板
data/                固定回放输入、逐到达可见消息、数值训练/评价状态
results/             两方法逐请求/逐程序结果、压缩决策日志与汇总
provenance/          文件校验和、已测源文件校验和、独立包验证记录
tests/               不依赖 torch/vLLM 的回归与等价检查
docs/                方法、架构、复现和实验
~~~

当前交付使用源码 checkout（可 editable 安装），不是包含模型、数据和部署环境的自包含 wheel。运行仅支持本仓库发布的两种配置；内部兼容标识不是对外实验版本列表。
