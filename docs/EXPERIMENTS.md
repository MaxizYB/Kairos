# Kairos 实验设置、结果与证据

本页只比较 **Kairos 与 Continuum**。所有数字来自随仓库保存的测量，不将重命名后的代码整理冒充新 GPU 实验。证据日期为 2026-10-26，每方法一次正向回放，没有重复试验置信区间。

## 1. 任务与工作负载

发布工作负载名为 `dense_return_burst_production_v1`：从公开 SWE teacher 轨迹取完整、自然结束且上下文可容纳的串行程序；保留逐轮输入与输出预算、历史结构和工具类型，不执行轨迹中的 shell 命令。

为制造首轮与续轮竞争，构造器按 program 身份把首次到达分成约 0.4 秒与 2.8 秒两组，并用独立 profile 回推首个工具等待，使其目标返回窗口靠近 4.8 秒。真实运行的首轮结束由实际服务决定，因此不保证所有返回真的在同一时刻。只改首次到达及首个工具间隔，后续提交均为“前一 call 完成 + 工具间隔”。它是用于机制开发的构造压力负载，不能称为自然生产到达，也不是未参与调优的盲测。

| 属性 | 发布值 |
|---|---:|
| Programs / calls | 16 / 178 |
| 每 program 调用数 | 7–15，均值 11.125 |
| 完成的 output tokens | 26164 |
| 每 call output tokens | 39–699，均值 146.99 |
| 最终上下文 | 6056–11455 tokens，均值 8003.56 |
| 最终上下文总和 | 128057 tokens |
| 工具间隔（改写后实际输入） | 162 个，均值 0.5626 秒，标准差 1.0189 秒，最大 6.4765 秒 |

`trace.json` 的 `measured_fixture`、`paper_reference` 和 `assumptions` 部分保留了构造来源的信息，有些统计在后续 burst 改写之前生成，有些资源描述属于生成器默认值；**正式输入是 `programs` 数组，正式资源以 configs 与 environment 为准**。不要把文件头中的旧模型路径、6.75 GiB/6 GiB 等生成器元信息误当成本次 8B 配置。输入 token IDs 在 8B tokenizer 上逐请求核对：178/178 一致，记录在 [tokenizer-verification.json](../provenance/tokenizer-verification.json)。上表工具统计已从改写后的非终止轮 `programs[*].turns` 重算，不使用文件头未更新的均值。

当前 call 的 output budget 在到达时对两方法公开，`ignore_eos=true`，固定生成同样 token 数。下一轮 prompt 来自固定轨迹，不接入本次生成的回答；因此这是有真实 GPU 计算与 KV 传输的 **teacher serving replay**，不是自主 agent 成功率实验，也不验证未知 EOS。

## 2. 硬件、模型和共同配置

| 项 | 固定配置 |
|---|---|
| GPU | 单卡 NVIDIA GeForce RTX 4090，报告总显存 24564 MiB，独占回放 |
| 模型 | Qwen/Qwen3-8B，BF16，revision `b968826d9c46dd6066d109eabc6255188de91218` |
| GPU KV | 原生自动分配；`gpu_memory_utilization=0.9`，未设置 KV 字节硬上限；实测 2524 页 |
| KV 布局 | 36 layers、8 KV heads、head_dim 128、BF16；147456 bytes/token |
| GPU page / CPU chunk | 16 / 256 tokens |
| CPU KV | 8 GiB，lazy MP helper，初始申请 4 GiB，可增长至上限 |
| 最大上下文 / seqs / batched tokens | 16384 / 8 / 2048 |
| 算法与数值 | APC；priority；async scheduling；FlashAttention 2；seed 1 |
| 图与编译 | compilation mode 0；FULL_DECODE_ONLY；capture sizes 1,2,4,8；warmups 2 |
| CPU 亲和 | Engine `4-5,68-69`；helper `6-7,70-71`；不相交 |
| 线程 | OMP/MKL/OpenBLAS 各 1；OMP_WAIT_POLICY=PASSIVE |
| 服务环境 | Python 3.12；torch 2.13.0+cu132 / CUDA build 13.2；vLLM 0.28.0；LMCache 0.5.5rc4；Transformers 5.16.1；NumPy 2.2.6 |

CPU 具体型号、驱动完整版本、内核及逐个 wheel 的安装文件哈希没有完整归档，不补造这些字段。完整已安装包清单在 [runtime-lock.txt](../provenance/runtime-lock.txt)，这是一份环境 freeze，并非自包含的环境镜像。模型所有配置、tokenizer 和五个权重 shard 的大小/哈希在 [environment.json](../results/environment.json)。

两方法使用相同模型、输入、输出合同、硬件、GPU/CPU 容量、成本 profile、传输后端与 warmup/correctness 流程。Continuum 是本仓库提供的兼容执行实现：pin-first priority、经验工具经济 TTL 和相同基础引擎，不宣称复现作者在不同引擎和硬件上的全部论文数字。两方法的上层容量、TTL 和缓存策略本身不同，因此这是 **完整系统对比，不是只替换 scorer 的严格消融**。

## 3. 端到端结果

程序 JCT 从该 program 第一轮实际到达到最后一轮完成，包含内部工具间隔、排队、计算与恢复；不包含程序到达前的人类思考时间。P95 用排序后 `(N−1)×0.95` 线性插值。Wall 是所有任务的整体回放计时，不包括启动、warmup、正确性 fixture 与独立成本标定。

| 指标 | Continuum | Kairos | 变化 |
|---|---:|---:|---:|
| Mean program JCT/s ↓ | 106.472 | 93.597 | 降低 12.09% |
| P95 program JCT/s ↓ | 150.118 | 136.181 | 降低 9.28% |
| 总回放时间/s ↓ | 161.167 | 150.258 | 降低 6.77% |
| Output tokens/s ↑ | 162.341 | 174.127 | 增加 7.26% |
| Mean request TTFT/s ↓ | 4.992 | 3.748 | 降低 24.92% |
| P95 request TTFT/s ↓ | 20.176 | 25.814 | 增加 27.94% |
| Mean request E2E/s ↓ | 9.056 | 7.898 | 降低 12.78% |
| P95 request E2E/s ↓ | 26.640 | 29.373 | 增加 10.26% |
| Mean ITL/ms ↓ | 27.827 | 28.423 | 增加 2.14% |
| P95 ITL/ms ↓ | 38.301 | 52.755 | 增加 37.74% |
| 完成 / 失败 calls | 178 / 0 | 178 / 0 | 相同 |

ITL 按客户端相邻单 token 流式 chunk 的间隔统计；不是逐 GPU kernel 时间。不同批组合和并发也会影响客户端尾部，不能只看平均 JCT 就宣称所有延迟都改善。

16 个 program 中 13 个 JCT 改善，3 个变差。例如 `pytest-dev__pytest-7521` 从 114.037 秒降至 53.205 秒，而 `sympy__sympy-17318` 从 74.140 秒增至 126.654 秒。完整逐程序 CSV 由 `python scripts/report.py` 生成到 [program_jct.csv](../results/program_jct.csv)，不筛掉退化程序。

## 4. 机制观测与可支持的解释

| 观测 | Continuum | Kairos |
|---|---:|---:|
| Prefill query tokens（日志累计） | 183846 | 151393 |
| H2D / D2H GiB（日志累计） | 6.574 / 37.512 | 10.793 / 19.020 |
| Native preemptions | 9 | 0 |
| Pin 建立次数 | 144 | 161 |
| Pin 引用 page-seconds | 61321.34 | 24218.56 |
| Nominal TTL mean/s | 0.528 | 1.070 |
| 记录的 decode batch mean | 4.090 | 4.934 |

Nominal TTL 更长但引用驻留积分更小并不矛盾：压力提前释放、返回后排队豁免、实际推进速度共同决定真实驻留。引用 page-seconds 可能含共享页多 owner 计数，不等于独占物理 HBM 用量积分。Decode batch 均值按记录事件算，不是时间加权吞吐。

观测与“稳定调用承诺减少 emergency preemption、互补恢复降低重算与重复写回、分布推进改善成熟续轮次序”的设计方向一致，但不能据此分别量化三个组件的因果贡献。H2D 增大、部分单请求尾延迟变差也是真实代价。

日志审计进一步确认：162 次后缀预测、288 次带报价的决策事件，1700 个候选报价可从当时布局独立复算；52 个不同 call 的预测实际用于比较；284 个事件改变了候选内部次序，其中 126 个事件伴随 grant。**排序改变不等于新次序每一次都会改变最终运行者**，仍受容量和 native 执行类别约束。162 个 TTL 决策也逐个复算通过。

归档累计派发细化用时 2.973 秒，异步预测 worker 5.818 秒，其中 forest inference 1.804 秒；这些是各次观测累加，不是可直接从 wall 中减掉的串行开销，也不包含完整控制器所有工作。

独立正确性 fixture 包含连续接续、leading hole、离散 GPU 页、交错两层和部分页恢复五类布局；两方法均保存结果。Kairos 的五组 greedy IDs 一致，最大抽样 logprob 差 0.03552，低于 BF16 fixture 阈值 0.1。本配置不会执行任意缺口 attention，fixture 的 `saved_queries=0` 不应被包装为分段 attention 加速。

## 5. 森林的离线预测结果

标准工作量单位来自冻结参考 profile，不是本次 8B wall 秒。表中 CRPS 与均值 MAE 按 program 宏平均；90% 区间为分布分位区间，不是 JCT 置信区间。

| 评价域 | Programs / states | CRPS ↓ | Mean MAE ↓ | 90% coverage | 区间宽度 |
|---|---:|---:|---:|---:|---:|
| 三折 program-disjoint OOF | 157 / 2106 | 5.1112 | 7.7041 | 93.84% | 30.6082 |
| 独立训练 ID 的保留域 | 36 / 401 | 3.7184 | 5.9274 | 97.40% | 30.0028 |
| 同域开发集 | 14 / 161 | 3.0578 | 4.7610 | 99.49% | 31.0259 |

保留域已在方法研究中被查看，不再宣称未触碰的盲测。同域开发集也不能代替新测试集。保留域 cold 同阶段配对准确率仅 50.63%，warm 为 67.91%；这支持“不让首轮个体预测主导成熟续轮”的保守边界，不证明所有未知域同样有效。覆盖率偏高且区间较宽，不能称为精确尾部预测。

纯森林数值 inference 的独立 CPU benchmark 为 243 次、median 0.502 ms / P95 0.683 ms，未包含 tokenizer、特征、目录 RPC 或完整上层控制器，且不是本次正式 GPU 回放的单次开销口径。

## 6. 证据与复算

- [comparison.json](../results/comparison.json)：端到端和缓存/批大小摘要。
- `results/{kairos,continuum}/requests.jsonl`、`jobs.json`：逐请求和逐 program 客户端记录。
- `scheduler.jsonl.gz`、`replicas.jsonl.gz`、`transfers.jsonl.gz`：无损压缩原始服务端证据；原始内部标签不是额外公开方法。
- [predictor.json](../results/predictor.json)：完整 OOF/域内外/阶段指标；[mechanism.json](../results/mechanism.json)：机制汇总与长等待实例。
- `provenance/`：输入、模型、原始源文件和发布文件校验关系。

执行 `python -m kairos.audit` 可从原始 jobs/request 与决策日志复算主结果、全部已消费报价及 TTL；`python -m kairos.evaluate` 复算保留域和开发集指标；`python -m kairos.train --cross-validate ...` 重建 OOF。命令与完整配置见 [REPRODUCE.md](REPRODUCE.md)。

## 7. 能与不能得出的结论

这次固定压力回放展示了程序平均及 P95 完成时间收益，并证明归档中的预测确实进入了受约束派发流程。它没有证明：统计显著性、所有负载普遍优于 Continuum、全局 JCT 最优、cold 无饥饿、未知输出长度能力、真实 agent 成功率或各组件单独贡献。独立包迁移后的 GPU 运行仍待有卡时复验；本次交付完成的是源码/材料整理与 CPU 可核验性。
