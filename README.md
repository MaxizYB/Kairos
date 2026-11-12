# Kairos

Kairos 面向单 GPU 上交替执行 LLM 调用与工具等待的多轮 Agent Program，联合安排执行机会、跨轮 KV 保护和 GPU/CPU 状态恢复，以降低 Program 完成时间（JCT）。

完整设计、动机、公式、数据分析和实验结果见[方法文档](Kairos_Method_Design_Section_2_ZH.md)。[配套图表](figures/)提供 PDF、PNG 与 SVG；三张方法结构图另有[可编辑 PowerPoint](figures/method/Kairos_Method_Figures.pptx)。

## 方法

轻量文本编码器与已观测数值历史预测当前输出 $O$ 和后续服务 $R$ 的连续联合分布。每个调用在到达时只推理一次，执行过程中根据已生成 $m$ 个 token 更新生存后验；Kairos 在同一后验下计算物理成本与推进比例的几何比值，对所有 ready 调用统一排序：

$$
J_i=\exp\!\left(\mathbb E\!\left[\log\frac{t_i(O)(s_i(O)+R_i^+)}{s_i(O)}
\;\middle|\;H_i,O>m_i\right]\right).
$$

独立的容量事务按此顺序检查原生执行前沿与执行名额，维护容量 claim，并仅撤销更低优先级的安全 optional pin。部分保留只作用于本次计划原本需要撤销的 victim，受 claim 和每一步提交余量约束。工具期间，有限 TTL 权衡可观测恢复收益与当前容量机会成本；后端按 GPU/CPU 联合可恢复前缀管理状态，并通过页边界恢复减少重复搬运。

为改善长期等待请求的用户体验，Kairos 在原有 J 排序之后提供一个队列局部的有界 aging guard：请求等待超过 4 秒后每秒获得一个优先级 credit，credit 最多把请求提升到队首。该 guard 只重排 waiting/skipped-waiting 队列，不修改 J、claim、TTL、victim 选择或容量账本；默认通过 KAIROS_STARVATION_GUARD=1 开启，阈值和量子可由 KAIROS_STARVATION_THRESHOLD_S、KAIROS_STARVATION_QUANTUM_S 配置。

## 代码导读

```text
kairos/
  scheduler.py             完整调度器的类组合与 vLLM 入口
  prediction/              encoder.py 可见特征编码；joint.py 连续密度及生存报价
                           visible.py 因果投影；其他文件维护辅助森林和历史
  scheduling/
    snapshot.py            ready/running/idle 的统一观测快照
    joint.py               到达联合预测、生存更新与几何 J 接入
    score.py               辅助森林报价与参考服务计算
    priority.py            就绪顺序和连续性位置
    admission.py           claim、grant、安全部分撤销和 idle 物理计价
    retention.py           只读容量计划与保留预算
    forecast.py            当前输出预测与原生 KV 前沿账本
    program.py             Program 历史、异步后缀预测与生命周期
    starvation.py          等待队列的有界 aging guard
    ownership.py           原生/DMA/pin 所有权对应的预约结算
    cache.py               请求级分布与服务报价缓存
    integral.py            running 推进积分的精确复用
  continuity/              工具返回分布与经济 TTL
  backend/                 GPU/CPU 联合覆盖及实际恢复传输
  replay/                  因果请求边界、私有执行停止、完成后观测
  observability/           分布日志压缩与事后展开
  _runtime/                vLLM/LMCache 引用、分配和传输适配
assets/joint/              tokenizer/config、权重哈希和外部权重部署说明
assets/                    辅助森林、训练先验和独立服务标定
configs/                   Kairos 与 Continuum 启动配置
data/                      SWE 回放输入、到达序列和离线训练状态
results/benchmark/cg1/      16/32-program 既有实测、排序消融与 guard 对比
results/benchmark/         历史九组逐 Program/请求指标与汇总
figures/
  workload/                调用跨度、等待长尾、上下文复用与剩余工作
  prediction/              辅助森林与真实观测、区间校准及任务例子
  method/                  三张方法示意图和可编辑 PPTX
  evaluation/              Dense 柱状图与 Poisson 到达曲线
  data/                    图表数据、指标、选例规则与来源哈希
provenance/                来源映射、校验和及验证记录
tests/                     CPU 决策、信息边界、引用和字节恢复测试
```

阅读主流程可从 `scheduler.py` 开始，依次看 `snapshot.py`、`score.py` 和 `admission.py`；TTL 见 `continuity/holding.py`，后端见 `backend/coverage.py` 与 `backend/transfer.py`。运行时中的原生分配、抢占与 DMA 引用边界由适配层维护。

## 本地检查

使用 Python 3.12+ 和轻量虚拟环境；本地 CPU 检查仅需 NumPy/SciPy，无需安装 torch 或 vLLM。

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
python -m kairos.verify
python -m unittest discover -s tests -v
python -m kairos.audit
python -m kairos.evaluate
```

测试以合成分布覆盖连续几何报价（无需训练权重）、正式入口、生存更新、精确缓存、队列 aging 与时钟边界，并保留 120 组冻结控制器快照/事务回归、claim 安全性和 CPU 字节恢复。`evaluate` 仍评价辅助后缀森林，不能当作联合预测器的精度报告；`audit` 核对历史九组及 CG1 归档的工作量与 JCT，连续几何事件重算另需外部密度权重。

图表可在本地 CPU 复现，绘图依赖单独安装，不需要 torch 或 vLLM：

```bash
uv pip install -e '.[figures]'
python scripts/build_figures.py
python scripts/draw_method.py
```

第一条脚本使用发布模型重算 SWE 留出预测，并读取随附的匿名聚合工作负载统计；第二条脚本用同一套几何布局生成三页原生 PowerPoint 形状与 PDF/PNG/SVG。PPT 中的文字、色块、页条和连线均可编辑。图表设计参考 CONTINUUM 的长尾分析、到达率曲线及模块分组，以及 CacheBlend 的缓存布局图；来源见 [sources.json](figures/data/sources.json)。预测器只与真实观测比较；区间覆盖率、宽度、Program 等权与调用等权的区别在方法文档中解释。

## 服务器启动

在独立服务器环境中准备 `requirements-server.txt` 对应的 CUDA、vLLM 和 LMCache 依赖。使用源码 checkout（或 editable 安装），模型权重单独存放。当前配置对应 Qwen3-8B BF16、FlashAttention 2、TP=PP=1、GPU KV 5,935,988,736 bytes、CPU KV 10 GiB、最大并发 8、16K 上下文及 2,048 batch tokens。

```bash
# 查看完整命令和环境；不会启动模型。
python -m kairos.run --method kairos \
  --model /path/to/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --output runs/check --dry-run

# Dense 固定工作量回放。
python -m kairos.run --method kairos \
  --model /path/to/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --output runs/dense --gpu 0 --engine-cpus 0-3 --helper-cpus 4-7

# Poisson 首次到达；JPS=2 使用 poisson_jps2.json。
python -m kairos.run --method kairos \
  --model /path/to/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --trace data/workload/poisson_jps1.json --output runs/poisson-jps1
```

`--method continuum` 使用同一回放边界启动 Continuum 对照。`--profile` 与 `--reload-profile` 可提供独立设备标定；随附文件来自归档 RTX 4090 实验，更换设备后需重新标定。运行产物包含启动配置、完成状态、逐请求/Program 指标及决策日志，可用 `python -m kairos.audit --run runs/dense` 检查。

固定 Replay 的输出标签仅由私有执行停止适配器使用。调度器在请求到达时接收独立公开上限、当前 prompt 和已完成历史；工具类别及显式结束意图在本轮输出完成后进入观测。训练和事后评价可使用标签。联合预测代码、配置与 tokenizer 随附于 assets/joint；训练权重和密度二进制不提交 Git，部署时通过 --joint-assets 指定外部目录（参见 assets/joint/README.md）；MiniLM 在服务 GPU 上运行，每个调用仅推理一次。辅助森林继续服务 idle/TTL 与已有资源账目。

本次迁移已对照 20,529 次已测报价重算（最大相对误差 1.33e-15），并在学校服务器用 CPU 严格加载权重、核验打包编码器与原编码器输出一致（误差为 0）。记录见 provenance/joint_quote_replay.json 与 provenance/joint_encoder_check.json。完整发布入口未新增 GPU 端到端重测；下面列出的是已经完成并归档的研究实测。

运行时 activation 和 configs/kairos.json 同时声明有界 aging guard 的默认参数；未显式设置环境变量时，guard 默认开启。

## CG1 实测与归档

Qwen3-8B，固定工作量 Replay，每种方法各一次。表中的两组使用各自同卡对照；32-program 的较早加速实现和最终 GPU1 整理结果分别报告。

| 比较组 | 方法 | 平均 JCT / P95（s）| Makespan（s）|
|---|---|---:|---:|
| 16 programs，GPU7 | 森林基线 | 87.179 / 140.525 | 143.480 |
| 16 programs，GPU7 | 连续联合＋几何 J | 88.010 / 136.007 | 139.556 |
| 16 programs，GPU7 | Continuum | 103.767 / 158.680 | 165.415 |
| 32 programs，GPU2 | 森林基线 | 120.679 / 208.027 | 222.170 |
| 32 programs，GPU2 | 连续联合＋几何 J，较早加速实现 | 119.733 / 214.015 | 241.032 |
| 32 programs，GPU2 | Continuum | 152.397 / 260.451 | 269.080 |
| 32 programs，GPU1 | 最终连续联合＋几何 J | 116.426 / 209.500 | 232.832 |

16-program 保留原到达与工具间隔；32-program 同时到达、零工具等待，是服务隔离负载。几何 J 与预测剩余工作量排序的同 GPU1 对照分别为 117.710 与 122.982 s 平均 JCT；最终整理组复用预先指定的前者作对照。上述均在 guard 之前测量。完整记录、来源、执行顺序与限制见 [CG1 归档](results/benchmark/cg1/README.md)。

## 历史九组归档

这九组使用原森林主预测配置，不代表当前连续联合入口重新测得的成绩。同一张 RTX 4090、Qwen3-8B、16 个 SWE Program、178 次调用。每个方法与负载组合运行一次；下表为平均 Program JCT，单位秒。

| 到达模式 | vLLM+LMCache | Continuum | Kairos |
|---|---:|---:|---:|
| Dense burst | 128.48 | 105.12 | **85.14** |
| Poisson，JPS=1 | 123.79 | 85.35 | **80.02** |
| Poisson，JPS=2 | 122.52 | 97.50 | **83.44** |

Poisson JPS=2 下，Kairos/Continuum 的 P95 JCT 为 141.57/141.34 秒，尾部并非全面领先。以上是集成系统的单次观测；Native 使用同步路径，研究方法使用异步路径。逐组数据见 [results/benchmark/summary.json](results/benchmark/summary.json)，完整解释见方法文档。

有界 aging 的历史 8B 配对数据见 [guard 归档](results/benchmark/starvation_guard_8b/)。16/32-program 两组均完成且零失败。历史 guard 将 monotonic 时钟与 epoch 到达时间相减，导致零激活；不能解释为没有请求超过阈值，也不能证明公平性收益。发布实现已修正时钟并增加队列测试，保留原实验值且注明其适用边界。


