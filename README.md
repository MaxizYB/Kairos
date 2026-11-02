# Kairos

Kairos 面向单 GPU 上交替执行 LLM 调用与工具等待的多轮 Agent Program，联合安排执行机会、跨轮 KV 保护和 GPU/CPU 状态恢复，以降低 Program 完成时间（JCT）。

完整设计、动机、公式、数据分析和实验结果见[方法文档](Kairos_Method_Design_Section_2_ZH.md)。[配套图表](figures/)提供 PDF、PNG 与 SVG；三张方法结构图另有[可编辑 PowerPoint](figures/method/Kairos_Method_Figures.pptx)。

## 方法

两个分布森林分别预测当前输出规模和本轮之后的剩余服务量。Kairos 根据当前布局的物理执行代价与预期任务推进比例，对所有 ready 调用统一排序：

$$
J_i=\frac{t_i}{\mathbb{E}[s_i/(s_i+R_i^+)]}.
$$

独立的容量事务按此顺序检查原生执行前沿与执行名额，维护容量 claim，并仅撤销更低优先级的安全 optional pin。部分保留只作用于本次计划原本需要撤销的 victim，受 claim 和每一步提交余量约束。工具期间，有限 TTL 权衡可观测恢复收益与当前容量机会成本；后端按 GPU/CPU 联合可恢复前缀管理状态，并通过页边界恢复减少重复搬运。

## 代码导读

```text
kairos/
  scheduler.py             完整调度器的类组合与 vLLM 入口
  prediction/              当前输出、后缀服务森林及因果历史
  scheduling/
    snapshot.py            ready/running/idle 的统一观测快照
    score.py               物理服务报价与推进率 J
    priority.py            就绪顺序和连续性位置
    admission.py           claim、grant、安全部分撤销和 idle 物理计价
    retention.py           只读容量计划与保留预算
    forecast.py            当前输出预测与原生 KV 前沿账本
    program.py             Program 历史、异步后缀预测与生命周期
    cache.py               请求级分布与服务报价缓存
    integral.py            running 推进积分的精确复用
  continuity/              工具返回分布与经济 TTL
  backend/                 GPU/CPU 联合覆盖及实际恢复传输
  replay/                  因果请求边界、私有执行停止、完成后观测
  observability/           分布日志压缩与事后展开
  _runtime/                vLLM/LMCache 引用、分配和传输适配
assets/                    两类森林、训练先验和独立服务标定
configs/                   Kairos 与 Continuum 启动配置
data/                      SWE 回放输入、到达序列和离线训练状态
results/benchmark/         九组逐 Program/请求指标与汇总
figures/
  workload/                调用跨度、等待长尾、上下文复用与剩余工作
  prediction/              当前森林与真实观测、区间校准及任务例子
  method/                  三张方法示意图和可编辑 PPTX
  evaluation/              Dense 柱状图与 Poisson 到达曲线
  data/                    图表数据、指标、选例规则与来源哈希
provenance/                来源映射、校验和及验证记录
tests/                     CPU 决策、信息边界、引用和字节恢复测试
```

阅读主流程可从 `scheduler.py` 开始，依次看 `snapshot.py`、`score.py` 和 `admission.py`；TTL 见 `continuity/holding.py`，后端见 `backend/coverage.py` 与 `backend/transfer.py`。运行时中的原生分配、抢占与 DMA 引用边界由适配层维护。

## 本地检查

使用 Python 3.12+ 和轻量虚拟环境；本地检查无需安装 torch 或 vLLM。

```bash
uv venv .venv
source .venv/bin/activate
python -m kairos.verify
python -m unittest discover -s tests -v
python -m kairos.audit
python -m kairos.evaluate
```

测试包含 120 组冻结控制器快照/事务回归、部分撤销与 claim 安全性、公开上限与真实工作量隔离、完成后观测，以及 CPU 模拟的异步字节恢复。`evaluate` 在留出 Program 上评价随附的后缀森林；`audit` 从九组逐程序和逐请求记录重新核对工作量与平均 JCT。

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

固定 Replay 的输出标签仅由私有执行停止适配器使用。调度器在请求到达时接收独立公开上限、当前 prompt 和已完成历史；工具类别及显式结束意图在本轮输出完成后进入观测。训练和事后评价可使用标签。此次仓库整理完成了 CPU 回归与命令检查，整理后的入口尚未重新做 GPU 端到端测量。

## 归档实验

同一张 RTX 4090、Qwen3-8B、16 个 SWE Program、178 次调用。每个方法与负载组合运行一次；下表为平均 Program JCT，单位秒。

| 到达模式 | vLLM+LMCache | Continuum | Kairos |
|---|---:|---:|---:|
| Dense burst | 128.48 | 105.12 | **85.14** |
| Poisson，JPS=1 | 123.79 | 85.35 | **80.02** |
| Poisson，JPS=2 | 122.52 | 97.50 | **83.44** |

Poisson JPS=2 下，Kairos/Continuum 的 P95 JCT 为 141.57/141.34 秒，尾部并非全面领先。以上是集成系统的单次观测；Native 使用同步路径，研究方法使用异步路径。逐组数据见 [results/benchmark/summary.json](results/benchmark/summary.json)，完整解释见方法文档。


