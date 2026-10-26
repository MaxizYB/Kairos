# Kairos 架构与代码导读

Kairos 将三个不同问题分开处理：**未来还有多少标准工作、此刻执行要花多少成本、哪些资源已经作出承诺**。森林回答第一个问题，物理报价回答第二个问题，容量与引用协议约束第三个问题。评分不能替代真实分配，预测也不能撤销已授权调用。完整公式和流程见 [METHOD.md](METHOD.md)。

## 1. 从入口到执行

```text
固定输入与逐到达可见消息
          │
  run.py / teacher_replay.py ── 当前调用、输出合同 ─┐
          │                                      │
          └── 历史快照 → arrival.py → forest.py ───┤
                              后缀概率分布        │
                                                 ▼
                       scheduler.py：读取当前 GPU/CPU 前缀
                       score.py：算 B、t、J=t/B
                                                 │
                       dispatch.py：仅细化成熟有效槽位
                                                 │
                 容量账本 → grant → native continuous batching
                                                 │
                           connector / worker / CPU helper
                                                 │
                  调用完成 → 经济 TTL → optional pin
                                                 │
                              下一轮实际到达后重复
```

这不是每个 token 都运行的全局规划器。原生引擎按步运行；上层只在状态 dirty 的事件上竞争，森林在到达时异步计算。

## 2. 面向读者的核心文件

| 文件 | 职责与阅读重点 |
|---|---|
| [scheduler.py](../kairos/scheduler.py) | 具体调度类；一次读 CPU 目录，逐候选查 HBM，拼接标准工作与物理成本 |
| [score.py](../kairos/score.py)、[cost.py](../kairos/cost.py) | 离散分布积分、单位推进成本、连续前缀恢复报价；均可在 CPU 单测 |
| [dispatch.py](../kairos/dispatch.py) | request-ID 校验、异步结果收集、成熟槽位细化、保留独立 victim 顺序 |
| [arrival.py](../kairos/arrival.py)、[resources.py](../kairos/resources.py) | 因果快照、历史重建、tokenizer/森林预加载、标准服务量 |
| [forest.py](../kairos/forest.py) | 纯 Python 树遍历和叶分布聚合；上线不导入 sklearn |
| [features.py](../kairos/features.py)、[causal.py](../kairos/causal.py) | 127 维特征以及冻结标准工作单位 |
| [train.py](../kairos/train.py)、[training_algorithm.py](../kairos/training_algorithm.py) | program 级分组、honest CDF 树、OOF、冻结工件重建 |
| [protection.py](../kairos/protection.py) | 独立经验连续性包络、工具 CDF、TTL 断点扫描 |
| [run.py](../kairos/run.py) | 两方法独立启动、热身/正确性、资源守护、固定回放、仅清理自身进程 |
| [verify.py](../kairos/verify.py)、[audit.py](../kairos/audit.py) | 文件/模型校验；从原始结果与日志独立复算指标、报价和 TTL |

从核心理解方法通常只需先读 scheduler → score → dispatch → arrival → protection。无需先阅读整个引擎兼容层。

## 3. 执行适配层为什么单独冻结

`kairos/_runtime/` 是随仓库提供的执行依赖闭包，不要求访问原研究工作区。保留这些依赖是为了不在无 GPU 的整理阶段大改已测执行协议。它比核心算法大，也包含未启用分支；并不是已经压缩为最小实现的全新引擎。公开入口只开放 `kairos` 和 `continuum` 两种配置，其他分支不属于本发布的支持范围。

| 文件组 | 正式路径的作用 |
|---|---|
| `compat/commitment_scheduler.py`、`probation_scheduler.py` | 成熟 cohort、完整调用 grant、cold 窗口；不将上层 grant 等同于 HBM 已分配 |
| `compat/online_path_scheduler.py`、`online_path.py` | 连续性包络、owner rank、optional 支持及事件状态 |
| `compat/pipelined_prefix.py`、`scheduler.py` | 原生调度衔接、真实分配后引用交接、异步 capacity fence；Continuum 入口也在此 |
| `layout/connector.py`、`execution_route.py`、`worker.py` | 可执行连续前缀恢复、传输完成与计算的同步、诊断 |
| `layout/runtime.py`、`pcv_runtime.py`、`compat/lmcache_replica.py` | 私有 CPU helper，副本目录、真正容量不足时的回收与 STORE |
| `compat/joint_eviction.py`、`layout/next_use.py` | 合法 free GPU 页的联合恢复边界损失，不能驱逐正在使用的强引用 |
| `compat/replica.py` | RPC、身份与租约元数据；本发布只支持固定 Qwen3-8B KV 布局 |
| `compat/online_path_native.cpp` | 独立经验包络的 C++ 辅助计算；不是森林、不读本次真实未来 |

内部字段和环境变量保留兼容标识，以免改变冻结协议；不把这些标识当作论文模块名称或公开版本列表。分布预测与当前报价的读者接口已经独立整理。

## 4. 四种状态不要混淆

- **预测就绪**：这次 request 的分布已经算完，能参与成熟槽位细化；未就绪不阻塞准入。
- **grant**：完整当前调用获得逻辑容量与名额承诺，后来的重排不能撤销。
- **optional pin**：工具期为连续性保留的可撤销引用；可因到期或合法压力释放。
- **native/DMA 引用**：实际执行或传输仍需要的数据，不能当 optional pin 直接回收。

同一 program 可以“有预测、无 pin”；也可以“已 grant、仍在 native waiting”。排序、资源权限和实际运行阶段不由一个布尔值代表。

## 5. 修改与复核边界

核心算分修改先运行轻量测试和离线指标；修改 grant、引用、恢复与 cleanup 协议则必须重新在 GPU 上做正确性和端到端测试。本仓库整理时没有重启服务器：源码迁移、路径参数化、文件与模型检查只获得本地验证，不冒充迁移后 GPU 实测。

`provenance/export_manifest.json` 保留导入文件与原始来源的哈希关系；`release_manifest.json` 覆盖整个交付。原始日志中的内部事件名按证据保存，方法解释只采用上面的公开命名。
