# Kairos 复现指南

本仓库支持三层复核：标准库结果/公式核验、轻量 CPU 森林重建、有 GPU 时的端到端复跑。此发布只完成前两层本地核验，没有重新进行独立包的 GPU 回放。

以下全部从 Kairos 仓库根目录运行。只需这个目录，不需要原研究仓库或旧服务器路径。模型权重不随仓库分发。

## 1. 离线检查发布材料

Python 3.12+ 标准库即可，无需 torch、vLLM 或 GPU：

~~~bash
python -m kairos.verify
python -m unittest discover -s tests -v
python -m kairos.audit
python -m kairos.evaluate --output /tmp/kairos-evaluation.json
python scripts/report.py
python -m kairos.run --method kairos --model /path/to/checkpoint --output runs/dry --dry-run
python -m kairos.run --method continuum --model /path/to/checkpoint --output runs/dry --dry-run
~~~

evaluate 的输出文件必须不存在；重跑时换名字或省略 --output 查看 stdout。report.py 重新生成 results/README.md 与 program_jct.csv，不修改归档请求与日志。

verify 检查整个源码 checkout 与冻结资产的 SHA256；audit 从逐程序结果验证 16 个 program、178 个 call、26164 个输出 tokens 和主指标，并逐条复算 1700 个报价、162 个 TTL，检查消费的 request ID。它证明归档证据内部一致，而非证明所有分支没有 bug 或方法普遍有效。哈希用于检查意外修改，不是第三方签名的真实性证明。

evaluate 重算保留域与开发域的所有子组指标。tests 还检查冻结预测探针、分布指标、CPU 洞口、恢复完成不重复计费、成熟槽位和过期预测边界。

运行时代码没有访问父目录研究代码的静态 import。数据、profile 与无损原始日志保留的机器路径是来源元信息，不是运行依赖；不要按这些元信息连接任何服务器。运行路径来自 CLI 与配置模板。

## 2. CPU 重建森林和 OOF

独立轻量环境，不装服务器依赖：

~~~bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements-train.txt
source .venv/bin/activate
python -m kairos.train --output runs/forest-rebuild
python -m kairos.train --cross-validate --output runs/forest-oof
~~~

重建使用 Python 3.14.7、NumPy 2.5.3、scikit-learn 1.9.1、threadpoolctl 3.7.0。不同版本可能产生不同树结构，差异未解释时不应宣称逐字节复现。输出目录必须不存在，训练程序不会覆盖已有结果。

输入 data/training/causal_states.json.gz 保存 127 维因果数值特征及标签；筛选完整、16K 域内 train programs，得到 157 个 program / 2106 个状态。重建 48 棵 program-honest CDF 树，depth=7、min_leaf=24、max_features=0.8、seed=20261020。CLI 不下载轨迹。

training_report.json 的 identical_to_bundled 应为 true。冻结森林 SHA256：

~~~text
cb3d0859f7c837bb4aa719cc2aaa3ca5e8ceec40fa593cd14ea55a286bab1274
~~~

OOF 在 program 层固定三折，任何 program 的状态不会跨 train/test，独立重训每折。oof_metrics.json 应复现 CRPS=5.1111628181、均值 MAE=7.7040795258、coverage90=0.9383705433、width90=30.6082291227。固定分组哈希盐也属于复现协议，不应仅为改名而修改。

训练可从数值状态重建；**从原始公开数据下载到重新生成全部数值状态的流水线不在本发布范围内**。可见消息、特征函数、逐到达协议与数值状态均已提供，但不承诺一键重新采集原始 teacher 轨迹。来源与再分发边界见 [THIRD_PARTY.md](../THIRD_PARTY.md)。

## 3. GPU 服务器环境

需要空闲专用 RTX 4090 24 GiB、Linux、支持对应 CUDA build 的 driver、g++ C++17、taskset、nvidia-smi。无 GPU 开发机不要执行本节。使用独立 .venv，不改系统包或其他人的进程。

记录的环境：Python 3.12、torch 2.13.0+cu132、vLLM 0.28.0、LMCache 0.5.5rc4、Transformers 5.16.1、NumPy 2.2.6、httpx 0.28.1。先安装 CUDA 13.2 torch，再安装主要依赖：

~~~bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install 'torch==2.13.0+cu132' --index-url https://download.pytorch.org/whl/cu132
uv pip install -r requirements-server.txt
uv pip check
~~~

索引是获取对应 CUDA build 的安装示例，未封存 wheel，整理时没有重新验证下载端点可用性。实测 wheel 来源及全部包版本保存在 [runtime-lock.txt](../provenance/runtime-lock.txt)。若解析冲突，按该快照重建并检查，不要随意升级引擎后直接比较。freeze 包含无关预装组件，**不是推荐不加判断全量安装的 requirements**。适配只作用于自身进程，不修改安装目录中的 vLLM/LMCache 文件。

安装后先检查：

~~~bash
python -c 'import torch, vllm, lmcache, transformers; print(torch.__version__, torch.version.cuda, vllm.__version__, transformers.__version__); assert torch.cuda.is_available()'
g++ --version
nvidia-smi
~~~

启动前要求至少 40 GiB 可用 host RAM；运行时检查至少剩余 8 GiB、归属进程 RSS 不超过 32 GiB、GPU 不出现外部 compute PID。原生 GPU KV 自动剖析，归档为 2524 页；不要人为用命令行强改成这个数。自动容量不同应先记录环境差异，不能只凭模型同名就断言公平。

## 4. 固定模型和文件核对

使用 Qwen/Qwen3-8B 固定 revision b968826d9c46dd6066d109eabc6255188de91218。在线服务器可用已有 HF 工具下载：

~~~bash
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --cache-dir models/hub
~~~

运行使用真实 HF snapshot 目录，形如 models--Qwen--Qwen3-8B/snapshots/REVISION。不要用任意目录别名：后端要将完全相同的模型身份用于 vLLM 与 CPU 对象哈希，本发布对布局白名单做了保守限制。

~~~bash
export KAIROS_CHECKPOINT="$PWD/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
python -m kairos.verify --model "$KAIROS_CHECKPOINT"
~~~

默认逐个验证配置、tokenizer 与五个权重 shard 的大小和 SHA256，共约 16.4 GB 读取。--metadata-only 可快速核对小文件，但不是完整权重验证；正式运行会再次全量验证。离线回放设置 HF/Transformers offline，不再下载模型。

## 5. 逐方法独立复跑

保持两个方法相同模型、成本、CPU 配置，串行各跑一次，不要求反向重复。没有记录的 CPU 编号时，两个方法选择相同的不相交 CPU 集合，并在报告中说明环境变化。

~~~bash
python -m kairos.run --method continuum --model "$KAIROS_CHECKPOINT" --gpu 0 --engine-cpus 4-5,68-69 --helper-cpus 6-7,70-71 --output runs/continuum
python -m kairos.run --method kairos --model "$KAIROS_CHECKPOINT" --gpu 0 --engine-cpus 4-5,68-69 --helper-cpus 6-7,70-71 --output runs/kairos
~~~

完整启动与环境模板在 [continuum.json](../configs/continuum.json) 和 [kairos.json](../configs/kairos.json)。runner 会把这两个公开方法名映射到冻结执行栈所需的内部兼容设置（见 `kairos/_runtime/activation.py`）；这些设置属于实现协议，不是额外公开方法或需要用户搜索的参数。输出目录必须不存在，端口 18765–18768 必须空闲。不会复用或清空不属于自己的 helper。

每次运行依次：

1. 校验发布与模型，检查 GPU、RAM、端口、CPU。
2. 创建独立 helper 与 engine 进程组，等待 readiness；预加载森林静态资源。
3. 相同热身及五组恢复正确性 fixture；排空并清空本次 helper；正式切换清理 warmup 状态。
4. 固定程序按时到达，客户端只提交本轮可见消息，使用相同输出合同及工具间隔。
5. 保存 requests、jobs、summary、scheduler/transfer/replica 日志、metrics、资源与启动参数。
6. 完成或报错仅终止本次创建的进程组；不删除模型或其他结果。

异常时 state.json 标记 failed，未完成回放不作正式性能点。启动 timeout 2400 秒、单请求 120 秒、正式回放 900 秒；启动/编译不计入性能计时。dry-run 只展开命令，不验证依赖、GPU、权重和端口；不要通过关闭 guard 获取不可比数字。

## 6. 成本标定、超参与配置含义

reference_profile.json 固定定义预测标准单位。换机器不重定义它，否则森林标签与当前 s 混用单位。deployment_profile.json 与 reload_profile.json 表示实际部署成本，两方法共用：

| 项 | 值 |
|---|---:|
| Prefill β0 / β1 / β2 / β3 | 0.0107894428 / 0.0000907022633 / 4.63651559e−9 / 0.00258494963 |
| 单序列 decode step | 0.0184442234 秒/token |
| H2D fixed / per-token | 0.000389696421 秒 / 0.00000601064725 秒 |
| 完整调用安全余量 | 32 GPU pages |
| TTL 类别 live 样本保留 | 最多 512 条 |
| 文本特征截取 | 首尾各 8192 字符 |

β 为 chunked prefill 模型；H2D 是无排队 CPU hit 相对 GPU hit 的增量 E2E，含 lookup/scatter 等开销，不是纯 PCIe 带宽。profile 保留原始样本及拟合说明，标定在正式回放前独立完成。

新硬件需要独立标定部署/恢复成本，并给两方法同样的新成本；可以用 --profile 和 --reload-profile 指定。新 profile 生成器与跨平台数值验证未包含在固定实验发布中，不宣称已完成一键跨硬件复现。

TTL 的 paper 兼容字段不是固定 2 秒：实际 policy 使用工具 CDF、恢复/等待收益及当前压力选择 deadline，见方法文档。其他兼容变量是冻结实现协议，不是需要遍历搜索的公开超参。

## 7. 校验值与结果解读

全部文件哈希在 provenance/release_manifest.json；核心资产：

| 文件 | SHA256 |
|---|---|
| assets/forest.json | cb3d0859f7c837bb4aa719cc2aaa3ca5e8ceec40fa593cd14ea55a286bab1274 |
| assets/reference_profile.json | c8e3e539858277bf5340e91fb36213379743eb2e0933f89aa91d67baa981be87 |
| assets/deployment_profile.json | a08d9e3b28095bc20366a3a7dfe9b7578c514b0b4cbebb60be048bf75c73e022 |
| assets/reload_profile.json | 5238b9cc074cef7794c8665ca05f032ffbc236d8e24f499e8cf3ae62cab4b7d2 |
| data/workload/trace.json | 4c896b93b14eff39bcc6e2fd0700ba59a1bfcfd785bd15e645d5e1a2ffeee2e8 |
| data/workload/visible_messages.json | 59deb05bfffb29825ee474f17376e4b6c9a23056b473cec061fc3b7123e81a1a |
| data/training/causal_states.json.gz | 9c04b4f5655269e4a76717b0c6efd6693b894736ab8612041c55d1ca1a955995 |

逐到达协议保留了之前生成阶段的 trace 源哈希，并非当前发布文件哈希。比较以同一 request signature、当前文件哈希和输出工作量为准，不混用不同阶段的完整 JSON hash。

重跑无需逐毫秒等于归档值。先检查 178/178、0 失败、26164 输出 tokens、模型/输入/配置匹配、无外部 GPU PID，再解释资源与延迟变化。一次回放不能保证任意环境容差内必胜，不应只保留最好的一次。

维护者修改后重新测试，再显式运行 python scripts/seal_release.py --write 封装新校验清单；这不是 GPU 正确性或性能认证。
