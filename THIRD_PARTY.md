# 第三方来源、证据与发布范围

本目录是 Kairos 的独立源码交付。它不包含模型权重、服务器凭据、原研究工作区或其他方法的完整实验矩阵。没有向远程仓库提交或推送。

## 依赖与来源

- 服务执行依赖 vLLM、PyTorch、LMCache、Transformers；它们不作为二进制依赖打入源码包。版本及安装来源见 provenance/runtime-lock.txt，各自版权与许可仍适用。
- 森林训练依赖 NumPy、scikit-learn 和 threadpoolctl；上线森林遍历只用 Python 标准库，tokenizer 属于服务环境。
- Qwen/Qwen3-8B 的固定 revision、配置/tokenizer/权重哈希在 results/environment.json；下载者单独取得模型并遵循其发布条款。
- 数值训练/评价状态源自公开 teacher 轨迹数据集 tarsur385/qwen3-30b-a3b-instruct-2507-swebench-verified-mini-swe-agent，固定 revision 58389eb5426db5a3ae6159faa37498072b4735ff，来源记录保存在压缩状态包 provenance 字段。工作负载含对应 SWE program 的可见对话、token IDs 与工具时间元信息；不实际执行对话里的命令。

数据集、原项目代码片段与模型的来源不因此变成 Kairos 自有版权。这里保留可复核材料和来源，但没有完成全部上游数据逐项再分发许可审查。公开发布前仍需项目负责人确认许可范围；本交付不擅自选择整个项目的开源许可证。没有新增一个可能错误覆盖上游材料的总 LICENSE。

## 随附上游源码快照

执行适配依赖闭包保留少量源码快照，位置在 kairos/_runtime/vendor/：

| 快照 | 固定 commit | 用途 |
|---|---|---|
| LMCache lazy-offload | 68a9c6b5667ff6d9493530984bd2e40fb3941451 | 进程内延迟 offload 兼容装载与参考源码 |
| SGLang HiCache | 3dc91366ac04b314f856839cb1a76e9a66aac9e7 | 兼容类依赖的源方法审计/提取；不是完整 SGLang 部署 |
| EfficientAgent | b21b12a07a3ccf473c82acf00820b1aa2073044b | 兼容类依赖与源码审计；不是本发布新增的完整对比方法 |

各快照的 PROVENANCE.md 保留上游来源、blob 身份及最初审计说明；这些原始说明不是 Kairos 论文叙事，也不是本发布对其他系统的性能声明。已有 SPDX、版权标头和随附 LICENSE 原样保留。未为未附完整许可的快照编造授权。

冻结兼容闭包含未启用分支，因此仍可看到内部标识与参考实现；对外运行器只开放 Kairos/Continuum 两个配置，不要求安装 SGLang 或部署其他比较系统。不要把“可 import 某兼容类”说成复现了该上游系统。

## 原始证据的保留方式

results 下无损压缩日志保留原始事件名、模型路径与工具对话，便于独立复算。路径来自来源元信息，不作为默认连接地址或运行目录；它们不是服务器凭据。不会为了表面去历史化而改写对话或数值证据、改变模型哈希身份。

面向读者的 README、方法、架构、实验说明只阐述 Kairos 与 Continuum；公开配置采用两种方法名，内部执行别名集中在 runner 中。provenance/export_manifest.json 记录最初导出映射；release_manifest.json 记录最终发布文件及与导出版本的差异。它们不能替代迁移后的 GPU 正确性验证。
