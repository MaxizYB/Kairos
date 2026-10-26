# Kairos 完整方法：概率推进、物理报价与稳定调用承诺

本文描述发布代码实际启用的方法。主线为：请求到达 → 因果后缀预测 → 当前物理成本报价 → 成熟续轮槽位排序 → 完整调用容量授权 → 原生执行与联合 KV 恢复 → 工具期经济保护 → 下一轮到达。

## 1. 问题与总体设计

Agent program 会反复交替执行 LLM 和工具。单次 LLM 调用结束不是整个 program 结束，暂时空闲的 KV 也不一定失去价值。Kairos 需要同时面对三类困难。

第一，刚到达的 program 缺少辨识未来工作量的历史；即使两个可见前缀相同，它们的后续轨迹也可以不同。不能让缺乏个体证据的首轮预测主导所有候选。

第二，剩余任务规模与本次执行成本不同：同样的任务，已有 HBM 前缀、仅有 CPU 副本、需要重算，价格并不相同。单纯按剩余 tokens 排序无法表达这些差别。

第三，排序改善不能以撤销已执行调用的资源为代价。预测是建议，grant 是当前完整调用的承诺，pin 是工具期可选保护；三者必须各守边界。

因此系统分成三层：信息层输出后缀分布；决策层比较预期推进与当前价格，仅细化可比较候选；执行层用容量账本、有限保护和安全两层恢复兑现被选动作。不构造所有 program 的未来交互 rollout，也不假定能够预测未来排队。

## 2. 对象、信息与单位

program 是一个用户目标；call 是其中一次 LLM 请求。当前工作负载每个 program 串行提交 call，前一 call 完成并经历工具间隔后才提交下一 call。

成熟请求指已有至少一次完整输出观测的续轮；cold 指尚未完成首轮。成熟不等于有 pin：历史保护丢失后，预测历史仍保留。

| 量 | 定义 |
|---|---|
| n、o、k | 当前 prompt、公开输出合同、标准继承前缀，tokens |
| s | 当前完整 call 的冻结标准独立服务量，参考秒 |
| R⁺ | 当前 call 之后的后续 LLM 标准服务量，参考秒 |
| t | 当前恢复/重算和剩余输出的价格，部署秒 |
| B | 当前 call 的预期标准工作推进比例，无量纲 |
| J | t/B，派发指标，部署秒；不等于预测 program JCT |
| π₀、πd | 连续性/回退 owner 顺序，以及细化后的派发顺序 |
| M、r、L | 本 call 需求、owner 预约、逻辑准入容量，GPU pages |

在线特征只读当前到达输入、可见历史消息、已完成输出及已发生工具间隔。当前 KV 状态与独立成本标定用于报价；未来 prompt/output、未来轮数、未来工具时长和最终任务成功标签不进入森林或 J。

复现实验使用固定输出合同：各方法在当前 call 到来时得到相同输出预算，并强制生成相同数量 tokens。这不是对未知 EOS 的预测。回放客户端持有完整轨迹，但向预测器只发送本轮已可见消息。编排器声明的工具类别与终止标志在完成/交接时用于生命周期；这种声明式协议也是实验前提，不能当作系统自动推断未知未来的能力。正式客户端将兼容 gap 字段固定为零，不向 TTL 发送真实未来时长；发布配置不启用任何 oracle/truth 模式。

## 3. 到达时冻结历史，异步产生分布

到达时先登记当前输入长度，并结算上次 departure 至本次 arrival 的已发生工具间隔，再复制 request ID、当前 token IDs、可见消息、输入与已完成输出历史。复制后不再依赖可变 request 对象。

单个 FIFO CPU worker 按到达顺序更新历史并预测。tokenizer 与森林预加载；决策线程只收取已完成的 future，不为预测阻塞准入。预测完成本身不新增调度事件。新 call 会清除同 program 的旧预测，消费时再次核对 request ID。

Worker 从当前可见消息重建上一轮已发生的 assistant 输出，与保存的上一轮 prompt 对齐，累计已完成标准服务。用于当前 call 的标准前缀为：

$$
k=b\left\lfloor\frac{\min\{\operatorname{LCP}(\text{上一轮已提交历史},\text{当前输入}),n-1\}}b\right\rfloor,
\quad b=16.
\tag{1}
$$

重建时排除尚未被模型计算的最后一个 token。k 表示历史内容可继承的边界，不是当时仍驻留 HBM 的长度。首轮没有 completed output，不产生用于细化的个体后缀预测。

## 4. 森林预测什么，怎么训练

### 4.1 标准工作单位与标签

固定参考 profile 定义：

$$
c_r(n,o,k)=\sum_h\left[\beta_0^r+\beta_1^r q_h+
\beta_2^r q_h(2a_h+q_h+1)/2+\beta_3^r\right]_++o d_1^r,
\quad a_1=k,\ q_h=\min(2048,n-a_h),\ a_{h+1}=a_h+q_h.
\tag{2}
$$

只累加非空 chunk。[x]₊ 为非负裁剪。第 u 轮的标签为：

$$
R_u^+=\sum_{v>u}c_r(n_v,o_v,k_v),\qquad s_u=c_r(n_u,o_u,k_u).
\tag{3}
$$

参考 profile 固定来自独立小模型标定，仅定义统一工作单位；不是本次 8B 实际速度。工具时间、排队和策略引起的额外恢复不在标签内。s 与 R⁺ 共用单位，不随本次缓存丢失改变。

### 4.2 因果特征与训练范围

冻结训练域为 157 个完整且完整路径位于 16K 范围内的 teacher programs、2106 个调用前状态。原始状态包还含其他 split/删失任务；训练入口显式过滤，不把截断尾部视为已完成标签。

输入共 127 维：81 个阶段、prompt/增长/已完成输出统计及文本首尾各 8192 字符的结构/关键词特征；46 个当前可见消息与历史特征，包括已完成服务、已发生命令类型、测试反馈、return code、重复操作和可见 Python 路径。全部定义在 [features.py](../kairos/features.py) 与 [causal.py](../kairos/causal.py)。program ID、当前输出合同、未产生的 assistant 内容和未来工具不作为森林特征。ID 只用于归属、分组和重叠检查。

每个状态权重 w=1/Np，使每个 program 总训练权重相同。48 棵随机树；深度 7，结构侧最小叶 24，特征比例 0.8，seed=20261020。每棵树按 program 确定性哈希二分，结构侧和叶估计侧 ID 不重叠。

结构侧独立构造 32 等分分位网格，补零/最大值并去重。区间中点 z、宽度 Δ 对应多输出回归标签：

$$
y_j(R)=\sqrt{\Delta_j}\,\mathbf1\{R\le z_j\}.
\tag{4}
$$

平方损失对应网格上的积分 CDF 误差近似，因而与 CRPS 对齐；不声称精确求解连续 CRPS 最优树。分裂阈值只由结构侧决定；叶分布只使用估计侧标签，空叶回退到该树全部估计侧样本。原子按 0.01 参考秒合并。

### 4.3 在线输出与评估

对特征 x，输出 48 个命中叶经验分布的平均：

$$
\widehat P(R\in A\mid x)=\frac1T\sum_{h=1}^T
\frac{\sum_{j\in E_h\cap L_h(x)}w_j\mathbf1\{R_j\in A\}}
{\sum_{j\in E_h\cap L_h(x)}w_j}
=\{(r_a,p_a)\}_a.
\tag{5}
$$

分母为空时按上述回退。部署只需 float32 对齐、树遍历、原子合并和归一化，不需 sklearn。零原子表示模型的“本轮后无服务量”质量，不是已知该 call 必为最后一轮；最大样本也不是物理上界。

模型选择使用固定三折 program 分组 OOF 的宏平均 CRPS：

$$
\operatorname{CRPS}(P,y)=\sum_a p_a|r_a-y|-\tfrac12\sum_{a,b}p_ap_b|r_a-r_b|.
\tag{6}
$$

先 program 内平均，再 program 等权平均。另报分布均值 MAE、90% 区间覆盖/宽度、同阶段配对排序。具体结果见[实验文档](EXPERIMENTS.md)。不能将“一个概率分布”的质量概括为单一准确率，也不把宽区间当作预测非常精确的证据。

## 5. 决策时计算物理价格

只在至少两个成熟 ready 请求具有有效预测时，批量读取一次 CPU 已提交居留目录，再查各请求真实可命中的 HBM 前缀。没有新增 offload，不为了改善分数修改布局。

令 g 为当前 computed tokens 与 HBM 命中的较大值，限制到 prompt 内再向下对齐 16-token 页。CPU chunk 为 256 tokens，已提交集合为 C。从 floor(g/256) 开始连续扫描 C，不能跳过缺口，停止位置为 c：

$$
e=\min(n,\max(g,256c)),\quad \ell=e-g,
\quad T^{restore}=\mathbf1\{\ell>0\}a_H+\ell b_H+C_d(n,e).
\tag{7}
$$

C_d 为式 (2) 同形的部署 prefill 成本，但使用部署 β、不含输出。CPU 后面孤立存在的块不能越过中间缺口。正在写入尚未提交的块不算副本。若异步接收已经完成或 prompt 全部 computed，则恢复项为零。

最终当前报价为：

$$
o^{rem}=\max(1,o-\#\text{已生成}),\qquad t=T^{restore}+o^{rem}d_1^d.
\tag{8}
$$

H2D 增量报价含查找/scatter 等开销，不是纯 PCIe 带宽。单序列 decode profile 用于比较，不预测未来 batch 组合。目录是建议性快照；真正执行前仍重查锁、hash 与存在性。恢复报价不是将来布局不变的承诺。

## 6. 核心指标与其解释

$$
\boxed{B=\sum_a p_a\frac{s}{s+r_a},\qquad J=\frac{t}{B}.}
\tag{9}
$$

按 J 升序，同分按 program 初始到达顺序。B 表示完成当前一步占“本轮加后缀标准工作”的预期比例，不是完成整个 program 的概率；t/B 表示单位这种推进比例的当前代价。

不能先把分布压成均值，因为 E[s/(s+R)] 通常不等于 s/(s+E[R])。例如 s=t=1，后缀恒为 9 时 J=10；后缀各一半为 0、18 时 J≈1.9，虽然后缀均值仍为 9。这是说明性算例，不是实验数据。该非线性偏好短后缀机会，也可能放大错误短尾概率，因此不宣称比均值策略普遍更优。

边界上：R⁺=0 时 J=t，退化为当前成本排序；后缀确定且 t=s 时 J=s+R⁺。固定 t、B 的串行不可抢占代理问题中，相邻交换对加权本轮完成时刻和的差为：

$$
[\sum B C]_{i\to j}-[\sum B C]_{j\to i}=B_jt_i-B_it_j.
\tag{10}
$$

因此 t/B 排序解这个简化代理。真实系统有批处理、自悬挂和容量约束；代理目标不等于 program JCT，不提供每次换序必改善 JCT 的保证。

## 7. 槽位细化：让预测改变顺序，不越过执行边界

先构造包含 ready、running、tool-idle owner 的连续性顺序 π₀。它来自按已观测阶段筛选的独立经验路径：平移 prompt 使当前长度一致，按剩余输出总量选择中间路径，再用当前进度与部署 prefill/decode 价计算剩余工作 W₀。其计价为各轮剩余输出乘 d₁，加缺失 prompt 的 β₁q+β₂q(n+e+1)/2；当前轮含已计算进度，未来只用经验路径。它是回退/连续性包络，不是森林的联合未来预测，也不用于仿真所有程序交互。

成熟 ready 原先占据这个完整顺序中的若干槽位；cold 用当前调用价，未获有效预测者保留原槽。集合 E 只包含成熟、未 grant、预测 request ID 匹配的请求。若 |E|<2，不新增比较；否则只将 E 按 J 重排并填回原 E 槽位。

例：[idle-A, ready-B, cold-C, ready-D, running-E] 中只有 B、D 预测有效且 D 更优，就变成 [idle-A, ready-D, cold-C, ready-B, running-E]。不把 D 直接升到所有 owner 前面。

πd 用于派发和 native program rank；π₀ 继续用于 optional victim 与 TTL 连续性收益。这是明确分工，不是同一个分数统治所有决策。没有预测不阻塞，也不因一个未就绪 future 让所有候选丢失细化。

Native 还保留执行类别：已恢复或已有 computed 进度的 waiting 优先，随后才比较 program rank。实际编码为：

$$
\operatorname{priority}=10^6\mathbf1\{\text{尚未恢复且 computed=0}\}+\operatorname{rank}_{\pi_d}.
\tag{11}
$$

这是类别编码，不是新效用权重。故执行并不是对全部 waiting 无条件取 J 最小者。

## 8. 事件、cohort 与完整调用授权

Native 每引擎步运行；上层竞争由到达、逻辑完成、真实容量释放、idle 保护到期等 dirty 事件触发。decode 增长不逐 token 重跑森林；逻辑完成与稍后的物理 cleanup 可能分别改变状态。

决策先排除已经 grant 的 request，即使它仍在 native waiting。若有未授权成熟续轮，本事件只扫描成熟 cohort；没有才开放 cold。前一个成熟请求放不下可检查后续成熟请求，但不会自动让 cold 越过整个阻塞成熟集合。Cold 授权窗口为 max(1,S−|active|)，S=8。

每个 call 使用公开当前输出合同/上限计完整需求：

$$
M_i=\lceil(n_i+o_i^{cap})/16\rceil,\quad
\sum_j r_j-r_i+M_i\le L,\quad L=C_{usable}-32.
\tag{12}
$$

同时检查 active 名额；工具期保留身份也可能占 active，所以它不等于 decode batch 大小。32 页为原执行协议安全余量，不是为 J 设计的调参阈值。归档实测 Cusable=2524，因此 L=2492。

已 resident 的续轮可在不足时撤销合法 optional 支持；失去 resident 身份的成熟请求只检查未承诺容量及名额；cold 先查空闲，必要时也可走既有合法 optional reclaim。Victim 按 π₀ 较差者优先逐个释放至 fit；不枚举所有 KV 子集，不因 J 高就撤销 running 或已 grant。清理中、DMA/计算占有的强引用不算可释放。

Grant 更新 owner 的完整当前调用预约和 request 集合；后来的重排不撤销它。它不是提前物理分配全部 decode KV，也不是永久 program 通行证。一次事件可以 grant 多个或零个。当前方法没有额外的、带效用分数的 WAIT 候选；等待来自容量、cohort 与安全条件。

合法回收仍有局限：失败前可能已经释放部分 optional 支持，不保证失败零副作用；resident 与非 resident 权限不同。这些是实际边界，不隐藏成“全局联合最优回收”。

## 9. 工具期 TTL：有限保护的经济决策

工具时间使用类别经验 CDF，不使用后缀森林 R⁺。样本来自独立 prior 与已经发生的 live 返回；每类 live 最多保留 512 条。已知类别用本类 prior，未知用 pooled prior；正式配置只在 prior 支持范围内更新 TTL 样本质量，单个长尾观测不直接创建极长 deadline。

对 gap 样本 d₁…dN 定义：

$$
F(\tau)=N^{-1}\sum_a\mathbf1\{d_a\le\tau\},\quad
H(\tau)=N^{-1}\sum_a\min(d_a,\tau).
\tag{13}
$$

保护收益 V=ĉ+α W̄：ĉ 是按已提交长度报价的恢复成本 aH+bH×length；W̄ 是此前无 pinned 连续性返回调用的等待样本均值；α 是当前未 grant ready/running 的其他 owner 中、位于 π₀ 下游的比例。Tool-idle 不纳入这些 peer。这里的恢复价没有升级为式 (7) 的精确布局差分。

令 Q 为当前未 grant ready 集合，a 为 free pages 加本轮预计释放页数：

$$
\Delta=\sum_{j\in Q}(M_j-a)_+,\quad
\rho=\min(1,\Delta/\max(1,L))\min(1,|Q|/\max(1,S)).
\tag{14}
$$

没有竞争者则 ρ=0。它是状态机会成本代理，不是逐页精确影子价格，重复 deficit 也不是独立物理需求之和。

$$
\boxed{\tau^*=\arg\max_{\tau\in\{0,d_1,\ldots,d_N\}}
\{F(\tau)\max(0,V)-\rho H(\tau)\}.}
\tag{15}
$$

断点间 F 不变而 H 不减，所以只需排序后以累计数/前缀和扫描经验断点；O(N log N)，同效用取更早 deadline，无样本取零。ρ=0 且 V>0 时通常选最大支持，不能仅因经济公式存在就声称每个状态都有额外收益。

正 TTL 只 pin 本轮已提交、有 hash、连续完整的合法前缀，不做保护长度/TTL 二维优化。到期且没有 queued continuation 时释放；已返回并在 waiting/skipped waiting 的 program 保留到期豁免，仍可因合法压力提前回收。它不是显式重新估价续租。请求获得执行所需 native 引用/分配后才 unpin 同 owner 的 optional 引用。

因此 TTL 不是实际驻留时长：TTL=1s、0.8s 返回但排队至 4s 时可能实际保护近 4s，期间并未一直使用 KV。H 不预测返轮后排队。

## 10. 两层 KV 数据面

恢复对象是 GPU 连续前缀加 CPU 可连续接续区间，不是两层命中块数之和。chunk 对齐的说明模型为：

$$
\mathcal R(g,C)=\min(\{j:g\le j<m,\ j\notin C\}\cup\{m\}).
\tag{16}
$$

GPU 覆盖 0…2，CPU 有 3、4、5 时可接续到 6；若 GPU 缺 2 且 CPU 也缺 2，后续 3、4、5 无法跨洞。实际仍按 16-token 页处理，不进行任意缺口 attention。

GPU allocator 只在 ref=0 的合法 free 候选中，逐页选择删除前后所有已知 owner 联合边界损失最小者：

$$
\delta(b)=\sum_{p:b\in p}(\mathcal R_p^{before}-\mathcal R_p^{after(b)}).
\tag{17}
$$

每删一页更新边界，共享页累计各 owner 影响；同损失沿 native 顺序。整组 native victims 对联合边界零损失时可直接采用，但不能把旧单页零分数当作共同删除零损失证明。这是贪心选择，不是全局最优子集求解。

CPU STORE 仅在实际空间不足时回收。新缺失对象数 Nnew、chunk 字节 z、当前用量/容量 U/C 对应：

$$
D=\max(0,Nnew\cdot z-(C-U)),\quad K=\lceil D/z\rceil.
\tag{18}
$$

候选必须原生可驱逐、非待写/在途/锁定。已知 owner 只允许删除不缩短当前联合前缀的对象；普通 history-cover 的恢复价 G(c)=[(β₁−bH)256c+β₂(256c)(256c+1)/2]₊ 及带时间归一的日志值不越过这个零损失可行域。

压力下优先保存 GPU 未覆盖的互补历史；每 owner 允许一个靠近边界的重复 chunk 作为备份，更多重复仅用回收前已有空闲，避免“删重复→立即重建→再删除”。容量仍不足时，partial reservation 优先前向保存边界后的缺失 chunks，再反向考虑边界内备份；既有对象保留，未获真实槽位的新 STORE 跳过。

跳过的是副本写入，不是输入、attention tokens 或精度。LOAD 前重查对象/hash/锁；D2H 完成前不声称副本已提交，H2D 完成前不交给计算。GPU/CPU 建议目录不能代替执行证书。普通 APC 写入建议有 0.15s 新鲜度约束。运行、DMA 与延迟释放引用不提前复用。

## 11. 一次完整执行的文字算法

~~~text
请求到达：
  结算已发生工具间隔，登记当前输入/合同，冻结因果快照
  清除同 program 的过期预测，提交单 worker；标记 dirty
原生调度步：
  维护真正到期的 idle pin
  若 dirty：
    构建连续性顺序 π₀、容量账本和候选；排除已 grant
    收集 request-ID 匹配的已完成预测
    若有效成熟 ready 至少两个：读取当前 KV；计算 s、B、t、J
    仅细化这些成熟槽位为 πd，victim/TTL 仍按 π₀
    有成熟等待则只扫描成熟；否则扫描 cold
    逐个检查完整调用容量/名额，必要时走合法 optional 释放
    成功则 grant 并记账；全部不 fit 则等待
  把已 grant 请求交给 native continuous batching
  用真实分配、引用、H2D/D2H 完成事件与 capacity fence 执行
逻辑完成：
  记录已完成输出；如进入工具，计算经济 TTL 并 pin 合法前缀
  标记 dirty；物理 cleanup 仍等真实引用/传输结算
下一轮实际到达：
  重复以上流程，历史与布局分别更新
~~~

## 12. 复杂度与边界

森林 O(Td) 遍历后还要合并叶原子；m 个候选评分 O(ΣKi)、排序 O(m log m)，另有目录 RPC 和 prefix lookup。完整控制器还保留经验包络快照，不能只以纯 scorer 时间代表总成本。

完整调用是上层非抢占承诺边界；下层仍有原生 emergency preemption 兜底。没有多级反馈队列、时间片或 running 长度预测。授权不绕过真实分配，异步未结算容量不足时由 fence 等待，不能提前使用仍被计算/DMA 占有的块。

当前局限包括：短后缀误预测会影响机会分配；首轮仍不可辨识；成熟优先没有无限到达下的 cold 无饥饿保证；失去 resident 身份后再进入权限有跳变；TTL 与派发不是统一全局最优目标；删失长尾未充分解决。已有端到端证据限定在发布的压力回放，不替代多工作负载或真实 agent 质量验证。
