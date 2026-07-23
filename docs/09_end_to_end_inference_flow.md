# M20-B 当前端到端推理流程

> 更新日期：2026-07-22  
> 适用范围：`runtime/m20` Group Double Buffer（GDB）stage-ready 主线  
> 当前说明点：`G=4,S=8,K=1,C=4,W=8,Q=4`，exact-oracle + min-delta，
> 正式历史证据仍是同步 delta；ticket-aware async runtime 已实现并完成一次小规模 GPU
> 同 trace 正确性诊断，待目标配置三轮性能验收  
> 状态：这是当前选出的下一轮正式候选，不是最终性能结论。

## 1. 一句话概括

系统把每个请求的 prompt 切成小 chunk，让不同请求的 ready chunks 在 12 个
layer-group 队列之间独立前进；每次选择同一 group 的最多 4 个 chunks，共享一套
逐层 GPU expert plan。同步参考先物化 A/B buffer 再执行 4 层；async replay 则在当前
action 计算时把冻结的下一 ticket 后台物化到 inactive buffer，下一 action 只等待尾部。
每层中命中 GPU plan 的路由在 GPU BF16 expert 上计算，其余路由由 CPU KT Q8
expert 计算，最后合并，因此小 GPU buffer 只影响性能，不会丢失 MoE 路由。

已有正式收益来自 **跨请求 stage 重排、cohort 共享、同 group 复用、zero-load 和
bounded delta**。新 async pipeline 已有一次真实 GPU 功能证据，但没有三轮性能证据，
不能把旧收益归因给它。

## 2. 当前实验口径

| 项目 | 当前值 | 含义 |
|---|---:|---|
| 模型层数 | 48 | Qwen3 MoE decoder layers |
| 每层专家数 | 128 | 每一层都有自己独立的 128 份 expert 权重 |
| Router top-k | 8 | 每个 token 在每层选择 8 个 logical experts |
| `G` | 4 | 一个 layer group 连续执行 4 层，共 12 groups |
| `S` | 8 | active group 中每层有 8 个 GPU expert slots |
| Buffer 数 | 2 | A/B 各保存一个 `G x S` working set |
| `K` | 1 | 同一层相对逻辑 source plan 最多换入 1 个 expert |
| `C` | 4 | 一个 action 最多合并 4 个 ready chunks |
| `W` | 8 | 每个 group 最多查看最老的 8 个候选 chunks |
| `Q` | 4 | 其他 group 也 ready 时，同一 group 最多连续 4 个 actions |
| Active states | 8 | stage 系统中最多 8 个 request-chunk states |
| Chunk 上限 | 256 tokens | SGLang chunked prefill 大小 |
| CPU | 64 threads / 2 pools | LLAMAFILE AVX2 expert 计算 |
| Materialization | sync delta reference / async replay candidate | async 仅用于冻结 replay |
| Workload | p8/c8, seq约2048 | 8 个并发请求，当前只生成 1 token |

一个 BF16 expert 的 `w13+w2` 约为 9 MiB，因此当前 expert execution buffer 为：

```text
每个 buffer = G x S = 4 x 8 = 32 experts = 288 MiB
A + B       = 2 x G x S = 64 experts = 576 MiB
```

这只是 expert execution buffer，不等于进程的全部 GPU 显存。

## 3. 必须区分的四个对象

| 对象 | 定义 |
|---|---|
| request | 一个用户请求和一条完整 prompt |
| chunk | request 中一段最多约 256 tokens 的连续区间 |
| state | 一个 chunk 当前的执行状态，包括 group、hidden/residual 和 KV ownership |
| group action | 选择一个 group、一个 cohort 和一套 plan，物化 buffer 并执行 4 层 |

一个 action 不是“一次 request”，也不是“一层”或“一次 expert 搬运”。一个 chunk 要完成
12 次 state-group visit 才走完 48 层；同一个 action 可以同时推进最多 4 个 chunks。

## 4. 总体流程

```text
请求到达
  |
  v
tokenize，并由 SGLang 切成 <=256-token chunks
  |
  v
当前 chunk 变成 state，进入 ready[group 0]
  |
  v
全局 stage scheduler
  |-- 从各 group 的 ready queue 查看候选
  |-- 选择同一 group 的 1..4 个 chunks
  |-- 聚合 oracle demand，生成逐层 S=8 plan
  |-- 冻结 NextActionTicket
  v
pack cohort
  |
  v
物化 A/B：sync，或接管已后台加载的 next ticket
  |
  v
原子 commit 目标 buffer 和 expert mapping
  |
  v
顺序执行该 group 的 4 个 Transformer layers
  |-- attention 等普通 GPU 计算
  |-- router top-8
  |-- GPU expert 分支 || CPU expert 分支
  |-- 同步并合并
  v
校验 ticket 和 materialization
  |
  +-- 非最终 group：demux，state 进入 ready[group+1]
  |
  +-- group 11：当前 chunk 完成 48 层
         |-- prompt 还有 chunk：恢复 continuation
         +-- 最终 prompt chunk：logits、sampling、返回首 token
```

## 5. 阶段一：服务启动

启动时完成以下工作：

1. SGLang 初始化 attention、router、norm、KV cache 等正常运行时。
2. CPU KT/LLAMAFILE 路径准备完整 Q8 expert 权重。
3. GPU group manager 分配两份 BF16 `[G=4][S=8]` expert storage。
4. 48 层的 GPU expert `Parameter` 只保留轻量 view；执行某个 group 时才绑定到 A 或 B
   对应的 4 个 layer offsets。
5. 读取 train-derived frequency top-8，作为每层的**逻辑初始 plan**。
6. 当前 oracle 实验读取预采集 routed-expert trace，供每个 token span 查询逐层 demand。
7. 创建 12 个 stage-ready queues，每个 group 一个。

frequency plan 在启动时只是 48 层的 expert ID 表，不表示已经把 `48 x 8` 份权重全部
放入 GPU。A/B 初始可以是空的，目标 group 第一次执行时才 cold load。

## 6. 阶段二：请求切块与 admission

### 6.1 同一请求的 chunk 顺序

当前 M20-B v1 对同一 request 只允许一个 in-flight chunk：

```text
R0/chunk0 完成 group 0..11
  -> R0/chunk1 才能进入 group 0
  -> R0/chunk2
  -> ...
```

这样可以保证后一 chunk 使用到前一 chunk 已经为所有 48 层发布完成的 KV。系统尚未做
同一 request 内的 chunk wavefront；复用主要来自多个并发 requests。

`seq约2048, chunk<=256` 通常形成约 8 个 chunks，尾块可能更短。实际边界以
`token_start/token_end` trace 为准，不能只用整数除法推断。

### 6.2 建立 ChunkStageState

当前 admission 要求原始 `ScheduleBatch` 恰好包含一个 request。系统将它转换成
`SPLIT_PREFILL` state，初始 `group_id=0`，并保存：

- request identity、token span 和 chunk 标识；
- 原始 `ScheduleBatch`/`ForwardBatch`；
- hidden states、residual 和模型特定中间状态；
- request/KV pool indices，保证 co-batch 后请求不会互相 attention；
- 48 层的 oracle demand 与 confidence；
- state version、ready 时间和 reservation owner。

一个 256-token chunk 的 BF16 hidden+residual 约为 2 MiB；当前最多 8 个 active states，
这部分中间状态显存受硬上限约束。

## 7. 阶段三：stage-ready 调度与 reorder

每个 state 自己必须保持：

```text
group 0 -> group 1 -> ... -> group 11
```

但不存在“所有请求都完成 group 0，才能有人进入 group 1”的全局 barrier。例如：

```text
ready[0] = [R3/chunk0]
ready[2] = [R1/chunk0]
ready[3] = [R0/chunk0, R2/chunk0, R4/chunk0]
```

调度器可以先执行 group 3 的三个 chunks，也可以根据搬运代价选择其他 ready group；
它只会重排依赖已经满足的 states，不会跳过某个 chunk 的前置 group。

### 7.1 组成候选 cohort

对每个 eligible group：

1. 查看最老的最多 `W=8` 个 states；
2. 最老 state 必须作为 anchor，防止同 group 内饥饿；
3. 从其余 states 中选择伙伴，cohort 最大为 `C=4`；
4. 一个 cohort 不能包含同一 request 的两个 chunks；
5. ready 数不足 4 时直接执行 partial cohort，不等待凑满。

`Q=4` 约束 group 级公平性：如果还有其他 non-empty group，同一 group 连续执行 4 个
actions 后必须让出选择机会；如果只有这一个 group ready，则不会为满足 Q 而空转。

当前 `max_wait_ms=0` 表示 deadline force 关闭，不是“等待 0 ms 后强制执行”。

### 7.2 为每个候选生成 plan

调度器聚合 cohort 中所有 token 的逐层 oracle demand。每层的 source plan 是：

1. 若该 group 有最近提交的 plan，使用最近 plan；
2. 否则使用该层的 train-frequency top-8。

对当前 min-delta、`K=1`：每层从需求最高的 missing experts 中最多加入 1 个，并淘汰
active plan 中需求最低的 1 个；只有新 expert demand 严格高于 victim 才置换。

```text
target_plan[l] 宽度始终为 S=8
changed[l] = |target_plan[l] - logical_source_plan[l]| <= K=1
```

`K` 是每层、每次 action 的**逻辑变化上限**，不是 H2D 数量，不是 cohort 大小，也不是
frequency。四层 action 最多逻辑换入 `G x K = 4` 个 experts。

### 7.3 min-delta 如何选最终 action

调度器用 A/B 的抽象状态模拟每个候选的物化操作，并按以下顺序比较：

1. H2D experts / token 越少越好；
2. replacements / token 越少越好；
3. D2D experts / token 越少越好；
4. oracle covered routes / token 越多越好；
5. cohort 越大越好；
6. 最后按确定性的 FIFO identity 打破平局。

这是真正的 group-level reorder：调度器同时决定“下一步执行哪个 group”和“该 group
由哪些 ready chunks 共享 plan”。

## 8. 阶段四：冻结 NextActionTicket

选定候选后，scheduler reserve 对应 states，并生成不可变 ticket。ticket 至少冻结：

- `ticket_id`、queue epoch、group ID；
- state/request IDs、state versions、token spans；
- cohort 总 token 数；
- 四层 target plans 和 plan hash；
- logical source plans/source buffer；
- 预期 target buffer 和 buffer versions；
- policy、oracle version、confidence、score 和 fallback reason。

ticket 是 scheduler 与 worker 之间的执行合同。当前只允许一个 outstanding ticket，避免
两个 actions 同时修改队列或 buffer 所有权。

## 9. 阶段五：pack cohort

若 ticket 含多个 states，`pack_stage_batches()` 会拼接：

- token/position tensors；
- request pool indices 和 KV locations；
- sequence/prefix lengths；
- hidden states、residual 和允许的 model-specific states；
- 每个 request 的 metadata。

它们只是共享一次 kernel/attention/MoE 调用。每个 request 仍使用自己的 mask、prefix 和
KV 地址，不能读取另一个 request 的 token。每个 action 都重新建立 attention metadata，
因为 group 和 cohort 组合可能已经改变。

当前 pack 路径只支持 TP=1 的普通文本 prefill；不支持 multimodal、grammar、speculative、
return-logprob，也不允许把 prefill 和 decode 混在一个 stage cohort。

## 10. 阶段六：A/B 物化

### 10.1 A/B 保存的是什么

A 和 B 各能保存一个 group working set：

```text
buffer record = group_id + 4个layer plans + 32份BF16 expert权重 + version
```

它们不是永久的“当前/下一 group”或“奇数/偶数 group”。每次 action 会根据现有内容选择
exact reuse、source 和 destination；旧内容被覆盖后，对应 group cache 就消失。

不同 layer 的同名 expert 是不同权重。例如 `layer 3/expert 7` 不能复用为
`layer 7/expert 7`。所有物理命中都必须按 `(layer_id, expert_id)` 判断。

### 10.2 四种物理路径

对 target plan 中的每个 slot，有四种可能：

| 路径 | 含义 | 传输 |
|---|---|---|
| zero-load | 某个 buffer 已有完全相同 group+plan | 无复制 |
| retain | target buffer 的目标 slot 已有正确 expert | 无复制 |
| D2D | 同 group source buffer 中有该 expert | GPU A/B 之间复制 |
| H2D | A/B 都没有需要的 `(layer,expert)` | CPU host 到 GPU |

### 10.3 同 group delta 示例

假设 A 保存 group 5 的旧 plan，四层各换入 1 个 expert，B 没有可 retain 的 slot：

```text
每层：7 个 retained experts A -> B（D2D）
      1 个 new expert       CPU -> B（H2D）

四层：最多 28 D2D + 4 H2D
```

如果 B 自己保留了目标 expert，实际 D2D/H2D 会更少；若 B 已经是完全相同 plan，则直接
zero-load。

### 10.4 为什么 K=1 仍可能 H2D 32 个 experts

若 A/B 都没有目标 group，必须 cold materialize 完整执行镜像：

```text
G x S = 4 x 8 = 32 experts H2D
```

此时 K 仍可能完全合法。K 比较的是 target 与该 group 的**逻辑历史 plan**，而物理
source 可能已经被其他 group 覆盖。逻辑变化小不等于物理 cache 一定存在。

### 10.5 同步 commit

当前 M20-B action 的顺序是：

```text
prepare host tensors
  -> enqueue retain/D2D/H2D
  -> 等待全部完成
  -> record READY
  -> 原子绑定4层Parameter views和logical-to-slot mapping
  -> record ACTIVE
  -> 开始计算
```

ACTIVE buffer 不会被覆盖，LOADING 或部分有效的 mapping 不会暴露给计算路径。A/B 的
当前主要价值是安全构造新 plan、原子切换、保留 same-group source 和缓存两个版本。

## 11. 阶段七：连续执行四层

一个 action 的 compute window 包含四个完整 Transformer layers，而不只是 GPU 中的
8 个 experts。四层必须按顺序执行，因为后一层依赖前一层输出。

对每一层，逻辑过程为：

```text
normalization / attention / residual
  -> MoE router 为每个 token 选择 top-8 of 128
  -> 按当前8-expert GPU mask拆分路由
  -> GPU BF16 expert branch || CPU KT Q8 expert branch
  -> 等待两路完成并按router weights合并
  -> 输出进入下一层
```

`S=8` 表示 GPU 中有 8 份 expert 权重，不表示该层只执行 8 次 expert。若 cohort 包含
`4 x 256 = 1024` tokens，则一层最多产生约：

```text
1024 tokens x top-8 = 8192 route entries
```

这 8192 条路由中，expert ID 命中当前 GPU plan 的部分在 GPU 计算，其余部分在 CPU
计算。跨 token 可能访问远多于 8 个 unique experts，但所有有效 route entries 都会执行。

### 11.1 层内 CPU/GPU overlap

每层 MoE 内部已经使用 submit-compute-sync：

1. 将 hidden states 放入共享 staging buffer；
2. mask 掉 GPU 已覆盖的路由，异步提交 CPU expert 任务；
3. 同时在主 CUDA stream 上计算 GPU expert 路由；
4. 等待 CPU output 回来；
5. 将 CPU 与 GPU 输出相加并继续下一层。

这是**层内 expert compute overlap**。每层合并前仍有同步点，不能让下一层使用未完成的
输出。

### 11.2 当前不存在的 overlap

当前 stage-ready M20-B 没有执行：

```text
当前ticket计算 || 下一个任意ticket异步物化
```

旧 M20-A1 验证过固定 `group g` 计算时向 inactive buffer 异步加载 `g+1`。stage-ready
reorder 后，下一个 ticket 可能是任意 group，且当前 `begin_action()` 强制 sync load，
因此 A1 的固定流水线不能当作当前 M20-B 的性能来源。

## 12. 阶段八：完成、校验、demux 和推进

四层完成后，worker 返回实际 plan、target buffer、versions 和 copy operations。scheduler
校验 ticket ID、group ID、split cursor 和 plan hash，防止 stale commit 或错误 action。

若不是 group 11：

1. packed hidden/residual 按各 state 的 token span 切回；
2. KV 已经写入各 request 自己的 cache slots，无需跨请求复制回来；
3. state 从 `group g` 变为 `group g+1`；
4. state 进入新的 ready queue；
5. 下一次可以和完全不同的 requests 重新组成 cohort。

因此 cohort 只在一个 action 内有效，不是从 layer 0 到 layer 47 固定不变。

完成的 buffer record 变为 REUSABLE。若随后又选择相同 group，它可能 zero-load 或做小
delta；若先选择其他 group，旧内容可能被覆盖。这就是调度顺序影响物理搬运的原因。

## 13. 阶段九：chunk continuation 与最终输出

group 11 执行 layers 44--47。完成后 state 进入 FINALIZING：

- 若 request 仍有未处理 prompt tokens，只完成当前 chunk 的 prefill/KV 发布，恢复
  SGLang continuation，再 admission 下一个 chunk；
- 若这是最终 prompt chunk，执行最终 norm/logits/sampling，得到首个输出 token；
- 当前 benchmark 为 `max_tokens=1`，首 token 后请求结束，不代表长 decode 吞吐。

当前 stage pack 明确不支持 mixed prefill/decode。若未来测多 token generation，需要把
decode 口径与本文件描述的 stage-reordered prefill 分开报告。

## 14. 一个具体的交错执行例子

以下只展示合法的一种可能顺序，不代表每轮 trace 必然相同：

```text
t0: R0/c0 到达
    action(group0, [R0/c0])                 # 低负载 singleton

t1: R1..R4/c0 到达，R0/c0 已在 ready[1]
    action(group0, [R1/c0,R2/c0,R3/c0,R4/c0])

t2: ready[0] 和 ready[1] 都非空
    min-delta 比较两边的 H2D/token、cohort 和 coverage
    可能选择 action(group1, [R0/c0])

t3: R1..R4/c0 完成 group0 后进入 ready[1]
    action(group1, [R1/c0,R2/c0,R3/c0,R4/c0])

...

R0/c0 完成 group11
    -> 若还有prompt，R0/c1才进入ready[0]
```

这里没有全局 group barrier，也没有同一 request 多 chunk 并行。调度器在满足各自依赖的
states 中重排，以延长同 group working-set 的使用时间并增大 cohort。

## 15. 当前性能收益应如何归因

当前 M20-B 的潜在收益来源是：

1. **Cohort amortization**：一次物化的四层 plan 同时服务最多 4 个 chunks。
2. **Stage reorder**：优先执行 H2D/token 更小、可复用性更好的 ready group/cohort。
3. **Bounded churn**：K 限制每层逻辑换入，避免为短期需求整体替换 S 个 slots。
4. **A/B delta**：保留旧 plan，用 retain/D2D 替代部分 CPU H2D。
5. **Zero-load**：相同 group+plan 再次执行时完全不搬专家。
6. **GPU coverage**：热点 routes 在 GPU BF16 路径执行，减少 CPU critical-path wait。

现有正式收益**不能**主要归因于 group 权重 H2D 与四层计算重叠，因为这些产物使用同步
物化。async runtime 已形成真实 lookahead，但其 overlap、tail 和吞吐要由新实验确认。

## 16. Full planner 与 delta replay 实验流程

当前正式证据不是直接只跑一次 delta。为了隔离调度决策与物理复制，实验分成两个独立
server runs：

```text
Run A: full planner
  -> 在线选择 group/cohort/plan
  -> 用 full materialization 执行
  -> 写入 hash-chained actions.jsonl

Run B: delta replay
  -> 重放完全相同的 state/group/cohort/plan 顺序
  -> 根据实际A/B状态重新推导 retain/D2D/H2D
  -> 用 delta materialization 执行并测性能
```

Run B 可用 `--replay-load-mode sync` 作为参考，或用 `async` 启用同 ticket/plan 的 A/B
lookahead；比较时必须分别运行二者，不能拿旧 sync 产物充当 async 结果。

严格检查包括：action 数、state 顺序、plan hash、trace chain、最终输出和 replacement
budget。主性能结果取 delta replay。

这不表示线上一个用户请求要推理两遍。它是当前研究阶段的 A/B 正确性方法；未来在线
系统应在一次运行中完成 planner + delta，且用 predictor 或 frequency 替代 exact oracle。

## 17. 三种容易混淆的对照

| 配置 | GPU expert 布局 | Stage reorder/cohort | 动态置换 |
|---|---|---|---|
| 传统 KT frequency `N` | 48 层各永久常驻 N 个 | 无当前 GDB reorder | 无 |
| M20 frequency `K=0` | A/B 中按 group 流式物化 S 个 | 有 | 无，逻辑 plan 固定 frequency |
| M20 oracle min-delta `K>=1` | 同一 A/B group buffer | 有 | 有，逐层最多 K 个逻辑换入 |

所以 M20 的 frequency 仍会 cold load、group 切换和 cohort 执行，它不是传统 KT。当前
K sweep 中的 `K=0 frequency` 与 `K>=1 exact-oracle min-delta` 还改变了 placement source，
`K0 -> K1` 不能直接解释成纯 K 消融。

## 18. 当前正确性不变量

1. 每个 state 只能处于一个 ready queue，并且只有一个 ticket 拥有它。
2. 每个 chunk 的 group 顺序严格为 0--11，每个 request 同时最多一个 chunk。
3. 一个 action 只包含同一 group、不同 requests 的 states。
4. final group 前不对用户发布生成 token，也不释放 request/KV ownership。
5. 每个有效 layer mapping 恰好包含 S 个唯一 logical experts。
6. `changed[layer] <= K`；同 group delta 时 `H2D_new <= changed`。
7. READY 前不 commit，ACTIVE 不覆盖，buffer version/plan hash 不匹配则拒绝执行。
8. CPU mask 与 GPU mask 互补，不能漏算或重复计算同一 route contribution。
9. packed requests 的 attention/KV ownership 保持隔离。
10. full/replay 与 delta/replay 在相同 action/plan 下必须输出一致。

不同 placement 会让某些 experts 分别走 GPU BF16 或 CPU Q8，数值可能不同。因此不能用
frequency 的首 token 要求 oracle placement 完全相同；输出一致性只在相同 action trace
和相同 logical plans 之间比较。

## 19. 当前限制与下一步

当前仍有以下明确边界：

- exact oracle 来自预采集 trace，不是可部署在线预测；
- arbitrary-ticket next-action async 已实现并通过并发/CUDA smoke；真实 GPU 小规模诊断
  已证明同 trace 输出和 39/39 action copy-op 等价，但目标 `S8` 三轮性能 gate 尚未完成；
- 一个 request 只允许一个 in-flight chunk；
- 当前只有一个 outstanding group action；
- 文本、TP=1、CUDA Graph 关闭，且不支持 mixed prefill/decode stage pack；
- 当前 workload 只测 prefill 到首 token，不代表 decode throughput；
- `S8/K1/Q4` 仍需 matched frequency 三轮正式交错复验，不能把单轮 screen 当最终结论。

下一阶段应在目标 `S8/K1/Q4,p8/c8,seq2048` 上跑同一 action trace 的 sync/async
三轮交错对照；小规模诊断已通过输出、copy ops 和 trace 等价门槛，但只有 async 三轮
端到端稳定为正，才能把它写成有收益的 weight-load/compute pipeline。predictor 必须
在 oracle 损坏曲线门槛通过后再进入 action 模式。

## 20. 代码与证据入口

| 内容 | 位置 |
|---|---|
| 当前总计划和状态 | [`07_small_slot_stage_reuse_plan.md`](07_small_slot_stage_reuse_plan.md) |
| 当前 S8/K1 示例配置 | [`config.json`](../experiments/m20_b1b_current_hash_p8c8_g4s8k1q4_screen_20260715/r1_b1b_stage_delta_replay_oracle_min_delta_g4_s8_k1/config.json) |
| State、ticket、ready queues、策略 | [`kt_stage_scheduler.py`](../runtime/m20/sglang/srt/managers/kt_stage_scheduler.py) |
| Admission、选择、推进和 finalization | [`scheduler.py`](../runtime/m20/sglang/srt/managers/scheduler.py) |
| Cohort pack/demux | [`kt_stage_batch.py`](../runtime/m20/sglang/srt/model_executor/kt_stage_batch.py) |
| Materialize 后执行 split group | [`model_runner.py`](../runtime/m20/sglang/srt/model_executor/model_runner.py) |
| Async pipeline GPU 诊断 | [`PIPELINE_VALIDATION.md`](../experiments/m20_b_async_pipeline_e2e_smoke_20260722/PIPELINE_VALIDATION.md) |
| A/B storage、delta、commit 和版本 | [`kt_group_expert_buffer.py`](../runtime/m20/sglang/srt/layers/moe/kt_group_expert_buffer.py) |
| 层内 CPU/GPU expert 执行 | [`kt_ep_wrapper.py`](../runtime/m20/sglang/srt/layers/moe/kt_ep_wrapper.py) |

## 21. 最终记忆版

```text
多请求各自按group顺序前进
  -> scheduler选择同group的ready chunks并共享plan
  -> K限制逻辑换入，不限制cold H2D
  -> sync构造当前GxS，或async接管后台构造的下一ticket
  -> 每层router top-8全部执行：命中GPU，否则CPU
  -> 四层后demux并进入下一group
  -> 当前chunk走完48层后，下一chunk才能进入
  -> 最终prompt chunk产生首token
```

当前 A/B 同时承担安全切换、working-set 复用和冻结 replay 的 next-ticket pipeline；
代码路径已经闭环，但真实 GPU 尚未证明其端到端收益。
