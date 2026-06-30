# 【工程】NCU 食用指南

# 摘要/结论

- **宏观定性**：使用 **GPU Speed Of Light Throughput** 查看算子在计算单元和内存系统的吞吐利用率。Sol throughput 指标会显示计算与内存各自的利用率；计算高说明算子计算密集，内存高说明内存带宽是瓶颈。若二者都低于 60% 表示延迟隐藏有问题。
- **资源瓶颈**：通过 **Occupancy & Scheduler Statistics** 判断硬件资源是否限制了延迟隐藏。理论占用率由 SM 中最大可并行的 warps（线程束）决定，而实际占用率反映活跃 warps 的平均值；如果理论占用率很低，通常是寄存器或共享内存限制。调度器的 **Eligible Warps** 显示有多少 warps 准备好发出指令；若该值接近 0 则说明没有足够的工作隐藏延迟。
- **深度拆解**：在 **Warp Stall Reasons**、**Memory Workload Analysis** 和 **Compute Workload Analysis** 中定位瓶颈。Stall Long Scoreboard 表示等待全局/纹理/表面加载；Stall Barrier 表示线程束在同步点等待；通过 L1/L2 Hit Rate 判断缓存利用效率；Pipe Utilization 显示算子使用的 ALU/FMA/Tensor 单元占比。
- **精准定位**：最后利用 **Workload Distribution** 观察各 SM/SM Scheduler 的负载均衡，使用 **Source Counters** 将 stall 百分比映射到具体的源代码行。热点行具有高 stall 百分比或低 branch 效率，结合上一阶段的瓶颈判断可确定优化目标。

<aside>

总体流程是：先看宏观吞吐确定算子属于计算绑定还是内存绑定；再分析资源限制和调度器状态，确认有无足够并行度；接着深入查找具体 stall 类型、缓存命中率和指令流水线利用率；最后追踪到不平衡的工作分布和源代码行，从而制定针对性的优化措施。

</aside>

# 具体内容

### 一、宏观体检：GPU Speed Of Light Throughput

Nsight Compute 的 *Speed Of Light Throughput* 报告提供了计算和内存吞吐率的高层视图，每个单元的数值表示实际吞吐与理论峰值的百分比。高吞吐率意味着相应单元充分利用，而低吞吐率表示资源闲置或延迟隐藏失败。建议按以下步骤解读：

1. **比较 compute vs memory throughput**：计算吞吐占比更高表示算子计算密集；内存吞吐占比高则说明内存带宽是主要瓶颈。若二者都低，说明 GPU 未“喂饱”，可能存在长延迟未被隐藏。
2. **延迟警告**：当计算与内存吞吐都低于 60% 时，需要查看下一级的调度器和占用率，确认是工作量太小还是瓶颈导致 stall 多。
3. **配合 Roofline 图**：Nsight Compute 还提供 roofline 分析，根据算子的算术强度(AI=flops/byte)判断理论性能上限。若算子位于带宽受限区，可考虑数据复用或算子融合提升 AI；若算子位于计算受限区，则需优化算法或并行度。

### 二、资源瓶颈分析：Occupancy 与 Scheduler Statistics

为了隐藏内存和执行延迟，GPU 需要有足够数量的活跃 warps。Nsight Compute 的 **Occupancy** 部分显示理论和实际占用率：

- **理论占用率 (sm__maximum_warps_per_active_cycle_pct)**：由 SM 能支持的最大 warps 数量与 launch 配置决定。
- **实际占用率**：实际活跃 warps 平均值，低于理论值通常是寄存器或共享内存数量限制引起的，或者 grid/block 配置不合理。

**Scheduler Statistics** 提供调度器层面的视图：

- **Active Warps/Scheduler** 表示调度器手中持有的活跃 warps 数；
- **Eligible Warps/Scheduler** 是准备执行的 warps 数；若该值接近 0 ，说明大部分 warps 正在等待数据或同步；
- **Issued Warps/Scheduler** 表示实际发出的 warp 数，现代 GPU 通常每个周期最多可发出 1–2 个 warp。若发射率较低，应查看 warp stall 原因。

分析顺序为：先看理论占用率，若远低于 100% ，查阅右侧 *Block Limit* 表确定是否受寄存器数量或共享内存限制；然后查看实际活跃 warps 与 eligible warps，若 eligible warps 过少，说明工作负载太轻或指令依赖导致 stall；最后查看 issued warps 与理论值的差异评估调度器效率。

提高占用率的常用方法：

- 减少内核中的局部变量数量，使寄存器使用量低于硬件限制；
- 使用 `__launch_bounds__` 指定块大小与最大寄存器数限制，以便编译器进行寄存器溢出优化；
- 调整 `blockDim` 及 `gridDim`，使每个 SM 有足够数量的 blocks，避免“波浪尾”效应（最后一波 warps 不足以填满 SM）。

### 三、深度拆解：Warp Stall、Memory & Compute Analysis

在有足够的 warps 情况下，性能仍然可能受特定 stall 或流水线瓶颈影响。Nsight Compute 提供了 **Warp Stall Reasons**、**Memory Workload Analysis** 和 **Compute Workload Analysis** 用于深入诊断。部分常见 stall 原因及其含义如下：

| Stall 指标 | 含义 | 可能原因及优化方向 |
| --- | --- | --- |
| **Long Scoreboard** (`smsp__pcsamp_warps_issue_stalled_long_scoreboard`) | Warp 正等待 L1/TEX (local/global/texture/surface) 内存操作的数据。 | 全局或纹理访问延迟高；检查访问模式是否合并、对齐，并尝试提高 L1/L2 命中率。可将热点数据放入 `__shared__`；针对新架构使用异步拷贝 (`cuda::memcpy_async`) 及 `__pipeline_memcpy_async` 提前发起加载。 |
| **Barrier** (`smsp__pcsamp_warps_issue_stalled_barrier`) | Warp 在同步点等待兄弟 warps。 | 由于条件分支或工作量不均导致部分 warps 提前到达 barrier。使每个 block 内工作量均衡；若线程数较大可尝试将大 block 拆分为多个小 block。 |
| **MIO Throttle** (`smsp__pcsamp_warps_issue_stalled_mio_throttle`) | Warp 等待 MIO（包括共享内存、特殊数学指令、动态分支等）的队列。 | 共享内存访问频繁或 bank 冲突，或者使用了特殊函数 (如 sin/cos)。减少动态索引或合并多个加载；避免频繁的 `__syncthreads()`；使用 `__ldg` 指令或 L1 cache 决策控制。 |
| **LG Throttle** (`smsp__pcsamp_warps_issue_stalled_lg_throttle`) | 等待本地/全局内存指令队列。 | 本地内存或全局内存访问过于频繁，引起 L1 指令队列饱和。检查是否因寄存器溢出导致大量局部数组被放入本地内存。 |
| **Not selected** (`smsp__pcsamp_warps_issue_stalled_not_selected`) | 有多个 warps 合格，但调度器选择了其他 warp。 | 并行度充足，但数据局部性不佳或缓存命中率低。可考虑减少活动 warps 数量以提高缓存命中率。 |
| **Short Scoreboard** (`smsp__pcsamp_warps_issue_stalled_short_scoreboard`) | 等待与共享内存或 MIO 操作相关的数据。 | 多由共享内存 bank 冲突或频繁执行特殊数学指令引起。应检查共享内存布局，确保访问模式避免 bank 冲突；对常用数据使用寄存器。 |

**Memory Workload Analysis** 将内存访问划分为不同类型（全局 load/store、共享 load/store、纹理 load、原子等），展示每种请求数量和字节数，并提供 **L1 Hit Rate**、**L2 Hit Rate** 等指标。高命中率说明大部分数据在较近的缓存中满足；低命中率提示应改善数据局部性或使用 `__shared__`。例如在课程材料中，优化后 L1 命中率从 25% 提升到 94.8%，同时全局内存请求数大幅减少。

**Compute Workload Analysis** 显示不同运算单元（INT、FP32、FP64、Tensor Core等）的利用率和指令比例，可判断算子是否充分利用 Tensor Core；若浮点/整数指令比例失衡，考虑重新排列计算以更好地利用硬件管线。

### 四、精准定位：Workload Distribution 与 Source Counters

当确定了主要 stall 类型后，需要将抽象指标映射到代码。**Workload Distribution** 图表展示各 SM 或各 SM Scheduler 的活跃周期差异；若最大与最小差异很大，表示工作分配不均，应调整 grid 划分或引入动态调度以平衡工作。

**Source Counters**（在 Detailed → Source 页面）提供行级别指标，如分支效率、stall 百分比等。报告会列出源代码行和对应的 PTX/SASS 指令，并显示 Long Scoreboard、Barrier 等 stall 百分比。解读方法：

1. 查找占比最高的 stall 类型对应的源代码行，确定瓶颈热点。
2. 对于分支效率低的行，减少条件分支或将常量计算提前。
3. 如果某行对应多个 PTX 指令，可拆分源代码为多行以得到更精确的定位。
4. 在优化前后比较占比变化，验证改动是否有效。

# 建议与思考

1. **渐进式分析**：不要一开始就启用所有详细的 section。可先用默认配置获取宏观吞吐和占用率，判断是否需要深入收集 Warp Stall、Memory Workload 等高开销指标。这样可以降低采样开销，提高分析效率。
2. **结合时间分析**：Nsight Compute 提供每个 kernel 的平均执行时间和方差。结合 Nsight Systems 系统级分析，可以查看内核启动间隔、异步复制等时间，找出调度器未发指令的根本原因。
3. **平衡 Occupancy 与 Cache 局部性**：高占用率并不总是最佳。有时减少活跃 warps 反而能增加缓存命中率、减小竞争，提升整体性能。这可通过调节 block 大小、使用 `cg::thread_block_tile` 或动态并行来实现。
4. **利用异步拷贝与 pipelining**：Ampere 及更新架构支持 `cuda::memcpy_async` 和 `cp.async` 指令，可在计算阶段并行发起内存拷贝，减少 Long Scoreboard stall；同时利用 `cuda::pipeline` API 创建流水线，手动覆盖延迟。
5. **继续学习 Nsight 文档**：Warp Stall Reasons 列表很长，不同硬件的命名也略有差异。建议阅读官方文档中 Warp Stall Reasons 的完整说明并结合自己的微架构进行对照。
6. **反向思考**：在优化过程中不应盲目追求单一指标。例如，减少寄存器使用可以提高理论占用率，但可能导致溢出到本地内存，引起 LG Throttle 增高。因此每次修改后需综合评估 Occupancy、Stall Reason 和 Cache Hit Rate 的变化，以确认优化的实际效果。若某一指标改善带来了另一个指标恶化，应重新权衡。

通过上述系统的分析流程和指标解读，可以将 Nsight Compute 的 Detailed 报告转化为明确的优化方向，把抽象的 stall 百分比映射到具体代码行，从而在实际工程中高效提升 CUDA 算子的性能。