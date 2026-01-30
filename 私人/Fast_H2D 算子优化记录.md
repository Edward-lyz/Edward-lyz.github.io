# 背景

该算子是 `KV Offload`的核心执行算子之一，以 `evict_ids` 指定的 `(src_idx, dst_idx)` 为索引，从 `CPU pinned memory` 的 `kv_cpu` 的张量中读取每个
  token （固定 656 字节），逐字节拷贝到 GPU 的 `sparse_gpu` 这个张量里。初版实现上是单线程/单 token 的 `UVA` 读写。

这里需要讲清楚两个概念：
1. 什么是 CPU 的 `pinned memory`?
2. 什么是 `UVA` 读写？

这里给出精简的回答：

- **锁页内存 (Pinned Memory)** 是指==在物理内存中被“锁定”的区域，操作系统**不允许**将其交换（Swap）到磁盘的虚拟内存中==。在深度学习或高性能计算中，数据需要频繁从 CPU 传输到 GPU。**传输加速：** 使用锁页内存可以避免“先从普通内存拷贝到临时锁页缓冲区，再传给 GPU”的中间步骤，从而显著提升带宽（通常可提升 **2-10 倍**）。**异步操作：** 它支持数据传输与计算的并行（Overlap）。比如在 PyTorch 中设置 `pin_memory=True`，可以在 GPU 训练当前批次时，CPU 异步准备下一批次数据。
- **UVA (Unified Virtual Addressing, 统一虚拟寻址)** 是 NVIDIA 在 CUDA 4.0 中引入的一项技术。它将 CPU 内存（主机内存）和所有 GPU 显存（设备内存）映射到一个**共享的虚拟地址空间**中。
- **UVA vs. UVM**：**UVA (Unified Virtual Addressing)：** 仅仅是地址空间的统一。它**不会自动搬运数据**。如果 GPU 访问 CPU 上的地址，数据仍然通过 PCIe 实时传输，速度受限且延迟高。**UVM (Unified Memory / `cudaMallocManaged`)：** 是在 UVA 之上的更高级功能。它不仅统一地址，还会**自动在物理层迁移数据**。当 GPU 需要数据时，驱动会按需将内存页搬运到显存中以提速。

进一步的，我们会思考这么一个问题：这里的原版本的实现中，为何要用略显繁琐的`UVA`，而不是看起来更简单的`UVM` 呢？
> [!note]
> 因为我们这个算子处理的是稀疏化的索引 gather 操作，几乎不存在空间局部性，导致直接换页的开销过大，不如 UVA 精准地命中某一个 token 的数据

# BaseLine 的代码分析

结合背景的介绍，我们已经得知了这个算子的语义，以及算子要使用的基本原语。那么剩下的就剩下使用 NCU 工具，具体地分析一下这个算子，实际表现出来的性能如何，以及报告中提到的缺陷是否能够和代码一一对应上。
使用了附录中的 bench 脚本，对初始版本的代码进行了测试，测试结果如下：
```Shell
Benchmarking fast_intra_layer_h2d with:
  BS: 128
  TopK: 2048
  Total Copies: 262144
  Data Size: 164.00 MB
Warming up for 2 iterations...
Benchmarking for 10 iterations...
----------------------------------------
Results:
  Avg Latency: 81.3025 ms
  Throughput:  1.97 GB/s
----------------------------------------
Sanity Check Passed: src[95407] == dst[0] == 142
```
可以看到：初始版本的实现，性能相当拉跨，实现的带宽大概只有 2GB/s。简单分析代码可知，最大的瓶颈点在于：原本的代码是串行执行地，压根没有向量化的 IO 操作。

# V1：向量化拷贝
执行效果如下：
```Shell
=============End of environment vars ======================
Benchmarking fast_intra_layer_h2d with:
  BS: 128
  TopK: 2048
  Total Copies: 262144
  Data Size: 164.00 MB
Warming up for 2 iterations...
Benchmarking for 10 iterations...
----------------------------------------
Results:
  Avg Latency: 8.8754 ms
  Throughput:  18.04 GB/s
----------------------------------------
Sanity Check Passed: src[250697] == dst[0] == 141
```

# V2: Warp-Level 读取
执行效果如下：
```Shell
=============End of environment vars ======================
Benchmarking fast_intra_layer_h2d with:
  BS: 128
  TopK: 2048
  Total Copies: 262144
  Data Size: 164.00 MB
Warming up for 2 iterations...
Benchmarking for 10 iterations...
----------------------------------------
Results:
  Avg Latency: 3.7483 ms
  Throughput:  42.73 GB/s
----------------------------------------
Sanity Check Passed: src[142855] == dst[0] == 247
```

# V3： 流水线优化
执行效果如下：
```Shell
=============End of environment vars ======================
Benchmarking fast_intra_layer_h2d with:
  BS: 128
  TopK: 2048
  Total Copies: 262144
  Data Size: 164.00 MB
Warming up for 2 iterations...
Benchmarking for 10 iterations...
----------------------------------------
Results:
  Avg Latency: 3.5992 ms
  Throughput:  44.50 GB/s
----------------------------------------
Sanity Check Passed: src[89744] == dst[0] == 154
```