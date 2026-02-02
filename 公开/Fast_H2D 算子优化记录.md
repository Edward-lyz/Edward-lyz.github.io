# Fast H2D 算子优化记录
# 背景
该算子是 `KV Offload`的核心执行算子之一，以 `evict_ids` 指定的 `(src_idx, dst_idx)` 为索引，从 `CPU pinned memory` 的 `kv_cpu` 的张量中读取每个
token （固定 656 字节），逐字节拷贝到 GPU 的 `sparse_gpu` 这个张量里。初版实现上是单线程/单 token 的 `UVA` 读写。
这里需要先搞清楚两个概念：
1. 什么是 CPU 的 `pinned memory`?
2. 什么是 `UVA` 读写？
这里给出精简的回答：
- **锁页内存 (Pinned Memory)** 是指==在物理内存中被“锁定”的区域，操作系统**不允许**将其交换（Swap）到磁盘的虚拟内存中==。在高性能计算中，数据需要频繁从 CPU 传输到 GPU。**传输加速：** 使用锁页内存可以避免“先从普通内存拷贝到临时锁页缓冲区，再传给 GPU”的中间步骤，从而显著提升带宽（通常可提升 **2-10 倍**）。**异步操作：** 它支持数据传输与计算的并行（Overlap）。比如在 PyTorch 中设置 `pin_memory=True`，可以在 GPU 训练当前批次时，CPU 异步准备下一批次数据。
- **UVA (Unified Virtual Addressing, 统一虚拟寻址)** 是 NVIDIA 在 CUDA 4.0 中引入的一项技术。它将 CPU 内存（主机内存）和所有 GPU 显存（设备内存）映射到一个**共享的虚拟地址空间**中。
- **UVA vs. UVM**：**UVA (Unified Virtual Addressing)：** 仅仅是地址空间的统一。它**不会自动搬运数据**。如果 GPU 访问 CPU 上的地址，数据仍然通过 PCIe 实时传输，速度受限且延迟高。**UVM (Unified Memory / `cudaMallocManaged`)：** 是在 UVA 之上的更高级功能。它不仅统一地址，还会**自动在物理层迁移数据**。当 GPU 需要数据时，驱动会按需将内存页搬运到显存中以提速。
进一步的，我们会思考这么一个问题：这里的原版本的实现中，为何要用略显繁琐的`UVA`，而不是看起来更简单的`UVM` 呢？
> [!note]
> 
> 因为我们这个算子处理的是稀疏化的索引 gather 操作，几乎不存在空间局部性，导致直接换页的开销过大，不如 UVA 精准地命中某一个 token 的数据
# BaseLine 的代码分析
结合背景的介绍，我们已经得知了这个算子的语义，以及算子要使用的基本原语。那么剩下的就剩下使用 NCU 工具，具体地分析一下这个算子，实际表现出来的性能如何，以及报告中提到的缺陷是否能够和代码一一对应上。

**代码形态 (Naive Byte-Copy):**
```C++
// 伪代码：每个线程处理一个 Token
__global__ void baseline_kernel(...) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= total_copies) return;
    // 逐字节拷贝，产生大量细碎 PCIe 事务
    uint8_t* s = src + src_idx * 656;
    uint8_t* d = dst + dst_idx * 656;
    for (int i = 0; i < 656; ++i) {
        d[i] = s[i]; 
    }
}
```
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
可以看到：初始版本的实现，性能相当拉跨，实现的带宽大概只有 2GB/s。简单分析代码可知，最大的瓶颈点在于：原本的代码是串行执行地，压根没有向量化的 IO 操作。原始 Kernel 使用 `uint8_t` 指针进行逐字节拷贝，导致 GPU 发出大量 1-byte 或 4-byte 的细碎 PCIe 请求，TLP (Transaction Layer Packet) Header 开销巨大，总线利用率极低。

---
# V1：向量化拷贝
针对 Baseline 中存在的细碎访问问题，我们利用 `BYTES_PER_TOKEN = 656` 可被 16 整除的特性，强制将指针转换为 `int4` (128-bit)。使用 `LDG.128` 和 `STG.128` 指令替代原有的标量访问，将单 Token 的循环拷贝次数从 656 次骤降至 41 次。这减少了指令发射数量，初步缓解了 PCIe 的压力。

**代码改动 (Vectorization):**
```C++
// 优化：强转为 int4 (16 Bytes) 进行拷贝
__global__ void v1_kernel(...) {
    // ... 索引计算 ...
    // 指针强转
    const int4* src_vec = reinterpret_cast<const int4*>(src + src_idx * 656);
    int4* dst_vec = reinterpret_cast<int4*>(dst + dst_idx * 656);
    // 循环次数 656 -> 41
    #pragma unroll
    for (int i = 0; i < 41; ++i) {
        dst_vec[i] = src_vec[i];
    }
}
```
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

---
# V2: Warp-Level 读取
虽然 V1 提升了带宽，但单线程处理单 Token 无法形成完美的合并访问 (Coalescing)，PCIe 控制器看到的仍是离散的 16-byte 请求。
在此版本中，我们改为 1 Warp (32 Threads) 处理 1 Token。Warp 内 32 个线程同时发起读取，GPU 内存单元将其合并为 512 Bytes 的大块请求，对 PCIe MPS (Maximum Payload Size) 更友好。

**代码改动 (Warp-Level Parallelism):**
```C++
// 优化：一个 Warp 处理一个 Token
__global__ void v2_kernel(...) {
    int warp_id = global_tid / 32;
    int lane_id = global_tid % 32;
    // 由 Warp ID 决定处理哪个 Token
    // ... 索引计算 ...
    // Round 1: 0-512B (32 threads * 16B)
    dst_vec[lane_id] = src_vec[lane_id];
    // Round 2: 512-656B (剩余 144B, 需 9 个线程)
    if (lane_id < 9) {
        dst_vec[32 + lane_id] = src_vec[32 + lane_id];
    }
}
```
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

---
# V3： 流水线优化
PCIe 访问延迟极高 (大于 1us)，Warp 发出请求后会长时间 Stall 等待数据。为进一步提升吞吐，我们引入了指令级并行 (ILP) 优化，采用 Load-Load-Store-Store 模式。利用寄存器重命名，一次性发出所有 Load 请求，允许 2 个 PCIe Read Request 同时在总线上飞行 (In-flight)，从而有效掩盖访问延迟。

**代码改动 (ILP Pipeline):**
```C++
// 优化：分离 Load 和 Store 阶段
// ...
    int4 r1, r2;
    // Step A: Issue ALL Loads (填充流水线)
    r1 = src_vec[lane_id];
    if (lane_id < 9) {
        r2 = src_vec[32 + lane_id];
    }
    // Step B: Issue ALL Stores
    dst_vec[lane_id] = r1;
    if (lane_id < 9) {
        dst_vec[32 + lane_id] = r2;
    }
// ...
```
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

---
# V4: 压缩SM 占用
为了避免一个 IO 算子，就把所有SM 吃满，我们这里给出一个启发式的SM 的占用限制：固定使用最多 16 个 SM。具体实现上，采用了 Grid-Stride Loop 和 Persistent Threads 技术，让每个 Warp 循环处理多个 Token，在限制 Block 数量的同时减少了 Kernel Launch 的开销。

**代码改动 (Grid-Stride Loop):**
```C++
// Kernel 内部：增加跨步循环
int total_warps_in_grid = (gridDim.x * blockDim.x) >> 5;
for (int token_id = warp_id; token_id < total_copies; token_id += total_warps_in_grid) {
    // ... V3 的流水线拷贝逻辑 ...
}
// Host 端 Launch 配置：
int sm_limit = 16;
int blocks_num = min(calculated_blocks, sm_limit * blocks_per_sm);
```
实测发现，实现的 IO 降低的比例不大，其实瓶颈已经卡在 PCIE 的控制器上了。

**瓶颈分析 (Roofline Analysis)：**
在 PCIe Gen5 x16 环境下，实测带宽稳定在 44.5 GB/s，距离物理极限 (63 GB/s) 仍有差距。根本原因在于**内存非对齐 (Misalignment)**。Token 大小为 656 Bytes，不是 Cache Line (128 Bytes) 的倍数。这导致几乎每个 Token 的读取都会跨越 Cache Line 边界，触发**拆分事务 (Split Transactions)**，PCIe Root Complex 必须将 1 个逻辑请求拆分为 2 个物理 TLP，导致 TLP Header 开销翻倍，从而锁死了有效带宽上限。

测试效果如下：
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
  Avg Latency: 3.6577 ms
  Throughput:  43.79 GB/s
----------------------------------------
Sanity Check Passed: src[200957] == dst[0] == 194
```

---
# 总结
通过从标量串行访问到向量化、Warp 级合并访问及指令级流水线的迭代优化，在 PCIe Gen5 x16 环境下，算子吞吐量从基线的 **$1.97 \text{ GB/s}$** 提升至 **$43.80 \text{ GB/s}$**，实现了约 **$22$ 倍**的性能跃升。最终性能瓶颈被定位为数据粒度（656 Bytes）与 Cache Line（128 Bytes）非对齐导致的 PCIe 事务层（TLP）拆分损耗，已经达到了优化的上限，且兼顾了双流场景下的并发资源调度。

---

# 附录
bench 代码：
```Python
#!/usr/bin/env python3
import argparse
import sys
import torch
from pathlib import Path
import time
def _add_repo_path() -> None:
    script_dir = Path(__file__).resolve()
    repo_root = script_dir.parents[2]
    python_root = repo_root / "BAIDU_REPO" / "aiak_sglang_offload" / "python"
    sys.path.insert(0, str(python_root))
def main() -> None:
    _add_repo_path()
    # Import the operator
    try:
        from sglang.srt.utils_op.kv_offload.intra_layer_h2d import IntraLayerH2D
    except ImportError as e:
        print(f"Error importing IntraLayerH2D: {e}")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="Benchmark fast_intra_layer_h2d CUDA op.")
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    parser.add_argument("--topk", type=int, default=2048, help="Number of tokens to copy per request")
    parser.add_argument("--total-cpu-slots", type=int, default=300000, help="Total slots in CPU KV buffer")
    parser.add_argument("--total-gpu-slots", type=int, default=300000, help="Total slots in GPU sparse buffer")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations for benchmarking")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Run a single kernel for ncu capture",
    )
    args = parser.parse_args()
    # Constants from the CUDA kernel
    BYTES_PER_TOKEN = 656
    print(f"Benchmarking fast_intra_layer_h2d with:")
    print(f"  BS: {args.bs}")
    print(f"  TopK: {args.topk}")
    print(f"  Total Copies: {args.bs * args.topk}")
    print(f"  Data Size: {args.bs * args.topk * BYTES_PER_TOKEN / 1024 / 1024:.2f} MB")
    # 1. Prepare Data
    # To avoid collisions in sanity check, total_slots should be >= bs * topk
    total_needed = args.bs * args.topk
    actual_cpu_slots = max(args.total_cpu_slots, total_needed)
    actual_gpu_slots = max(args.total_gpu_slots, total_needed)
    # CPU Buffer (Pinned)
    kv_cpu = torch.randint(
        0, 255, 
        (actual_cpu_slots, 1, BYTES_PER_TOKEN), 
        dtype=torch.uint8
    ).pin_memory()
    # GPU Buffer
    sparse_gpu = torch.zeros(
        (actual_gpu_slots, 1, BYTES_PER_TOKEN), 
        dtype=torch.uint8, 
        device="cuda"
    )
    # Dummy cpu_slot_to_gpu_slot
    cpu_slot_to_gpu_slot = torch.zeros(
        (actual_cpu_slots,), 
        dtype=torch.int32, 
        device="cuda"
    )
    # Evict IDs: [bs, topk, 2] -> (src_idx, dst_idx)
    # Use unique indices for dst to avoid race conditions
    src_indices = torch.randint(
        0, actual_cpu_slots, 
        (args.bs, args.topk), 
        dtype=torch.int32, 
        device="cuda"
    )
    # Ensure unique destination slots for sanity check
    dst_indices = torch.arange(total_needed, dtype=torch.int32, device="cuda").reshape(args.bs, args.topk)
    evict_ids = torch.stack([src_indices, dst_indices], dim=-1) # [bs, topk, 2]
    op = IntraLayerH2D()
    # 2. NCU Mode
    if args.ncu:
        print("Running in NCU mode (1 iteration)...")
        # Ensure data is ready
        torch.cuda.synchronize()
        op.cuda_impl(kv_cpu, sparse_gpu, cpu_slot_to_gpu_slot, evict_ids)
        torch.cuda.synchronize()
        print("Done.")
        return
    # 3. Warmup
    print(f"Warming up for {args.warmup} iterations...")
    for _ in range(args.warmup):
        op.cuda_impl(kv_cpu, sparse_gpu, cpu_slot_to_gpu_slot, evict_ids)
    torch.cuda.synchronize()
    # 4. Benchmark Loop
    # We use CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    print(f"Benchmarking for {args.iters} iterations...")
    start_event.record()
    for _ in range(args.iters):
        op.cuda_impl(kv_cpu, sparse_gpu, cpu_slot_to_gpu_slot, evict_ids)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / args.iters
    # 5. Calculate Bandwidth
    total_bytes = args.bs * args.topk * BYTES_PER_TOKEN
    total_gb = total_bytes / (1024**3)
    avg_s = avg_ms / 1000.0
    bw_gbps = total_gb / avg_s if avg_s > 0 else 0
    print("-" * 40)
    print(f"Results:")
    print(f"  Avg Latency: {avg_ms:.4f} ms")
    print(f"  Throughput:  {bw_gbps:.2f} GB/s")
    print("-" * 40)
    # 6. Basic Validation (Optional but good for sanity check)
    # Check the first copy of the last iteration
    # To avoid high overhead, we just check one element on CPU side vs GPU side
    # after synchronization.
    # Pick the first task of the first batch
    check_src_idx = src_indices[0, 0].item()
    check_dst_idx = dst_indices[0, 0].item()
    # Copy back a small slice from GPU to CPU to verify
    # We only check if the first byte matches to avoid massive transfers
    gpu_val = sparse_gpu[check_dst_idx, 0, 0].item()
    cpu_val = kv_cpu[check_src_idx, 0, 0].item()
    if gpu_val == cpu_val:
        print(f"Sanity Check Passed: src[{check_src_idx}] == dst[{check_dst_idx}] == {cpu_val}")
    else:
        print(f"Sanity Check FAILED: src[{check_src_idx}]={cpu_val} != dst[{check_dst_idx}]={gpu_val}")
if __name__ == "__main__":
    main()
```