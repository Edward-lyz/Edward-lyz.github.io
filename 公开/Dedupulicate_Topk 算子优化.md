# 背景：
在 `kvoffload` 这个版本的 `sglang` 的推理的投机采样（Speculative Sampling）阶段，引擎需要对生成的 `TopK` 候选 `Token` 索引进行去重，这就引入了本次的优化算子。

**算子语义**：
- 对每个 `batch` 将 `mtp_step` 个 `top‑k index` 行做去重，输出长度固定为 `mtp_step * k`，不足用 -1 填充

**场景特征**：
- **小 Batch**：`bs=115`，GPU 并行度（Occupancy）天然不足。
- **延迟敏感**：这是在线服务链路的一环，对 Latency 极度敏感。
- **输入规模**：固定几档（Total elements = 2048, 4096...），适合特化。
我们经历了一个从“Naive 实现”到“指令级 Hacking”的完整过程，最终将耗时压缩了 **120倍** ，从 **2ms** 压缩至 **16us**。
---
# 基线：直觉的陷阱 (Shared Hash Table)
当我们拿到“去重”这个需求时，第一反应通常是：**哈希表**。这在 CPU 上是标准答案（$O(N)$）。于是前人就写出了第一版 `Kernel`，在 `Shared Memory` 中维护一个哈希表。
## 代码片段 (Naive Implementation)

```C++
__global__ void deduplicate_topk_kernel_v0(
    const int* topk_indices, 
    int* topk_indices_spec, 
    int total_elements, 
    int k) 
{
    // ... setup shared memory ...
    extern __shared__ int shared_mem[];
    int* seen = shared_mem; // 哈希表
    // 初始化哈希表
    for (int i = tid; i < total_elements; i += blockDim.x) seen[i] = -1;
    __syncthreads();
    // 循环处理每个元素
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int val = topk_indices[row * k + col];
        // 简单的线性探测哈希
        int hash_idx = val % total_elements;
        // 【致命点1】探测循环：可能 1 次成功，也可能 10 次
        // 导致 Warp 内线程严重发散 (Divergence)
        for (int probe = 0; probe < total_elements; probe++) {
            int idx = (hash_idx + probe) % total_elements;
            // 【致命点2】原子操作竞争：所有线程抢着写
            int old = atomicCAS(&seen[idx], -1, val);
            if (old == -1) { // 抢到了，是新元素
                int pos = atomicAdd(unique_count, 1);
                unique_list[pos] = val; // 【致命点3】随机写，非合并访存
                break;
            } else if (old == val) {
                break; // 已存在
            }
        }
    }
}
```
## 为什么慢？
NCU 的报告显示：
1. **Warp Divergence (发散)**：哈希冲突是随机的。Warp 里有的线程 1 次 probe 就结束了，有的线程要 probe 10 次。根据 SIMT 机制，整个 Warp 必须陪着最慢的那个线程空转。
2. **Memory Coalescing (非合并)**：哈希表的读写是完全随机的，显存带宽利用率（Memory Throughput）只有可怜的 **4%**。
3. **Atomic Contention (原子竞争)**：数百个线程同时轰炸 `atomicCAS`，硬件序列化严重。
**初步结论**：在 GPU 上，**“确定的执行路径”** 往往比理论上的算法复杂度更重要。哈希表在 GPU 上通常不是好选择。同时，考虑到推理阶段的特性，输入的 2048 个索引中，很可能重复的索引并不多，导致了哈希表的冲突比理想情况糟糕得多。
---
## 1. V1 ：范式切换 (Sort + Unique)
为了消除“随机性”，我们决定改用 **CUB (CUDA Unbound)** 库，采用 **排序 (Sort) -> 查重 (Unique)** 的范式。
虽然排序理论复杂度是 $O(N \log N)$，但它能让数据访问变成完美的线性流，且 Warp 内所有线程行为一致。
## 前置知识：CUB 三板斧
在重构后的第一版代码中，我们用到了三个核心原语：
1. **`BlockRadixSort`**: 块内基数排序。把乱序数据排整齐，这是去重的前提。
2. **`BlockDiscontinuity`**: 找不同。比较 `data[i]` 和 `data[i-1]`，如果不同，说明是 Unique 的。
3. **`BlockScan`**: 前缀和。算出每个 Unique 元素应该写到输出数组的哪个位置。
## 代码片段 (Standard CUB)
```C++
// 寄存器大户：CUB 需要一个数组来存每个元素的状态
int unique_flags[kItemsPerThread]; // Flag: 1=Unique, 0=Duplicate
// 1. 排序
BlockRadixSortT(temp_storage).Sort(values);
// 2. 找不同 (FlagHeads)
// CUB 自动处理了跨线程的边界比较，结果存入 unique_flags 数组
BlockDiscontinuityT(temp_storage).FlagHeads(unique_flags, values, NotEqual());
// 3. 计数
int unique_count = 0;
#pragma unroll
for (int i = 0; i < kItemsPerThread; ++i) unique_count += unique_flags[i];
// 4. 前缀和计算写入偏移 (Offset)
BlockScanT(temp_storage).ExclusiveSum(unique_count, thread_offset, total_unique);
// 5. 写回 (Scatter)
#pragma unroll
for (int i = 0; i < kItemsPerThread; ++i) {
    if (unique_flags[i]) {
        // 依然依赖 flags 数组做判断
        output[thread_offset + local_offset++] = values[i];
    }
}
```

**结果**：性能提升到 **~28us**。但 NCU 警告 **Register Pressure** 过高，主要因为 `int unique_flags[16]` 占用了太多寄存器。

---
# V2：IO 瓶颈突破 (Vectorized Load & SMEM Staging)
在 V1 切换到排序算法后，计算复杂度已经不再是核心矛盾。NCU (Nsight Compute) 的 `profiling` 结果显示，**Memory Throughput（显存吞吐）** 成为了新的拦路虎。
## 2.1 读操作：强制向量化 (Vectorized Load)

V1 版本中，每个线程通过循环逐个读取 `int` 数据。虽然看似简单，但这导致大量的指令开销和带宽浪费。编译器在复杂逻辑下往往无法自动合并访存。
**优化手段**： 我们手动将输入指针强转为 `int4` 类型（128-bit），迫使 GPU 生成 `LDG.E.128` 指令，一条指令就能搬运 4 个 int，指令数直接砍掉 75%。

**代码对比**：
```C++
// Before: 标量读取 (Scalar Load)
// 产生大量 LDG.E.32 指令，带宽利用率低
int values[kItemsPerThread];
for (int i = 0; i < kItemsPerThread; ++i) {
    values[i] = input[tid + i * blockDim.x];
}
// After: 向量化读取 (Vectorized Load)
// 产生 LDG.E.128 指令，一次搬运 16 字节
const int4* vec_ptr = reinterpret_cast<const int4*>(input);
int4 loaded = vec_ptr[tid + i * blockDim.x]; 
values[0] = loaded.x; values[1] = loaded.y; ...
```
## 2.2 写操作：Shared Memory 中转 (Staging & Vectorized Store)
**痛点分析**： 读进来容易，**写出去（Scatter）** 却是个大麻烦。 去重后的数据在逻辑上是紧凑的，但在线程视角是离散的。例如：
- Thread 0 可能需要写输出位置 `[0, 1, 2]`
- Thread 1 可能需要写输出位置 `[15, 16]`（因为中间有些元素被去重了）
如果我们直接往全局内存 (GMEM) 写，这种**离散写（Scattered Write）** 会导致显存控制器收到大量碎片化的请求，编译器完全无法合并这些写操作，导致 `Global Store Efficiency` 极低。
**优化策略：SMEM Staging** 为了解决这个问题，我们引入了 **Shared Memory (SMEM)** 作为中转站：
1. **Gather to SMEM**：线程先把去重后的数据写入 SMEM。虽然这也是离散写，但 SMEM 就在片上，带宽高且延迟极低，这点开销可以忽略。
2. **Coalesced Store to GMEM**：所有线程同步后，从 SMEM 中**连续**读取数据，并利用 `int4` 向量化指令，整齐划一地写回 GMEM。

**代码对比**：
```C++
// Before: 直接离散写回 GMEM (低效)
// 线程写的位置跳跃，无法合并，带宽浪费严重
if (is_unique) {
    gmem_output[global_offset + local_idx] = val; 
}
// After: SMEM 中转 + 向量化写回 (高效)
// Step 1: 先写到 SMEM (片上极速)
if (is_unique) {
    smem_buffer[local_idx] = val;
}
__syncthreads();
// Step 2: 连续、向量化写回 GMEM
// 此时 idx 是连续的 (tid * 4)，可以生成高效的 STG.128 指令
int4* out_vec = reinterpret_cast<int4*>(gmem_output);
const int4* smem_vec = reinterpret_cast<const int4*>(smem_buffer);
out_vec[tid] = smem_vec[tid]; // 完美合并写
```
## 2.3 内存布局小技巧：SMEM 复用
这一步优化还带来了一个额外的好处：**显存复用**。 CUB 的排序操作 (`BlockRadixSort`) 需要一块临时的 SMEM (`temp_storage`)。当排序完成后，这块内存就闲置了。我们巧妙地让 `smem_buffer` (用于中转输出) 的地址复用了 `temp_storage` 的一部分空间（或者紧挨着它，视具体生命周期而定）。这样既实现了 Staging 优化，又没有增加额外的 Shared Memory 占用，避免了因为 SMEM 超限导致 Occupancy 下降。

**成效**： 通过这一套“向量化读 + SMEM 中转 + 向量化写”的组合拳，NCU 显示 Memory Throughput 飙升至 **100 GB/s+**，Kernel 耗时稳定在 **19us** 左右。

---
# V3：位运算 (Bitmask Magic & \_\_popc)
这是本次优化的“高光时刻”。虽然 V3 很快，但 V1 遗留的 `unique_flags` 数组依然导致寄存器压力大，进而限制了 GPU 的 Occupancy 或者导致指令溢出 (Spill)。
**思考**：我们需要存 16 个 boolean 状态，真的需要 16 个 `int` (512 bits) 吗？
**答案**：不需要。16 个 bit 就够了。我们可以用一个 `uint32_t` 搞定。
这就引入了 **Bitmask** 和 **`__popc`** 技术。
## 核心优化解析
1. **Bitmask (位掩码)**：
    用一个 `uint32_t mask` 的二进制位表示状态。
    - 第 `i` 位是 1 $\to$ 是 Unique 元素。
    - 第 `i` 位是 0 $\to$ 是 Duplicate 元素。
2. **`__popc` (Population Count)**：
    CUDA 硬件指令，**单周期**计算一个二进制数里有多少个 1。
    - `__popc(0b101) = 2`。
3. **Magic Scatter (无依赖计算)**：
    **如何知道第 `i` 个元素该写到 Output 的哪个位置？** 只需要知道 **"在 `i` 之前有多少个 1"**。 这可以通过位运算瞬间算出：`mask & ((1 << i) - 1)`。
	- `(1 << i) - 1`：生成一个低 `i` 位全为 1 的掩码。
	- `&` 操作：保留 `mask` 中低 `i` 位的状态，屏蔽高位。
	- `__popc(...)`：数一下剩下的 1 有多少个，这就是 `Local Offset`
## 代码片段 (Bitmask Magic)
```C++
// 【优化点1】砍掉数组，只用 1 个寄存器
uint32_t unique_mask = 0; 
// 手写 Discontinuity 逻辑，替代 CUB
// ... (省略跨线程边界交换代码) ...
// 生成掩码：如果当前元素 != 上一个，则置位
#pragma unroll
for (int i = 0; i < kItemsPerThread; ++i) {
    if (val != prev && val != sentinel) {
        unique_mask |= (1u << i); // 设置第 i 位为 1
    }
}
// 【优化点2】用 __popc 替代循环计数
int unique_count = __popc(unique_mask);
BlockScanT(temp_storage).ExclusiveSum(unique_count, thread_offset, total_unique);
// 【优化点3】极速 Scatter
#pragma unroll
for (int i = 0; i < kItemsPerThread; ++i) {
    // 检查第 i 位是否为 1
    if (unique_mask & (1u << i)) {
        // Magic: 计算低 i 位有多少个 1，直接得到局部偏移
        // (1u << i) - 1 生成形如 000...011111 的掩码
        int local_idx = __popc(unique_mask & ((1u << i) - 1));
        // 直接写入 Shared Memory，无循环依赖
        output_buffer[thread_offset + local_idx] = values[i];
    }
}
```
**成效**：
通过这一手操作，我们将指令数（Instruction Count）大幅压缩，从 V4 的 2.89M 降回 2.04M。在保持高带宽的同时，极大地降低了计算流水线的压力。最终耗时来到：**16us**

---
# 总结
1. **不要在 GPU 上做细粒度哈希**。排序+线性扫描通常是更好的选择。
2. **带宽不够，向量化来凑**。`int4` 是提升吞吐的利器。
3. **当通用库 (CUB) 成为瓶颈时，手写位操作 (Bitmask) 往往能带来奇迹**。

# 附录
下面附上最小的 `bench` 脚本，可以交叉验证。
```python
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
def _add_repo_path() -> None:
    script_dir = Path(__file__).resolve()
    repo_root = script_dir.parents[2]
    python_root = repo_root / "BAIDU_REPO" / "aiak_sglang_offload" / "python"
    sys.path.insert(0, str(python_root))
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark deduplicate_topk CUDA op.")
    parser.add_argument("--bs", type=int, default=115)
    parser.add_argument("--mtp-step", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=1 << 31)
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Run a single kernel for ncu capture",
    )
    args = parser.parse_args()
    _add_repo_path()
    import torch
    from sglang.srt.utils_op.kv_offload.deduplicate_topk import DeduplicateTopk
    total = args.bs * args.mtp_step
    x = torch.randint(
        0,
        args.vocab_size,
        (total, args.k),
        device="cuda",
        dtype=torch.int32,
    )
    op = DeduplicateTopk()
    if args.ncu:
        _ = op.cuda_impl(x, args.mtp_step)
        torch.cuda.synchronize()
        print(f"bs={args.bs} mtp_step={args.mtp_step} k={args.k}")
        return
    for _ in range(args.warmup):
        _ = op.cuda_impl(x, args.mtp_step)
    torch.cuda.synchronize()
    y_cuda = op.cuda_impl(x, args.mtp_step)
    y_aten = op.aten_impl(x, args.mtp_step)
    torch.cuda.synchronize()
    is_match = bool(torch.equal(y_cuda, y_aten))
    print(f"check_equal={is_match}")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _ = op.cuda_impl(x, args.mtp_step)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / args.iters
    print(f"bs={args.bs} mtp_step={args.mtp_step} k={args.k} total={total}")
    print(f"avg_ms_cuda_graph={avg_ms:.4f}")
if __name__ == "__main__":
    main()
```