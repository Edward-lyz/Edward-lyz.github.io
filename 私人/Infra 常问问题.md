# A. 硬件和算子

## A1. Hopper 架构相比 Ampere 有什么区别？Blackwell 呢？

**考察目的**

这题不是考你背 A100/H100/B200 参数表，而是看你能不能把架构变化映射到主流 AI 算子的瓶颈。面试官真正想听的是：为什么 Ampere 需要 `cp.async`，为什么 Hopper 需要 TMA/WGMMA/mbarrier，为什么 Blackwell 又引入 TMEM、`tcgen05`、2CTA MMA。

**第一性原理**

以 GEMM 为例，数据流是：

```text
HBM / L2
  → shared memory
  → register / TMEM
  → Tensor Core
  → accumulator
  → epilogue
  → global memory
```

每一代 GPU 的改进，本质都是在缩短这条链路、提高复用、减少同步、降低寄存器压力、提升 Tensor Core 被喂饱的概率。

**回答**

Ampere 的关键是把 `global → shared` 的搬运异步化。过去很多 load 要经过 register，再写 shared，既占用指令，也占用 register file。Ampere 引入 `cp.async` / `cuda::memcpy_async`，允许 global memory 到 shared memory 的异步拷贝和计算重叠，并且可以避免额外寄存器中转。NVIDIA Ampere Tuning Guide 明确说明 Ampere 增加了 global-to-shared 异步拷贝硬件能力，能显式 overlap compute 和 data movement。([NVIDIA Docs][1])

Hopper 的核心是：手工 `cp.async` 和 warp-level MMA 已经不够。Transformer/GEMM 的 tile 越来越大，copy 需要更多地址生成和同步，MMA 也需要更大执行粒度。因此 Hopper 引入 TMA、WGMMA、mbarrier、thread block cluster。TMA 负责大块多维 tensor copy，WGMMA 让 warpgroup 级别发 Tensor Core 指令，mbarrier 管异步 copy 和 compute 的同步。NVIDIA Hopper 架构资料明确把 TMA 和 asynchronous transaction barrier 作为 Hopper 异步执行和 memory/compute overlap 的关键特性。([NVIDIA Developer][2])

Blackwell 的核心可以从 “register accumulator 成为瓶颈” 推出来。Hopper WGMMA 的 accumulator 很大，寄存器压力会影响 occupancy、epilogue 和 pipeline。Blackwell SM100 引入 `tcgen05.mma`、TMEM、1CTA/2CTA MMA 等机制，把部分矩阵累加状态放到 Tensor Memory，而不是完全压在普通 register 上。CUTLASS Blackwell 文档也明确把 `tcgen05.mma` 和 Blackwell SM100 GEMM 作为新能力。([NVIDIA Docs][3])

**发散思考**

这题可以继续追问：

```text
cp.async 和 TMA 的本质区别？
mma.sync、wgmma.mma_async、tcgen05.mma 的执行粒度差异？
为什么 TMEM 可以缓解 register pressure？
2CTA MMA 解决的是数据复用、同步，还是 Tensor Core 喂数问题？
```

强回答的主线是：

```text
Ampere：global → shared 异步化
Hopper：copy 和 MMA 都 warpgroup 化、异步化
Blackwell：accumulator 和 CTA 协作进一步硬件化，降低 register pressure 和跨 CTA 协作成本
```

---

## A2. 有什么办法可以增加访存效率？

**考察目的**

考你是否理解 GPU 性能瓶颈往往不是算力，而是数据搬运。面试官想看你能不能分清 global memory coalescing、L2 reuse、shared memory bank conflict、pipeline overlap 各自解决什么问题。

**第一性原理**

访存效率可以粗略写成：

```text
访存效率 = 实际有用字节 / 实际搬运字节
```

GPU 不是按变量搬数据，而是按 transaction、sector、cache line、memory partition 搬数据。warp 内地址越连续、越对齐，请求越容易 coalesce；CTA 之间 reuse distance 越短，L2 命中越高；shared memory bank 越分散，片上访存越高效。

**回答**

常见优化方法：

```text
1. coalescing：warp 内线程访问连续地址
2. alignment：地址按 16B / 32B / 128B 对齐
3. vectorized load/store：减少访存指令数量
4. shared memory tiling：把 HBM 数据搬到片上多次复用
5. threadblock swizzle：改变 CTA 执行顺序，提高 L2 reuse
6. shared memory swizzle / padding：减少 bank conflict
7. cp.async / TMA：把搬运和计算重叠
8. L2 persistence：让高复用数据更倾向留在 L2
9. fusion：减少中间结果写回 HBM
10. prefetch：提前加载下一 tile
```

threadblock swizzle 的本质是跨 CTA 的 cache locality。比如 GEMM：

```text
C[M,N] = A[M,K] @ B[K,N]
```

如果连续 CTA 沿 N 方向走，可能复用 A tile；如果连续 CTA 沿 M 方向走，可能复用 B tile。swizzle 不是改变单个 CTA 内部的访存，而是改变 CTA 执行顺序，让刚进入 L2 的数据尽快被其他 CTA 用到。

**发散思考**

可以继续问：

```text
threadblock swizzle 和 shared memory swizzle 区别？
为什么 coalesced access 不一定 L2 hit 高？
为什么 L2 hit 高但 kernel 仍然慢？
vector load 为什么可能增加无效带宽？
```

一句强回答：

> 访存优化要先定位浪费发生在哪一层：global transaction、L2 reuse、shared bank、register pressure，还是 copy/compute overlap。threadblock swizzle 主要优化跨 CTA 的 L2 复用。

---

## A3. GPU 的 memory 架构是什么样的？

**考察目的**

考你是否能分清 GPU memory hierarchy，而不是只会说 register/shared/global。更深层是考你是否知道不同层级对应不同优化手段。

**第一性原理**

存储层级的基本规律是：

```text
越近：越快、越小、越需要程序员精细控制
越远：越大、越慢、越依赖并行度和带宽掩盖延迟
```

**回答**

从近到远：

```text
register：
    每个 thread 私有，最快，但太多会降低 occupancy。

local memory：
    线程私有地址空间，通常是 register spill 后落到 global memory 体系。

shared memory：
    CTA 内共享，程序员显式管理，适合 tiling，但有 bank conflict。

L1 / TEX cache：
    SM 局部自动缓存，服务 global/local load。

L2 cache：
    全 GPU 共享，是跨 SM global memory 访问的重要汇聚点。

HBM：
    GPU device memory，带宽高但延迟远高于片上存储。

NVLink / PCIe / Host memory：
    跨 GPU 或 CPU-GPU 访问，延迟更高。
```

warp 发出访存请求时，大致路径是：

```text
warp memory instruction
  → coalescer
  → L1/TEX
  → L2
  → HBM partition / memory controller
```

**发散思考**

可以继续追问：

```text
shared memory 和 L1 是否共享物理资源？
local memory 为什么名字叫 local 却很慢？
L2 为什么是跨 SM 通信的重要一致性点？
寄存器越多为什么 occupancy 越低？
```

---

## A4. 什么是 sector？为什么要 sector？

**考察目的**

考你是否理解 GPU 访存的真实粒度。很多人会说 “cache line 是 128B”，但不知道实际填充和统计经常以 32B sector 为粒度。

**第一性原理**

warp 内线程访问的地址可能连续，也可能只覆盖 cache line 的一小部分。如果每次都搬完整 128B，稀疏访问会浪费大量带宽；如果每个 thread 都单独搬 4B，请求数量又太多。sector 是折中。

**回答**

sector 可以理解为 cache line 的子块。Nsight Compute 文档把 sector 定义为 cache line 或 device memory 中 aligned 32-byte chunk，并说明一个 L1 或 L2 cache line 是 4 个 sector，也就是 128B。([NVIDIA Docs][4])

例如：

```text
32 个线程各读 1 个 float
32 × 4B = 128B
```

如果地址连续且对齐，刚好覆盖 4 个 32B sector，效率高。

如果只有 8 个线程读连续 float：

```text
8 × 4B = 32B
```

那只需要 1 个 sector，而不是整条 128B cache line。

**发散思考**

这题常追问 vector load：

```text
float4 load 是否一定比 float load 高效？
如果 float4 跨 sector 边界怎么办？
sector/request 高说明什么？
dram sectors 和 L1 sectors 有什么区别？
```

强回答：

> sector 是 GPU 为 SIMT 访存设计的细粒度搬运单位，目的是减少 cache line 内无效字节搬运，同时避免每个 thread 独立发小请求。

---

## A5. 什么是 bank conflict？

**考察目的**

这是 shared memory 最基础问题。面试官想确认你是否理解 shared memory 高带宽来自 bank 并行，而不是 magic fast memory。

**第一性原理**

shared memory 被拆成多个 bank。一个 warp 同时访问 shared memory 时，如果不同线程访问不同 bank，可以并行；如果多个线程访问同一 bank 的不同地址，就要拆成多次请求。

**回答**

CUDA 文档说明：如果一个 memory request 的两个地址落到同一个 shared memory bank，就发生 bank conflict，硬件会把请求拆成多个无冲突请求，吞吐按拆分次数下降。([NVIDIA Docs][5])

可以近似理解为：

```text
bank_id = (address / bank_width) % num_banks
```

例子：

```cpp
// 通常无冲突
x = smem[threadIdx.x];

// 可能 32-way conflict
x = smem[threadIdx.x * 32];
```

但如果多个线程访问同一个 bank 的同一个地址，可能触发 broadcast，不一定是 conflict。

**发散思考**

常见解决方法：

```text
padding：smem[M][N] → smem[M][N+1]
layout swizzle：逻辑连续，物理打散
改变访问 stride
使用适配 ldmatrix / wgmma 的 shared layout
```

强回答：

> shared memory 的本质是 bank-level parallelism；bank conflict 就是多个线程竞争同一 bank 的不同地址，导致一次 warp memory instruction 被串行化。

---

## A6. 为什么要用 vector 指令？vector 指令大小对访存效率有什么影响？

**考察目的**

考你是否知道 vectorized load/store 的收益和风险。不是 `float4` 一定快，而是要看 sector 覆盖、对齐、无效字节、寄存器压力。

**第一性原理**

vector 指令优化的是：

```text
1. 指令数量
2. 地址计算数量
3. 访存规整性
```

但底层搬运仍然按 sector/cache line 服务。

**回答**

例如：

```cpp
float4 x = reinterpret_cast<float4*>(ptr)[i];
```

一个线程一次读 16B，比 4 次 `float` load 指令更少。warp 内如果地址连续且 16B 对齐，硬件能生成规整 sector 请求。

但 vector size 不是越大越好。真实效率是：

```text
有效使用字节 / 实际覆盖 sector 字节
```

如果你 load `float4` 只用其中一个 float，就是浪费 12B。如果 vector 跨 32B sector 或 128B line 边界，会产生更多 sector。如果 vector 太大，还可能增加 register pressure，降低 occupancy。

**发散思考**

面试官可能继续问：

```text
为什么 int4 / float4 load 要求 alignment？
vector load 对 L1 sector 有什么影响？
什么时候 scalar load 反而更好？
vectorized store 对 epilogue 有什么要求？
```

一句话：

> vector 指令减少指令开销，但访存效率仍由对齐、连续性、sector 覆盖率和是否真正使用所有字节决定。

---

## A7. 为什么要做 persistence？

**考察目的**

考你是否能区分 L2 persistence 和 persistent kernel。二者名字相近，但一个是数据复用，一个是执行上下文复用。

**第一性原理**

persistence 的本质是：

```text
如果某个东西会被重复使用，就不要频繁驱逐或重复初始化。
```

**回答**

第一种是 **L2 cache persistence**。CUDA 支持 L2 access policy window，可以指定一段 global memory 区域，使其访问更倾向在 L2 中持久保留。CUDA Programming Guide 明确说明 access policy window 可以为一段连续 global memory 设置 L2 persistence 属性。([NVIDIA Docs][6])

适合：

```text
反复访问的权重
KV cache metadata
GEMM 中跨 CTA 复用的 A/B tile
routing table
scale / zero point
```

第二种是 **persistent kernel / persistent CTA**。不是每个 CTA 算一个固定 tile 后退出，而是让 CTA 常驻 SM，从全局 work queue 持续取任务。适合任务不均匀或 tile 数不足的场景：

```text
decode attention
MoE
variable-length sequence
stream-K GEMM
small-batch GEMM
```

**发散思考**

可以追问：

```text
persistent kernel 和 occupancy 什么关系？
为什么 persistent kernel 适合 decode？
L2 persistence 什么时候反而有害？
persistent CTA 如何做 load balance？
```

强回答：

> L2 persistence 复用数据，persistent kernel 复用执行上下文；前者减少 HBM traffic，后者改善调度和负载均衡。

---

## A8. 为什么要做 warp specialization？

**考察目的**

考你是否理解 Hopper 之后的高性能 kernel 不是所有 warp 做同样的事情，而是 producer-consumer pipeline。

**第一性原理**

copy 和 compute 是不同类型的任务：

```text
copy：
    地址生成、TMA descriptor、barrier、shared buffer 管理

compute：
    MMA issue、accumulator、register/TMEM 管理
```

如果同一批 warp 同时做所有事，会增加寄存器压力、同步复杂度和 pipeline 气泡。

**回答**

Hopper GEMM 常见结构：

```text
producer warpgroup:
    TMA global → shared

consumer warpgroup:
    WGMMA shared → Tensor Core

epilogue:
    accumulator → scale / bias / activation → global
```

CUTLASS Efficient GEMM 文档也描述了 warp-level GEMM 从 shared memory load 到 register，再通过 Tensor Core 或 CUDA core 执行计算，并强调 shared memory 访问要避免 bank conflict。([NVIDIA Docs][7])

warp specialization 的收益：

```text
1. copy 和 compute 更好 overlap
2. producer 不需要大 accumulator，降低 register pressure
3. consumer 专注 WGMMA，Tensor Core 更容易喂满
4. pipeline 状态更清晰
```

代价：

```text
1. barrier / mbarrier 复杂
2. pipeline stage 管理复杂
3. 小问题规模可能不划算
```

**发散思考**

常见追问：

```text
warp specialization 和 persistent kernel 如何结合？
producer warpgroup 会不会浪费算力？
如果 TMA latency 很小，还需要 specialization 吗？
```

强回答：

> warp specialization 是把 kernel 从“所有 warp 同构执行”变成“producer-consumer 异步流水”，本质是用更清晰的分工换更好的 overlap 和更低 register pressure。

---

## A9. FlashAttention 几代论文有什么区别？

**考察目的**

考你能不能从瓶颈演进解释算法，而不是背 FA1/FA2/FA3 名词。

**第一性原理**

标准 attention 的中间矩阵是：

```text
S = QK^T
P = softmax(S)
O = PV
```

`S` 和 `P` 都是 `N × N`，对长序列会产生巨大 HBM 读写。FlashAttention 的本质是避免 materialize 这些中间矩阵。

**回答**

FA1 解决 IO。它通过 tiling 和 online softmax，在 SRAM/shared memory 中分块计算 exact attention，避免把完整 attention matrix 写回 HBM。FlashAttention 论文明确提出 IO-aware exact attention，用 tiling 减少 HBM 和 on-chip SRAM 之间的读写。([arXiv][8])

FA2 解决 work partition 和并行度。FA1 省 IO，但不够像 GEMM 一样高效。FA2 减少 non-matmul FLOPs、优化 thread block/warp 分工，并让单个 head 内也能跨 thread blocks 并行。FA2 论文指出 FA1 只能达到 A100 理论 FLOPs 的 25–40%，FA2 通过更好 work partition 达到更高利用率。([arXiv][9])

FA3 解决 Hopper 架构利用率。FA3 利用 Hopper 的 TMA、异步 Tensor Core、warp specialization，把 matmul、softmax、数据搬运交错起来，并支持 FP8 相关优化。FA3 论文明确说核心是利用 Tensor Core 和 TMA 的异步性 overlap computation/data movement、interleave matmul/softmax，以及 FP8 block quantization。([arXiv][10])

**发散思考**

可以继续问：

```text
FA1 为什么 memory 从 O(N²) 降到 O(N)？
FA2 为什么要减少 non-matmul FLOPs？
FA3 为什么特别依赖 Hopper？
FA backward 为什么选择 recomputation？
```

一句话：

> FA1 是 IO-aware，FA2 是 work-partition-aware，FA3 是 architecture-aware。

---

## A10. 简单实现一下 FA v2

**考察目的**

考你是否真的理解 online softmax，而不是只知道 “FlashAttention 不存 attention matrix”。

**第一性原理**

softmax 不需要一次看到完整行。对一行 score，只要维护当前最大值、exp 和、加权 V 的 numerator，就可以分块合并。

**回答**

对一个 Q block，维护：

```text
m    = 当前行最大值
l    = 当前行 exp denominator
acc  = 当前行 softmax numerator
```

每来一个 K/V block：

```text
S = Q_block @ K_block^T * scale

m_new = max(m, rowmax(S))
P     = exp(S - m_new)
alpha = exp(m - m_new)

l_new   = alpha * l + rowsum(P)
acc_new = alpha * acc + P @ V_block
```

最后：

```text
O = acc / l
```

伪代码：

```python
for q_block in Q_blocks:
    Q = load(Q_block)

    m = -inf
    l = 0
    acc = 0

    for kv_block in KV_blocks:
        K = load(K_block)
        V = load(V_block)

        S = Q @ K.T * scale
        S = apply_mask(S)

        m_new = max(m, rowmax(S))
        P = exp(S - m_new[:, None])
        alpha = exp(m - m_new)

        acc = alpha[:, None] * acc + P @ V
        l = alpha * l + rowsum(P)
        m = m_new

    O = acc / l[:, None]
```

FA2 的工程优化在于 tile shape、warp 分工、减少 shared memory 往返、减少非矩阵 FLOPs、提高 occupancy。

**发散思考**

可以追问：

```text
为什么 online softmax 是 exact？
causal mask 如何影响 tile？
backward 为什么可以不保存 P？
如果 KV block 遍历顺序改变，结果是否 bitwise 一样？
```

强回答：

> FA 的正确性来自 online softmax 的等价合并，性能来自不写 N² 中间矩阵和更好的 Tensor Core work partition。

---

## A11. GPU cache 怎么保证一致性？

**考察目的**

考你是否理解 GPU 不是 CPU 式所有 L1 强一致。尤其是跨 CTA 通信时，不能只靠普通 global memory load/store。

**第一性原理**

GPU 追求吞吐，不追求每个 SM 私有 cache 自动强一致。跨线程、跨 CTA、跨设备的可见性必须通过同步语义建立。

**回答**

层级上可以这样理解：

```text
shared memory：
    CTA 内共享，用 __syncthreads / barrier 保证顺序。

L1：
    SM 局部 cache，不应假设跨 SM 自动强一致。

L2：
    全 GPU 共享，是 global memory 和 atomic 的重要一致性点。

global memory：
    跨 CTA 通信需要 atomic、fence、release/acquire、scope。

kernel boundary：
    kernel 结束通常是更强同步边界。
```

CUDA memory model 对 fence、release/acquire 和 scope 有明确规定：release fence 和 acquire fence 需要通过同步对象建立关系，并且 scope 要覆盖参与线程。([NVIDIA Docs][5])

producer-consumer 模式：

```cpp
// producer
data[i] = value;
atomic_store_explicit(&flag, 1, memory_order_release);

// consumer
while (atomic_load_explicit(&flag, memory_order_acquire) == 0) {}
x = data[i];
```

**发散思考**

继续追问：

```text
volatile 能不能替代 acquire-release？
device scope 和 system scope 有什么区别？
atomic 本身保证什么，不保证什么？
为什么同 kernel 内 CTA 间同步很难？
```

强回答：

> GPU cache 一致性主要靠 L2、atomic、fence、memory order 和 scope；不要假设跨 SM 的 L1 自动强一致。

---

## A12. release-acquire 语义

**考察目的**

考你是否分清 “原子性” 和 “内存可见性顺序”。

**第一性原理**

多线程通信不是只要 flag 原子就够，还要保证 flag 之前写入的数据在另一个线程看到 flag 后可见。

**回答**

典型模式：

```text
producer:
    write data
    release store flag

consumer:
    acquire load flag
    read data
```

release 表示：

```text
release 之前的写不能被重排到 release 之后
```

acquire 表示：

```text
acquire 之后的读不能被重排到 acquire 之前
```

配对后形成 happens-before：

```text
producer 写 data
    happens-before
consumer 读 data
```

CUDA 里还要说 scope：

```text
block scope：block 内
device scope：一个 GPU 内
system scope：CPU / GPU / 多设备系统范围
```

**发散思考**

可以继续问：

```text
release/acquire 和 seq_cst 区别？
fence 和 atomic memory_order 区别？
为什么 flag 必须是同一个同步对象？
```

一句话：

> release-acquire 是 producer-consumer 的可见性协议：release 发布数据，acquire 接收数据。

---

## A13. 做 GEMM 该怎么切分任务？

**考察目的**

考你能否把 GEMM 从数学公式映射到 GPU 层级并行。不是只问 block size，而是问 CTA、warp、MMA、pipeline、memory layout 全链路。

**第一性原理**

GEMM：

```text
C[M,N] = A[M,K] @ B[K,N]
```

每个 C 元素都沿 K reduction。为了复用 A/B，必须 tile：

```text
CTA tile：BM × BN
K tile：BK
warp / warpgroup tile：WM × WN
MMA tile：m16n8k16 / WGMMA / tcgen05 tile
```

**回答**

切分步骤：

```text
1. 选 CTA tile：
   让 A/B tile 在 shared memory 中有足够复用。

2. 选 K tile：
   控制 shared memory 占用和 compute per stage。

3. 选 warp / warpgroup tile：
   匹配 Tensor Core 指令形状。

4. 选 MMA tile：
   对应 PTX 指令粒度。

5. 设计 multistage pipeline：
   当前 tile 计算时预取下一 tile。

6. 设计 shared memory layout：
   避免 bank conflict，适配 ldmatrix / wgmma。

7. 设计 epilogue：
   accumulator → scale/bias/activation → coalesced store。

8. 调度层优化：
   threadblock swizzle、split-K、stream-K、persistent。
```

CUTLASS GEMM 文档把 GEMM 明确拆成 threadblock-level、warp-level、instruction-level 层级，warp-level GEMM 从 shared memory load tile 到 register，再用 Tensor Core 或 CUDA core 计算。([NVIDIA Docs][7])

**发散思考**

继续追问：

```text
为什么 BM/BN 变大 arithmetic intensity 更高？
为什么 tile 太大反而慢？
split-K 什么时候有用？
stream-K 和 persistent GEMM 区别？
```

强回答：

> GEMM 切分要同时满足数据复用、Tensor Core 指令粒度、shared memory 容量、register pressure、occupancy 和 L2 locality。

---

## A14. 什么是 roofline 模型？缺点是什么？怎么解决？

**考察目的**

考你是否有性能建模能力，而不是只会 profile。Roofline 是判断 memory-bound / compute-bound 的第一性原理工具。

**第一性原理**

性能上限：

```text
achievable performance = min(peak compute, arithmetic intensity × memory bandwidth)
```

其中：

```text
arithmetic intensity = FLOPs / bytes
```

**回答**

如果 arithmetic intensity 低，每搬 1 byte 做不了多少计算，kernel 受内存带宽限制。
如果 arithmetic intensity 高，数据复用充分，可能受计算峰值限制。

NVIDIA Nsight Compute roofline 资料也用 arithmetic intensity 和 FLOP performance 判断 kernel 距离 memory roof 或 compute roof 的位置。([NVIDIA Developer][11])

缺点：

```text
1. 不看 latency
2. 不区分 HBM / L2 / L1 / shared
3. 不区分 Tensor Core / CUDA core / SFU / LDST pipe
4. 不看 occupancy、register pressure、barrier stall
5. 不看通信
6. 对 attention、MoE、decode 这类混合算子解释力有限
```

解决：

```text
hierarchical roofline：分 HBM/L2/shared 看
instruction roofline：分 Tensor Core/CUDA core/SFU 看
communication roofline：多 GPU 加 NVLink/IB roof
profile-driven：结合 Nsight stall reason、L2 hit、dram throughput、SM busy
```

**发散思考**

可以追问：

```text
roofline 判断 memory-bound，但 dram 带宽没打满，为什么？
AI 很高但 Tensor Core 利用率低，可能是什么问题？
decode attention 的 roofline 怎么建？
```

一句话：

> Roofline 给方向，不给完整答案；它告诉你理论瓶颈在算力还是带宽，但不能解释所有同步、调度、延迟和指令混合问题。

---

## A15. multistage 怎么确定 stage 深度？

**考察目的**

考你是否理解 pipeline depth 的本质，不是背 “3 stage/4 stage”。

**第一性原理**

stage depth 是为了隐藏 copy latency：

```text
stage_depth ≈ ceil(copy_latency / compute_time_per_stage) + 1
```

但 stage 越深，shared memory 和状态开销越大。

**回答**

判断逻辑：

```text
如果每个 K tile compute 很长：
    2-stage 可能够。

如果 K tile 小，copy latency 明显：
    需要 3/4/5-stage。

如果 shared memory 占用导致 occupancy 大幅下降：
    stage 不能继续加。

如果 barrier stall 很高：
    pipeline 同步设计可能有问题。
```

资源约束：

```text
shared memory ≈ stages × (A_tile + B_tile)
register ≈ accumulator + iterator + pipeline state
occupancy ≈ 受 shared/register/CTA 数限制
```

**发散思考**

追问方向：

```text
为什么 stage 多了可能更慢？
copy latency 被谁隐藏，warp scheduler 还是 pipeline？
TMA 和 cp.async 的 stage depth 选择是否一样？
```

一句话：

> stage depth 是 latency hiding 和资源占用的折中；最优值由 copy latency、每 stage compute cycles、shared memory、register pressure 和 occupancy 共同决定。

---

## A16. Tensor Core 的使用，PTX 级别

**考察目的**

考你是否知道 Tensor Core 是固定 tile shape 的矩阵乘指令，不是自动魔法。

**第一性原理**

Tensor Core 计算的是：

```text
D = A × B + C
```

但 A/B/C/D 都是由 warp 或 warpgroup 按特定 fragment layout 协作提供。

**回答**

Ampere 常见：

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

表示一个 warp 协作完成 `m16n8k16` tile 的 MMA，输入 FP16，累加 FP32。

Hopper 使用：

```text
wgmma.mma_async
```

粒度从 warp 变成 warpgroup，通常 128 threads 协作，异步执行，需要 commit/wait/barrier 配合。PTX ISA 文档列出了 `wgmma.mma_async`、mbarrier、`tcgen05` 等指令族。([NVIDIA Docs][12])

Blackwell 使用：

```text
tcgen05.mma
```

核心变化是引入 TMEM 和新 MMA 指令族，减少普通 register accumulator 压力。CUTLASS Blackwell 文档也把 targeting `tcgen05.mma` 作为 Blackwell SM100 GEMM 的关键。([NVIDIA Docs][3])

**发散思考**

可以追问：

```text
mma.sync 和 wgmma.mma_async 的同步语义差异？
ldmatrix 为什么重要？
为什么 Tensor Core 利用率高但整体 kernel 仍慢？
```

一句话：

> Tensor Core kernel 的难点不是写 MMA 指令，而是让 global/shared/register/TMEM 的 layout 和 pipeline 正好匹配 MMA 消费速度。

---

## A17. mbarrier 的原理

**考察目的**

考你是否理解 Hopper 异步流水里的同步机制。普通 barrier 同步线程，mbarrier 同步异步事务和 phase。

**第一性原理**

异步 copy 的完成时间不是 producer 线程发出指令的时间。因此 consumer 不能只等 producer “走到某行代码”，还要等数据事务真的完成。

**回答**

普通 barrier：

```text
所有线程 arrive
→ 所有线程继续
```

mbarrier：

```text
producer arrive / expect transaction
TMA 发起异步 copy
copy engine 完成 transaction
consumer wait 当前 phase
```

它维护：

```text
arrival count
transaction count
phase
pending bytes
```

典型流程：

```text
producer:
    mbarrier.expect_tx(bytes)
    issue TMA copy

TMA engine:
    copy done → complete_tx

consumer:
    wait barrier phase
    read shared buffer
```

**发散思考**

追问方向：

```text
mbarrier 和 __syncthreads 区别？
为什么 TMA 必须配 mbarrier？
phase 是什么？
mbarrier 如何支持 double-buffer / multi-stage？
```

一句话：

> mbarrier 同步的是异步数据事务是否完成，而不只是线程是否到达某个代码点。

---

## A18. 通信相关：Ethernet、RDMA、IB、IMEX、NVLink 的层级

**考察目的**

考你是否能把多 GPU 通信按物理距离和语义层次讲清楚。

**第一性原理**

通信距离越远，带宽越低、延迟越高、协议越复杂：

```text
GPU 内：
    SM / L2 / HBM

节点内或 rack scale-up：
    NVLink / NVSwitch

节点间 scale-out：
    InfiniBand / RoCE RDMA

普通网络：
    Ethernet / TCP/IP

多节点 NVLink 内存导入导出：
    IMEX
```

**回答**

概念关系：

```text
Ethernet：
    通用网络基础，可以跑 TCP/IP，也可以跑 RoCE。

RDMA：
    远端内存直接访问语义，不是某根线。

InfiniBand：
    面向高性能计算的低延迟网络，天然支持 RDMA。

RoCE：
    RDMA over Converged Ethernet。

NVLink：
    GPU-GPU 高带宽低延迟互联。

NVSwitch：
    把多条 NVLink 组成交换网络。

IMEX：
    NVIDIA 多节点 NVLink 系统中的 GPU memory export/import 和映射服务。
```

NVIDIA RDMA 文档把 InfiniBand、Ethernet RoCE、Ethernet iWARP 都列为 RDMA 技术路线；NVIDIA 多节点 NVLink 文档说明 IMEX 用于 NVLink 网络中的 GPU memory mapping。([NVIDIA Developer Forums][13])

**发散思考**

可以追问：

```text
RDMA 和 NCCL 是什么关系？
NVLink 和 PCIe 区别？
IB 和 RoCE 什么时候选哪个？
为什么 scale-up 和 scale-out 策略不同？
```

一句话：

> RDMA 是远端内存访问语义，IB/RoCE 是承载方式，NVLink 是 GPU scale-up 互联，IMEX 是多节点 NVLink memory mapping 的软件/驱动机制。

---

## A19. cp.async 和 TMA 的本质区别是什么？

**考察目的**

这是 A1 的自然深挖。面试官想看你是否知道 Hopper 为什么还需要 TMA，而不是 Ampere `cp.async` 已经够了。

**第一性原理**

copy 的成本不只是带宽，还有：

```text
地址生成
线程参与
同步
描述多维 tensor layout
pipeline 状态维护
```

**回答**

`cp.async` 更像 thread/warp 级别的异步拷贝。它把 global memory 数据异步搬到 shared memory，适合手工 tiled kernel，但需要很多线程参与发 copy、做地址计算、组织 transaction。

TMA 更像专门的 tensor copy engine。它可以根据 descriptor 搬多维 tensor tile，减少线程参与和地址生成开销，更适合 Hopper 上的大 tile GEMM/attention。NVIDIA Hopper 架构资料明确把 TMA 描述为新的 async memory copy unit，用于增强 memory copy 和 computation overlap。([NVIDIA Developer][2])

**发散思考**

可以继续追问：

```text
TMA descriptor 解决什么？
为什么 TMA 适合 producer warp specialization？
TMA multicast 什么时候有用？
TMA 是否完全替代 cp.async？
```

一句话：

> cp.async 是更细粒度的 global-to-shared 异步 copy；TMA 是面向大块 tensor tile 的专用 copy engine，减少线程参与和地址生成成本。

---

## A20. mma.sync、wgmma.mma_async、tcgen05.mma 分别是什么粒度？

**考察目的**

考你是否能把架构代际和 Tensor Core 编程模型对应起来。

**第一性原理**

Tensor Core 指令的演进方向是：

```text
更大 tile
更异步
更少 register 压力
更强 CTA/warpgroup 协作
```

**回答**

```text
mma.sync：
    Ampere 常见，warp-level，32 线程协作，结果进 register accumulator。

wgmma.mma_async：
    Hopper，warpgroup-level，通常 128 线程协作，异步发起 MMA。

tcgen05.mma：
    Blackwell SM100，新 Tensor Core 指令族，配合 TMEM，支持更复杂的 CTA group 形态。
```

PTX ISA 文档列出 `wgmma.mma_async` 和 Blackwell `tcgen05` 指令；CUTLASS Blackwell 文档也把 `tcgen05.mma` 作为 SM100 GEMM 的核心新能力。([NVIDIA Docs][12])

**发散思考**

追问：

```text
为什么 wgmma 是 async？
tcgen05 为什么需要 TMEM？
warp-level 到 warpgroup-level 对 register layout 有什么影响？
```

一句话：

> MMA 指令从 warp-level 到 warpgroup-level 再到 TMEM-backed，是为了适配更大 tile、更高 Tensor Core 吞吐和更低寄存器压力。

---

## A21. Hopper thread block cluster 和 distributed shared memory 解决什么问题？

**考察目的**

考你是否知道 Hopper 不是只有 HBM 和 Tensor Core 提升，还提供了跨 CTA 的局部协作能力。

**第一性原理**

普通 shared memory 只在一个 CTA 内可见。如果多个 CTA 需要协作复用片上数据，过去只能走 global/L2 或复杂同步，成本高。

**回答**

thread block cluster 允许一组 CTA 在调度上更紧密，并支持 cluster-level synchronization 和 distributed shared memory。这样 CTA 之间可以更高效地协作，例如共享中间数据、做跨 CTA reduction、或者配合 TMA multicast。

适合场景：

```text
large GEMM tile
multi-CTA attention tile
histogram / reduction
跨 CTA 数据复用明显的 kernel
```

**发散思考**

追问：

```text
cluster 和 grid 区别？
distributed shared memory 和普通 shared memory 区别？
cluster barrier 成本如何？
为什么不是所有 kernel 都适合 cluster？
```

一句话：

> cluster 把 CTA 协作从 global/L2 层部分拉回片上/近片上层级，降低跨 CTA 数据交换和同步成本。

---

## A22. TMA multicast 适合什么场景？

**考察目的**

考你是否理解数据复用不仅发生在 CTA 内，也发生在多个 CTA 之间。

**第一性原理**

如果多个 CTA 需要同一份 global memory tile，与其每个 CTA 各自从 HBM/L2 读一遍，不如一次 copy 后 multicast 到多个 shared memory 目标。

**回答**

TMA multicast 适合：

```text
GEMM 中同一个 A tile 被多个 N 方向 CTA 复用
GEMM 中同一个 B tile 被多个 M 方向 CTA 复用
cluster 内多个 CTA 共享一个 operand tile
attention 中某些 K/V block 被多个 Q block 使用
```

它的目标是减少重复 global memory traffic，并让 cluster 内 CTA 更高效协作。

**发散思考**

追问：

```text
TMA multicast 和 L2 reuse 区别？
什么时候 multicast 不划算？
multicast 对 cluster placement 有什么要求？
```

一句话：

> TMA multicast 是把跨 CTA 的数据复用显式化：同一份 tile 一次搬运，多 CTA 消费。

---

## A23. Blackwell TMEM 对 epilogue 有什么影响？

**考察目的**

考你是否理解 TMEM 不只是 “多了一个存储空间”，而是改变 accumulator 和 epilogue 的设计。

**第一性原理**

GEMM 主循环结束后，accumulator 需要经过：

```text
accumulator
  → scale / bias / activation / residual
  → store global
```

如果 accumulator 全在 register，epilogue 会受到 register pressure 和搬运路径限制。

**回答**

TMEM 把 Tensor Core accumulation 的部分状态放到专门空间，缓解 register pressure。但 epilogue 仍然需要把 TMEM 中结果取出，做 scale、bias、activation、quantization 或 store。

影响：

```text
1. accumulator 不完全占普通 register
2. 主循环可用更大 tile 或更高 occupancy
3. epilogue 需要新的 TMEM load / transform 路径
4. epilogue fusion 需要考虑 TMEM → register → global 的成本
```

CUTLASS Blackwell 文档围绕 Blackwell SM100 GEMM、`tcgen05.mma`、epilogue config 等展开，说明新指令和新 accumulator 路径会影响 kernel 设计。([NVIDIA Docs][3])

**发散思考**

追问：

```text
TMEM 是否让 epilogue 更快？
TMEM 和 register accumulator 如何配合？
quantized GEMM epilogue 在 Blackwell 上有什么新机会？
```

一句话：

> TMEM 缓解主循环 register accumulator 压力，但 epilogue 设计要重新考虑从 TMEM 取数、融合和写回的路径。

---

## A24. 为什么 register pressure 会影响 Tensor Core kernel 吞吐？

**考察目的**

考你是否知道 Tensor Core kernel 的瓶颈不只是 Tensor Core 本身，还有 register file、occupancy 和调度。

**第一性原理**

register 用得越多：

```text
每个 thread 占用 register 越多
→ 一个 SM 能驻留的 warp/CTA 越少
→ latency hiding 能力下降
```

同时 register file 带宽也可能成为瓶颈。

**回答**

Tensor Core kernel 中 register 主要用于：

```text
accumulator fragment
A/B fragment
iterator state
地址计算
pipeline metadata
epilogue 临时值
```

如果 accumulator tile 很大，register 数会暴涨。后果：

```text
1. occupancy 降低
2. spill 到 local memory
3. epilogue 无法融合更多操作
4. warp scheduler 可调度 warp 变少
5. copy/compute overlap 变差
```

**发散思考**

追问：

```text
降低 register pressure 的方法？
tile 变小会有什么代价？
TMEM 如何缓解这个问题？
为什么 occupancy 高不一定好，但过低一定危险？
```

一句话：

> Tensor Core 需要持续喂数，register pressure 过高会减少可调度 warp 和增加 spill，使 Tensor Core 出现气泡。

---

## A25. shared memory swizzle 和 threadblock swizzle 区别？

**考察目的**

考你是否分清两个 “swizzle” 解决的问题完全不同。

**第一性原理**

```text
shared memory swizzle：
    改变 CTA 内数据在 shared memory 中的物理布局。

threadblock swizzle：
    改变 CTA 在 grid 上的执行/映射顺序。
```

**回答**

shared memory swizzle 解决：

```text
bank conflict
ldmatrix / wgmma 访问模式适配
shared memory 片上带宽
```

threadblock swizzle 解决：

```text
L2 reuse
memory partition locality
CTA 调度顺序
跨 CTA 数据复用
```

例如 GEMM 中，shared swizzle 让 `A/B` tile 在 shared memory 中被 warp 读取时不冲突；threadblock swizzle 让相邻 CTA 复用同一个 A 或 B tile 的 L2 cache。

**发散思考**

追问：

```text
二者能否同时使用？
shared swizzle 是否会影响 coalesced store？
threadblock swizzle 是否一定提高 L2 hit？
```

一句话：

> shared swizzle 是片上布局优化，threadblock swizzle 是跨 CTA 调度和 L2 locality 优化。

---

## A26. ldmatrix 为什么需要特殊 shared memory layout？

**考察目的**

考你是否知道 Tensor Core fragment 的读取不是普通连续读，而是特定 warp 协作模式。

**第一性原理**

Tensor Core MMA 的输入 fragment 由 warp 多个线程共同持有。`ldmatrix` 需要把 shared memory 中的矩阵 tile 以特定方式加载到各线程寄存器中。

**回答**

如果 shared memory 中 layout 不合适，会出现：

```text
bank conflict
fragment 排列不匹配 MMA 指令
额外 shuffle / transpose
shared memory bandwidth 下降
```

因此高性能 GEMM 会用特殊 layout，例如 swizzled layout，让 warp 执行 `ldmatrix` 时各线程访问分散到不同 bank，同时直接得到 MMA 需要的 fragment。

**发散思考**

追问：

```text
ldmatrix.x1/x2/x4 区别？
row-major 输入是否可以直接 ldmatrix？
为什么 CUTLASS layout 很复杂？
```

一句话：

> ldmatrix 的 layout 目标不是人类阅读方便，而是让 warp 以无 bank conflict 的方式直接取到 Tensor Core fragment。

---

## A27. split-K、stream-K、persistent GEMM 分别适合什么 shape？

**考察目的**

考你是否理解 GEMM 调度不是固定 2D tile，特殊 shape 需要特殊并行策略。

**第一性原理**

普通 GEMM 并行度来自 M、N 方向 CTA 数量。如果 M/N 小、K 大，普通 2D tiling 可能 CTA 数不足，SM 吃不满。

**回答**

split-K：

```text
沿 K 维切分，同一个 C tile 由多个 CTA 计算 partial sum，最后 reduction。
适合 K 很大、M/N 并行度不足。
代价是额外 reduction。
```

stream-K：

```text
把 K 维 work 以更细粒度流式分配给 CTA，提高负载均衡。
适合不规则或 M/N tile 不足场景。
```

persistent GEMM：

```text
CTA 常驻 SM，从 work queue 取多个 tile。
适合 tile 数不足、小 batch、shape 不规则。
```

**发散思考**

追问：

```text
split-K reduction 怎么做？
stream-K 和 persistent 是否冲突？
为什么小 M 大 K 的 GEMM 很难？
```

一句话：

> split-K 增加 K 方向并行度，stream-K 改善负载均衡，persistent GEMM 改善调度和小问题规模利用率。

---

## A28. 小 batch GEMM 为什么难优化？

**考察目的**

考你是否理解大模型 decode 中很多 GEMM 不是训练时的大矩阵乘。

**第一性原理**

GEMM 性能依赖 tile 数量和 arithmetic intensity。batch 小时，M 维很小，CTA 数和数据复用都不足。

**回答**

小 batch GEMM 的问题：

```text
1. M 很小，grid tile 数不足
2. Tensor Core tile 填不满
3. launch overhead 相对变大
4. epilogue / dequant / activation 占比变高
5. memory-bound 倾向更强
6. threadblock swizzle 可复用空间变小
```

解决方向：

```text
persistent kernel
grouped GEMM
fused GEMM
weight prepacking
quantization
batching 多 request
CUDA graph 减少 launch overhead
```

**发散思考**

追问：

```text
decode GEMM 为什么常是 GEMV-like？
grouped GEMM 如何调度？
小 batch 下 Tensor Core 是否总是合适？
```

一句话：

> 小 batch GEMM 难在并行度不足和固定开销占比高，不是单纯优化 tile 就能解决。

---

## A29. 为什么 decode attention 通常 memory-bound，而 prefill attention 更接近 compute-bound？

**考察目的**

考你是否理解 LLM 推理两个阶段的计算形态。

**第一性原理**

prefill 一次处理 prompt 中很多 token；decode 每步只生成一个 token，却要读取所有历史 KV。

**回答**

prefill：

```text
Q,K,V 都是长序列
attention 是大块 QK^T 和 PV
矩阵乘规模大
Tensor Core 利用率较好
更接近 compute-bound
```

decode：

```text
每步 query token 很少
需要扫描历史 KV cache
主要成本是读 KV
batch 小时 GEMV-like
更 memory-bound
```

所以 decode 优化重点常是：

```text
KV cache layout
PagedAttention
KV quantization
GQA/MQA/MLA
continuous batching
FlashDecoding
```

**发散思考**

追问：

```text
decode batch 变大后是否仍 memory-bound？
MLA 为什么可能把 decode attention 推向 compute-bound？
KV4 量化对 decode 有什么收益？
```

一句话：

> prefill 是大块矩阵计算，decode 是小 query 反复扫 KV cache；前者更像 GEMM，后者更像 memory streaming。

---

## A30. 如何用 Nsight 判断 kernel 是 memory-bound、barrier-bound 还是 instruction-bound？

**考察目的**

考你是否能从 profiler counter 反推瓶颈，而不是凭感觉优化。

**第一性原理**

性能瓶颈看三类信号：

```text
资源利用率
stall reason
数据流量
```

**回答**

memory-bound 常见表现：

```text
dram throughput 高
L2 hit 低或 dram sectors 多
stall_memory_dependency 高
SM busy 不高
```

barrier-bound：

```text
stall_barrier 高
warp 等待同步
pipeline stage 或 mbarrier 设计不合理
```

instruction-bound：

```text
Tensor Core 利用率不高
但 memory 不满
scheduler/issue 受限
non-matmul 指令占比高
SFU/LDST pipe 成为瓶颈
```

Tensor Core 饥饿：

```text
sm__pipe_tensor_active 低
但 copy/LDST/barrier stall 高
```

**发散思考**

追问：

```text
L2 hit 高为什么仍 memory-bound？
SM busy 高但性能差说明什么？
roofline 和 stall reason 冲突时信谁？
```

一句话：

> Nsight 要结合 roofline、throughput、stall reason、pipe utilization 看；单一 counter 很容易误判。

---

## A31. L2 hit rate 高但性能差，可能是什么原因？

**考察目的**

考你是否知道 cache hit 不等于快，性能还受带宽、同步、指令、occupancy 影响。

**第一性原理**

L2 hit rate 只说明数据命中 L2，不说明：

```text
L2 带宽是否够
L1/shared 是否高效
consumer 是否及时消费
Tensor Core 是否被喂饱
```

**回答**

可能原因：

```text
1. L2 hit 高但 L2 带宽成为瓶颈
2. shared memory bank conflict
3. register pressure 导致 occupancy 低
4. barrier stall 高
5. non-matmul 指令占比高
6. memory access pattern 命中但不 coalesced
7. L2 hit 的是无用数据
8. epilogue/store 成为瓶颈
```

**发散思考**

追问：

```text
怎么看 L2 bandwidth？
L2 hit rate 和 sectors/request 有什么关系？
L2 hit 高是否说明 threadblock swizzle 有效？
```

一句话：

> L2 hit rate 只是 locality 指标，不是性能指标；真正要看 L2 带宽、SM pipe、stall 和有效字节利用率。

---

## A32. 为什么 occupancy 高不等于性能高？

**考察目的**

考你是否理解 occupancy 只是可调度 warp 数，不等于有效计算吞吐。

**第一性原理**

occupancy 的作用是隐藏 latency。但如果 kernel 已经被 Tensor Core、memory bandwidth 或 synchronization 限制，更多 warp 不一定提高性能。

**回答**

occupancy 高但性能差的可能原因：

```text
1. 每个 warp 做的有效计算少
2. memory access 不 coalesced
3. shared memory bank conflict
4. Tensor Core 没吃满
5. barrier 频繁
6. arithmetic intensity 低
7. register 少导致重复访存
```

occupancy 低但性能好的情况：

```text
large tile GEMM
register blocking 充分
Tensor Core 利用率高
memory latency 被 pipeline 隐藏
```

**发散思考**

追问：

```text
什么时候要牺牲 occupancy 换 tile size？
occupancy 和 register pressure 如何 tradeoff？
Nsight 的 theoretical occupancy 和 achieved occupancy 区别？
```

一句话：

> occupancy 是 latency hiding 的手段，不是目标；目标是有效吞吐。

---

## A33. 为什么 Tensor Core 利用率高但端到端不一定快？

**考察目的**

考你是否区分 kernel 内主循环性能和端到端系统性能。

**第一性原理**

端到端时间包括：

```text
data movement
layout transform
dequant
epilogue
normalization
attention
sampling
communication
launch overhead
scheduler overhead
```

**回答**

Tensor Core 利用率高说明 GEMM 主循环不错，但可能：

```text
1. epilogue 很慢
2. dequant / scale 在 CUDA core 上
3. layout transform 很多
4. attention / norm / sampling 拖后腿
5. batch 太小，launch overhead 高
6. 多 GPU communication 成瓶颈
7. serving scheduler 排队严重
```

**发散思考**

追问：

```text
怎么做 end-to-end profiling？
为什么低比特 GEMM 快但系统不快？
fusion 能解决什么？
```

一句话：

> Tensor Core 利用率是局部指标；推理系统要看端到端 critical path。

---

## A34. global memory coalescing 和 shared memory bank conflict 有什么本质区别？

**考察目的**

考你是否能区分片外访存和片上访存的优化机制。

**第一性原理**

global memory 优化目标是减少 transaction/sector；shared memory 优化目标是让 bank 并行服务。

**回答**

global coalescing：

```text
对象：global memory
粒度：warp 访问合并为 sector/transaction
目标：减少 HBM/L2 请求数量和无效字节
```

shared bank conflict：

```text
对象：shared memory
粒度：bank 并行
目标：避免同一 bank 不同地址串行化
```

一个访问可以 global coalesced，但写入 shared layout 后产生 bank conflict；也可以 shared 无冲突，但 global load 不连续。

**发散思考**

追问：

```text
transpose kernel 如何同时优化 global coalescing 和 shared conflict？
padding 为什么解决 shared conflict 但不一定解决 global coalescing？
```

一句话：

> coalescing 解决片外 transaction，bank conflict 解决片上 bank 并行。

---

## A35. CUDA memory scope：block/device/system 分别什么时候用？

**考察目的**

考你是否知道同步不是越大越好，scope 影响成本和可见范围。

**第一性原理**

同步范围越大，语义越强，成本越高。

**回答**

```text
block scope：
    CTA 内线程通信，例如 shared memory producer-consumer。

device scope：
    同一 GPU 内 CTA 之间通信。

system scope：
    CPU-GPU 或多 GPU 之间通信。
```

选择原则：

```text
只用覆盖参与通信线程的最小 scope。
```

例如 CTA 内不用 system scope；CPU 等 GPU flag 时需要 system scope。

**发散思考**

追问：

```text
device scope atomic 能否同步 CPU？
block scope atomic 能否跨 CTA？
scope 和 memory_order 是什么关系？
```

一句话：

> memory scope 决定同步可见范围，memory order 决定顺序语义。

---

## A36. `__syncthreads()`、`__syncwarp()`、mbarrier、cluster barrier 区别？

**考察目的**

考你是否能按同步对象和范围分类。

**第一性原理**

不同同步原语回答的问题不同：

```text
谁需要同步？
同步线程到达，还是异步事务完成？
同步范围是 warp、CTA、cluster？
```

**回答**

```text
__syncwarp():
    warp 内线程同步。

__syncthreads():
    CTA 内所有线程同步，并保证 shared memory 可见。

mbarrier:
    异步 pipeline barrier，可同步 TMA/cp.async 等事务完成。

cluster barrier:
    thread block cluster 内 CTA 之间同步。
```

**发散思考**

追问：

```text
什么时候 __syncwarp 足够？
为什么 mbarrier 不能简单替代 __syncthreads？
cluster barrier 成本为什么更高？
```

一句话：

> 同步原语要按范围和对象选择：warp、CTA、cluster，线程到达还是异步事务完成。

---

## A37. GPU atomic 的性能瓶颈在哪里？

**考察目的**

考你是否知道 atomic 不是只是“原子加一下”，它会引入竞争和序列化。

**第一性原理**

atomic 的成本来自：

```text
同一地址竞争
cache line/sector 竞争
L2 serialization
memory ordering
跨 scope 可见性
```

**回答**

瓶颈场景：

```text
1. 很多线程 atomic 到同一地址
2. global atomic 跨 SM 竞争
3. system-scope atomic 跨 CPU/GPU
4. atomic 后马上读结果导致 dependency
5. hash table / histogram 热点严重
```

优化：

```text
warp-level aggregation
block-level reduction 后再 atomic
分桶减少热点
使用 shared memory atomic
改变数据结构
```

**发散思考**

追问：

```text
atomicAdd FP32 是否 deterministic？
atomic 和 reduction 哪个更好？
什么时候 atomic 性能可以接受？
```

一句话：

> atomic 的核心问题是竞争导致序列化；减少热点比换一个 atomic 指令更重要。

---

## A38. FP8/FP4 kernel 为什么常难在 scale 管理，而不是 MMA 本身？

**考察目的**

考你是否理解低精度不是只换数据类型，scale 和 layout 是核心。

**第一性原理**

低精度表示范围小，必须通过 scale 把真实值映射到有限动态范围。scale 本身需要存、读、广播、参与计算。

**回答**

难点：

```text
1. scale 粒度选择：per-tensor / per-channel / per-block
2. scale 读取增加带宽
3. scale 乘法增加非 Tensor Core 操作
4. packing/unpacking 复杂
5. overflow/underflow 风险
6. epilogue 需要 rescale / requant
7. attention 中 softmax 对数值范围敏感
```

FP4/NVFP4 更依赖 block scale；如果 scale 粒度太粗，精度差；太细，metadata 和 dequant overhead 高。

**发散思考**

追问：

```text
FP8 和 INT8 的 scale 差异？
block scaling 对 Tensor Core layout 有什么影响？
scale 放 global、shared 还是 register？
```

一句话：

> 低精度 MMA 很快，但 scale 管理决定端到端是否真的快且准。

---

# B. Attention / LLM Kernel / 推理系统 / 量化

## B1. 什么是 Ring Attention？不用 Ring Attention 怎么做 SP？

**考察目的**

考你是否理解 sequence parallel 的核心矛盾：每个 Q token 需要看到全局 K/V，但 K/V 被分布到多卡。

**第一性原理**

attention 的依赖是：

```text
每个 Q block 需要 attend 所有 K/V block
```

如果 sequence 被切到 P 张卡，每张卡只有一段 K/V，就必须通过通信让 Q 看到全局 K/V。

**回答**

AllGather KV：

```text
每张卡 all-gather 全部 K/V
本地计算自己的 Q 对全量 K/V
```

优点简单，缺点显存峰值高、通信集中、overlap 差。

Ulysses：

```text
沿 sequence 切分
通过 all-to-all 重排到 head 维
每张卡处理部分 head 的完整 sequence
再 all-to-all 回去
```

DeepSpeed-Ulysses 论文说明其核心是沿 sequence 维 partition，并用 all-to-all collective 进行 attention computation。([arXiv][14])

Ring Attention：

```text
每张卡固定持有自己的 Q block
K/V block 沿 ring 传递
每一步边接收 KV 边做 blockwise attention
用 online softmax 合并 partial result
```

Ring Attention 论文强调通过 blockwise computation 分布长序列，并 fully overlap K/V block communication with blockwise attention computation。([arXiv][15])

**发散思考**

追问：

```text
Ring Attention 通信量是否减少？
为什么它能 overlap？
Ulysses 和 Ring 如何组合？
Ring Attention 如何做 online softmax merge？
```

一句话：

> AllGather 是先拿全再算；Ring 是边传边算，用 attention compute 覆盖 K/V 通信。

---

## B2. SmoothQuant 是什么？怎么做？

**考察目的**

考你是否理解 activation outlier 是 W8A8 量化的核心难点。

**第一性原理**

线性层：

```text
Y = XW
```

如果 activation X 某些 channel 有 outlier，INT8 activation 的 scale 会被 outlier 拉大，导致大部分普通值精度很差。

**回答**

SmoothQuant 利用等价变换：

```text
Y = XW
  = (X / s) · (diag(s) W)
```

把 activation 的 outlier “迁移”到 weight 中。activation 变平滑后更容易 INT8 量化；weight 虽然 range 变大，但 weight 是离线固定的，更容易校准。SmoothQuant 论文明确说它把量化难度从 activation 离线迁移到 weight，从而实现 W8A8 PTQ。([arXiv][16])

常见 scale：

```text
s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
```

**发散思考**

追问：

```text
alpha 越大意味着什么？
SmoothQuant 和 AWQ 区别？
为什么 activation 比 weight 更难量化？
```

一句话：

> SmoothQuant 的本质是利用线性层 channel scaling invariance，把 activation outlier 平滑掉。

---

## B3. AWQ 是什么？

**考察目的**

考你是否理解 weight-only 低比特量化不是所有权重同等重要。

**第一性原理**

输出误差来自：

```text
quant_error(W) × activation
```

activation 大的 channel，对输出更敏感。因此重要性不只由 weight 自身决定，而由 activation-aware 的误差贡献决定。

**回答**

AWQ 的核心：

```text
1. 用 calibration data 统计 activation
2. 找出 salient weight channels
3. 对重要 channel 做 scaling 保护
4. 再做低比特 weight-only quantization
```

AWQ 论文指出，不是所有权重都同等重要，保护约 1% salient weights 就能显著降低量化误差，并且 salient channel 应由 activation distribution 识别。([arXiv][17])

**发散思考**

追问：

```text
AWQ 和 GPTQ 区别？
为什么 AWQ 不用 backprop？
scale up salient channel 为什么能降低量化误差？
```

一句话：

> AWQ 是 activation-aware 的 weight quantization：用 activation 找哪些 weight channel 更值得保护。

---

## B4. 什么是渐进式量化？QoQ 在性能上有什么问题？

**考察目的**

考你是否理解低比特量化的系统开销。低 bit 不等于自动快。

**第一性原理**

量化减少了数据字节，但引入了：

```text
scale
zero point
dequant
packing/unpacking
layout transform
requant
```

**回答**

渐进式量化是逐步压缩：

```text
FP16/BF16
→ W8A8
→ W4A8
→ W4A8KV4
```

每一步重新校准 scale 和误差，而不是一步压到极低比特。

QoQ 来自 QServe，表示 W4A8KV4：4-bit weight、8-bit activation、4-bit KV cache。QServe 论文指出已有 INT4 方法在 GPU 上 dequant weights 或 partial sums 会产生 20–90% runtime overhead，因此需要算法和系统协同设计。([arXiv][18])

性能问题：

```text
1. dequant 可能落在 CUDA core
2. scale 读取增加带宽
3. INT4 packing/unpacking 有成本
4. 小 batch decode kernel overhead 明显
5. KV4 精度和 scale 管理复杂
```

**发散思考**

追问：

```text
为什么 W4A8 比 W4A16 更难？
为什么低比特 GEMM 快但 serving 不一定快？
KV cache quantization 为什么敏感？
```

一句话：

> QoQ 的问题意识是：低 bit 节省 memory，但如果 dequant、scale、layout transform 吃掉收益，端到端不会快。

---

## B5. 手推 MLA，考察矩阵吸收

**考察目的**

考你是否理解 MLA 为什么能省 KV cache，以及矩阵吸收怎么避免恢复完整 K/V。

**第一性原理**

decode 每生成一个 token，都要反复读历史 KV cache。KV cache bytes/token 是 decode 阶段核心瓶颈。

**回答**

标准 attention：

```text
Q = X W_Q
K = X W_K
V = X W_V

O = softmax(QK^T) V W_O
```

MLA 压缩 KV：

```text
C = X W_DKV
K = C W_UK
V = C W_UV
```

推理缓存 `C`，不缓存完整 K/V。

K 吸收：

```text
QK^T
= Q (C W_UK)^T
= Q W_UK^T C^T
```

令：

```text
Q' = Q W_UK^T
```

则：

```text
score = Q' C^T
```

V 吸收：

```text
P V W_O
= P (C W_UV) W_O
= P C (W_UV W_O)
```

令：

```text
W'_O = W_UV W_O
```

则：

```text
O = (P C) W'_O
```

DeepSeek-V2 论文说明 MLA 是其高效推理架构之一，通过压缩 KV 表示降低 KV cache 开销。([arXiv][19])

**发散思考**

追问：

```text
RoPE 为什么不能无脑吸收？
MLA 是省 memory 还是省 compute？
MLA kernel 是否可能变 compute-bound？
```

一句话：

> MLA 用低维 latent cache 替代完整 K/V cache，并通过矩阵吸收避免 decode 时显式恢复 K/V。

---

## B6. 标准 attention 计算 KV 迭代方向对精度是否有影响？

**考察目的**

考你是否理解浮点计算的非结合性和 online softmax 的数值稳定性。

**第一性原理**

数学上：

```text
sum(a_i) 与遍历顺序无关
```

浮点上：

```text
(a + b) + c != a + (b + c)
```

**回答**

online softmax 合并：

```text
m'   = max(m, max(S))
l'   = exp(m - m') l + sum(exp(S - m'))
acc' = exp(m - m') acc + exp(S - m') V
```

左到右扫 KV 和右到左扫 KV，实数结果相同，但浮点舍入路径不同，因此 bitwise 结果可能不同。

影响来源：

```text
block 顺序
m 更新路径
exp/rescale 舍入
accumulator 精度
reduction tree
```

**发散思考**

追问：

```text
如何保证 deterministic attention？
FP32 accumulator 能否完全解决？
pairwise reduction 有什么帮助？
```

一句话：

> 理论无影响，有限精度有影响；影响大小取决于累加精度、block size、reduction order 和数值范围。

---

## B7. 10000 个数字做 reduce，怎么尽量避免精度损失？

**考察目的**

考你是否理解数值误差传播，而不是只会说 “用 FP32”。

**第一性原理**

浮点加法会舍入。小数加到大数上可能被吞掉。线性累加误差随 n 增长，而树形累加误差更低。

**回答**

从简单到强：

```text
1. FP16/BF16 输入，用 FP32 accumulator
2. pairwise/tree reduction
3. 按绝对值从小到大加
4. Kahan / Neumaier compensated summation
5. 分块 compensated + tree merge
6. 固定 reduction order，避免 nondeterministic atomic
```

Kahan：

```python
s = 0.0
c = 0.0

for x in xs:
    y = x - c
    t = s + y
    c = (t - s) - y
    s = t
```

**发散思考**

追问：

```text
为什么 tree reduction 比 linear reduction 稳？
GPU 上 Kahan 成本高吗？
atomicAdd 为什么不可复现？
```

一句话：

> 精度优化顺序是：高精度 accumulator、稳定 reduction tree、compensated summation、固定顺序。

---

## B8. 什么是 EP？EP 中 AllGather 和 A2A 的区别？

**考察目的**

考你是否理解 MoE 的通信模式。EP 不是简单把 expert 放不同 GPU，而是 token dispatch 问题。

**第一性原理**

MoE 中每个 token 只去 top-k expert。如果 expert 分布在不同 GPU，token 必须被发送到 expert 所在 rank。

**回答**

流程：

```text
hidden states
  → router
  → token to expert id
  → dispatch
  → expert compute
  → combine
  → restore token order
```

AllGather dispatcher：

```text
每个 rank 把 token 发给所有 rank
每个 rank 从全量 token 中挑自己 expert 需要的
```

优点简单，缺点通信和显存浪费。

All-to-All dispatcher：

```text
token 直接发往目标 expert 所在 rank
```

优点通信更精确，EP 扩展性更好；缺点是 token permutation、capacity、load balance、metadata 更复杂。

Megatron-Core token dispatcher 文档也把 all-to-all 作为跨 expert parallel ranks dispatch tokens 的通信机制。([USENIX][20])

**发散思考**

追问：

```text
MoE load imbalance 怎么处理？
capacity factor 是什么？
token dropping 有什么风险？
AllGather 什么时候反而更好？
```

一句话：

> AllGather 是大家拿到所有 token 再筛，A2A 是 token 直接去 expert 所在 GPU。

---

## B9. PD 分离是什么？解决了什么问题？

**考察目的**

考你是否理解 prefill 和 decode 是两种不同 workload，不应该被同一套资源策略硬绑。

**第一性原理**

prefill 和 decode 的瓶颈不同：

```text
prefill：
    prompt 多 token，大 GEMM，更 compute-bound，关注 TTFT

decode：
    每步一个 token，反复读 KV，更 memory/latency-bound，关注 TPOT
```

**回答**

PD 分离就是把 prefill worker 和 decode worker 分开：

```text
prefill worker：
    处理 prompt
    生成 KV cache

KV transfer：
    传 KV block 或 cache handle

decode worker：
    continuous batching
    逐 token 生成
```

DistServe 论文指出，prefill 和 decode colocate 会造成 prefill-decoding interference，并耦合两阶段的资源分配和并行策略；DistServe 将 prefill 和 decoding 分配到不同 GPU，并按 TTFT/TPOT 优化资源和并行策略。([USENIX][20])

**发散思考**

追问：

```text
KV cache 传输成本怎么控制？
什么时候 PD 分离不划算？
prefill/decode pool 比例如何调？
```

一句话：

> PD 分离是把 compute-bound 的 prefill 和 memory/latency-bound 的 decode 用不同资源、不同 batch、不同并行策略优化。

---

## B10. Continuous batching 是什么？

**考察目的**

考你是否理解 LLM decode 是 iteration-level workload，而不是 request-level workload。

**第一性原理**

不同请求输出长度不同。如果 batch 固定到所有请求结束，短请求完成后会产生空洞，新请求也不能及时加入。

**回答**

static batching：

```text
凑一批请求
一起跑到全部结束
```

continuous batching：

```text
每个 decode iteration：
    移除 finished request
    加入 waiting request
    动态组成 batch
```

Orca 论文提出 iteration-level scheduling，即按 iteration 而不是 request 粒度调度，并结合 selective batching。([USENIX][21])

**发散思考**

追问：

```text
continuous batching 如何处理 fairness？
prefill 如何插入 decode batch？
KV cache block 如何动态管理？
```

一句话：

> continuous batching 把 batch 从“请求级固定集合”变成“每个 token step 可变集合”。

---

## B11. SGLang 的 completion、chat_completion、generate 有什么区别？

**考察目的**

考你是否分清 API 抽象层，而不是把所有生成接口混为一谈。

**第一性原理**

同样是生成 token，区别在输入表示和模板处理：

```text
raw prompt
messages
runtime-level generation
```

**回答**

completion：

```text
输入 raw prompt
不自动套 chat template
适合 base model、benchmark、手写 prompt
```

chat_completion：

```text
输入 messages
服务端套 chat template
适合 instruct/chat model
```

generate：

```text
更底层的原生生成抽象
输入 prompt/input_ids + sampling params
适合程序化控制、多段 generation、structured decoding
```

SGLang 文档说明它支持 OpenAI-compatible completions 和 chat completions API。([SGLang][22])

**发散思考**

追问：

```text
chat template 错了会怎样？
base model 用 chat_completion 有什么风险？
generate 为什么更适合复杂 LM program？
```

一句话：

> completion 是 raw prompt，chat_completion 是 messages + chat template，generate 是更底层的 runtime generation。

---

## B12. 为什么需要 SGL router？

**考察目的**

考你是否理解生产推理不是单 worker 跑模型，而是 control plane、routing、cache affinity、SLO 的问题。

**第一性原理**

多 worker、多副本、PD 分离、prefix cache、异构后端时，请求不能随机打到任意 worker。

**回答**

router 解决：

```text
1. 负载均衡
2. worker 健康检查
3. prefix cache affinity
4. prefill/decode 分离路由
5. DP/TP/EP 副本选择
6. 限流和队列
7. 协议适配
8. observability
9. 故障转移
```

SGLang Model Gateway 文档把它描述为大规模 LLM 部署的高性能模型路由网关，集中管理 worker lifecycle，并在 HTTP、gRPC、OpenAI-compatible 等异构协议之间做流量平衡。([GitHub][23])

**发散思考**

追问：

```text
cache affinity 和 load balance 如何权衡？
router 如何支持 PD 分离？
router 是 data plane 还是 control plane？
```

一句话：

> router 的价值不是转发请求，而是让请求在 cache、负载、SLO、worker 状态之间做最优分配。

---

## B13. 为什么 SGLang 最近拿 Rust 重写 RPC？

**考察目的**

考你是否知道推理系统中 GPU kernel 越快，Python control plane 越容易成为瓶颈。

**第一性原理**

RPC/control plane 做的是：

```text
连接管理
请求解析
tokenize/detokenize
队列
调度
流式返回
worker 通信
序列化/反序列化
```

这些是 CPU/I/O-heavy，不是 GPU-heavy。

**回答**

Python 问题：

```text
GIL
对象分配和 GC
async overhead
多进程 IPC 复杂
tail latency 抖动
内存占用
```

Rust 优势：

```text
Tokio async runtime
无 GIL
低内存
类型安全
gRPC/protobuf 集成稳定
更低 tail latency
```

SGLang 2026 年 RFC 提出 Rust-native gRPC server，通过 Tonic gRPC 和 Tokio runtime 运行，并逐步减少 GIL acquisition；另一个 RFC 提出把 non-GPU worker processes 从 Python 迁到 Rust，保留 Python 处理 GPU-accelerated workloads。([GitHub][24])

**发散思考**

追问：

```text
为什么 GPU worker 不一定全迁 Rust？
Rust RPC 对 tokenizer 有什么影响？
control plane latency 如何影响 GPU utilization？
```

一句话：

> Rust 重写 RPC 不是让 GPU 算得更快，而是让 control plane 不拖 GPU data plane。

---

## B14. FlashAttention 和 PagedAttention 分别解决什么问题？

**考察目的**

考你是否区分 attention kernel 优化和 KV cache memory management。

**第一性原理**

两个问题不同：

```text
FlashAttention：
    attention 计算中间矩阵 IO 太大。

PagedAttention：
    serving 中 KV cache 动态增长导致显存碎片和浪费。
```

**回答**

FlashAttention 解决：

```text
不 materialize QK^T 和 softmax matrix
用 tiling + online softmax 减少 HBM IO
```

PagedAttention 解决：

```text
把 KV cache 拆成 block
允许逻辑连续、物理不连续
减少显存碎片
支持 prefix sharing / dynamic batching
```

vLLM/PagedAttention 论文把 PagedAttention 描述为受操作系统 virtual memory/paging 启发的 attention 算法，用于 near-zero waste 的 KV cache memory 和跨请求 KV cache sharing。([arXiv][25])

**发散思考**

追问：

```text
FlashAttention 能否解决 serving KV fragmentation？
PagedAttention 是否减少 attention FLOPs？
两者如何结合？
```

一句话：

> FlashAttention 优化 attention 计算 IO，PagedAttention 优化 serving KV cache 分配和复用。

---

## B15. FlashAttention backward 为什么不保存完整 attention matrix？

**考察目的**

考你是否理解 memory-compute tradeoff。

**第一性原理**

保存完整 `P = softmax(QK^T)` 需要 O(N²) memory。重算 attention block 虽然增加 compute，但省大量 HBM memory。

**回答**

FA forward 保存少量统计量：

```text
m：row max
l：row sum
O：output
```

backward 时按 block 重算：

```text
S = QK^T
P = softmax(S)
再计算 dQ, dK, dV
```

这样避免保存 N² attention matrix。

**发散思考**

追问：

```text
为什么 recompute 可以更快？
训练时 memory bandwidth 和 compute 如何 tradeoff？
FA backward 数值稳定性怎么保证？
```

一句话：

> FA backward 用重计算换显存，避免保存 O(N²) attention matrix。

---

## B16. causal attention 和 bidirectional attention 在 tiling 上有什么区别？

**考察目的**

考你是否知道 mask 会改变有效 tile 和 workload balance。

**第一性原理**

bidirectional attention 中每个 Q 可以看所有 K；causal attention 中第 i 个 Q 只能看 `K[0:i]`。

**回答**

bidirectional：

```text
Q block × KV block 全部有效
tile 形状规则
```

causal：

```text
右上角 tile 被 mask
对角 tile 需要 partial mask
不同 Q block workload 不均匀
```

优化方式：

```text
跳过完全无效 tile
对 diagonal tile 使用 causal mask
调整 tile order 和 load balance
```

**发散思考**

追问：

```text
causal FA 为什么可以少算一部分 tile？
decode attention 是否还需要 causal mask？
变长 sequence 怎么处理 mask？
```

一句话：

> causal mask 让 attention tile 呈三角结构，优化重点是跳过无效 tile 和处理对角 tile。

---

## B17. GQA/MQA/MLA 对 KV cache 带宽有什么不同影响？

**考察目的**

考你是否能从 KV cache bytes/token 推导注意力结构的系统意义。

**第一性原理**

decode 阶段每步读取历史 KV。KV cache 越小，memory bandwidth 压力越低。

**回答**

MHA：

```text
每个 query head 都有独立 K/V
KV cache 最大
```

MQA：

```text
所有 query heads 共享一组 K/V
KV cache 大幅下降
可能损失模型质量或表达能力
```

GQA：

```text
多个 query heads 共享一组 K/V
介于 MHA 和 MQA 之间
```

MLA：

```text
缓存低维 latent C
通过矩阵吸收/投影参与 attention
KV cache 更小，但多一些 compute
```

**发散思考**

追问：

```text
GQA group 数如何影响带宽？
MLA 是否一定更快？
MQA 为什么可能影响质量？
```

一句话：

> MHA/GQA/MQA/MLA 的系统差异主要体现在 decode 阶段每 token 需要读取多少 KV cache。

---

## B18. MLA kernel 为什么可能从 memory-bound 变成 compute-bound？

**考察目的**

考你是否理解 MLA 是 memory-compute tradeoff。

**第一性原理**

MLA 减少 KV cache 读取，但增加 latent projection 或吸收后的额外矩阵计算。

**回答**

MHA decode：

```text
读完整 K/V
memory bandwidth 压力大
```

MLA decode：

```text
读低维 latent C
memory bandwidth 降低
但需要 Q projection、latent attention、output projection
```

当 memory bytes 大幅下降后，瓶颈可能转向 matrix compute 或 projection compute。硬件分析论文也指出 MLA 可降低 KV-cache size 和 memory bandwidth demand，并可能把 attention workload 推向 compute-bound regime。([arXiv][26])

**发散思考**

追问：

```text
MLA 对不同硬件是否收益一样？
latent dimension 选大选小有什么影响？
MLA 是否适合所有 batch size？
```

一句话：

> MLA 用额外计算换更少 KV 读取；当 bandwidth 不再是瓶颈时，compute 可能成为新瓶颈。

---

## B19. FlashDecoding 和 FlashAttention 的瓶颈为什么不同？

**考察目的**

考你是否区分 prefill/training attention 和 decode attention。

**第一性原理**

FA 主要处理大块 Q 和大块 K/V；decode 每步 Q 很小、K/V 很长。

**回答**

FlashAttention：

```text
Q block 大
QK^T 是矩阵乘
适合 Tensor Core
核心是减少 N² 中间矩阵 IO
```

FlashDecoding：

```text
Q 通常是 1 个或少量 token
扫描长 KV cache
核心是并行化 KV 维度、减少 memory bandwidth、合并 partial softmax
```

**发散思考**

追问：

```text
decode batch 增大是否能复用 FA？
KV split 后 partial softmax 如何 merge？
为什么 decode attention 难吃满 Tensor Core？
```

一句话：

> FA 优化大矩阵 attention，FlashDecoding 优化小 Q 扫长 KV 的 memory-bound 场景。

---

## B20. long context 下 attention 的瓶颈是算力、HBM，还是跨卡通信？

**考察目的**

考你是否能按场景分析瓶颈，而不是统一说 “attention O(N²)”。

**第一性原理**

瓶颈取决于：

```text
序列长度
batch size
head dimension
并行策略
KV 是否跨卡
是否 decode
```

**回答**

prefill long context：

```text
如果单卡：QK^T/PV 计算量大，可能 compute-bound 或 HBM-bound。
如果多卡 SP：跨卡 K/V 通信可能成为瓶颈。
```

decode long context：

```text
每步读长 KV cache，通常 memory bandwidth-bound。
多卡时还要看 KV 分布和通信。
```

Ring Attention 场景：

```text
瓶颈取决于 blockwise attention compute 是否能覆盖 K/V block 传输。
```

**发散思考**

追问：

```text
如何建立 long context roofline？
sequence parallel 如何 overlap 通信？
KV cache quantization 对 long context 有多大收益？
```

一句话：

> long context 不是固定一个瓶颈；prefill 看大矩阵计算和跨卡 SP，decode 看 KV cache bandwidth 和通信 overlap。

---

## B21. Ring Attention 如何做 online softmax merge？

**考察目的**

考你是否理解 Ring Attention 的正确性，不只是 “KV 环形传”。

**第一性原理**

不同 GPU/不同 step 看到的是不同 K/V block。每个 block 产生 partial softmax numerator 和 denominator，需要稳定合并。

**回答**

每个 Q row 维护：

```text
m：当前全局 max
l：当前 exp sum
acc：当前 numerator
```

新 block score 为 S：

```text
m_new = max(m, rowmax(S))
P = exp(S - m_new)
alpha = exp(m - m_new)

l_new   = alpha * l + rowsum(P)
acc_new = alpha * acc + P @ V_block
```

每次收到一个 KV block，就按这个公式合并。最终：

```text
O = acc / l
```

**发散思考**

追问：

```text
Ring 的 KV 顺序改变会不会影响浮点精度？
partial attention result 是否需要 all-reduce？
Ring 如何处理 causal mask？
```

一句话：

> Ring Attention 的 correctness 来自 online softmax 的可合并性。

---

## B22. Ulysses 和 Ring Attention 如何组合？

**考察目的**

考你是否知道 sequence parallel 不止一种，可以按 head 和 sequence 两个维度组合。

**第一性原理**

attention 有多个可切分维度：

```text
batch
head
sequence
hidden
```

不同切分对应不同通信模式。

**回答**

Ulysses 主要通过 all-to-all 在 sequence/head 之间重排，让每张卡拿到部分 heads 的完整 sequence。Ring 则让 K/V block 沿 sequence 维流动，每张卡边传边算。

组合方式：

```text
先用 Ulysses 在 head 维分组
再在每组内用 Ring 处理更长 sequence
```

或者按硬件拓扑：

```text
节点内用 Ulysses / all-to-all
节点间用 Ring overlap 通信
```

**发散思考**

追问：

```text
什么时候 Ulysses 更好？
什么时候 Ring 更好？
跨节点 IB 下 Ring 是否更容易 overlap？
```

一句话：

> Ulysses 擅长 head/sequence 重排，Ring 擅长流式 K/V overlap；两者可以按拓扑和模型形状组合。

---

## B23. KV cache quantization 的 scale 应该 per-token、per-channel 还是 per-block？

**考察目的**

考你是否理解 KV quantization 的精度和性能 tradeoff。

**第一性原理**

scale 粒度越细，精度越好，但 metadata 和 dequant 成本越高。

**回答**

per-tensor：

```text
scale 少，最快
精度最差
```

per-channel：

```text
适合不同 channel range 差异大
scale 数适中
```

per-token：

```text
适合 token 间 range 差异大
metadata 多
```

per-block：

```text
折中方案
适合 paged KV block 和 attention block
```

实际选择要看：

```text
KV 分布
head dim
block size
attention kernel 是否能高效加载 scale
```

**发散思考**

追问：

```text
K 和 V 是否应该用同样 scale 粒度？
KV4 为什么比 weight4 更敏感？
scale 存在哪里？
```

一句话：

> KV quant scale 的本质是精度和 metadata/dequant overhead 的折中。

---

## B24. RoPE 对 KV cache 压缩和矩阵吸收有什么影响？

**考察目的**

考你是否知道位置编码破坏某些线性吸收。

**第一性原理**

矩阵吸收要求变换是固定线性的。但 RoPE 是位置相关旋转：

```text
RoPE(x, pos)
```

不同 token 位置不同，因此不能简单把所有位置相关操作吸收到固定权重里。

**回答**

在 MLA 中，非 RoPE 部分可以做矩阵吸收：

```text
Q W_UK^T C^T
```

但带 RoPE 的 key/query 维度需要保留或特殊处理，因为它依赖 token position。常见做法是：

```text
partial RoPE
decoupled RoPE
缓存一小部分 rope key dims
latent 部分单独吸收
```

**发散思考**

追问：

```text
为什么 ALiBi 和 RoPE 对 kernel 影响不同？
RoPE 是否影响 KV cache quantization？
长上下文 RoPE scaling 对数值有什么影响？
```

一句话：

> RoPE 的位置相关性让完整矩阵吸收不再直接成立，MLA 通常需要把 rope 相关维度单独处理。

---

## B25. sliding window attention、chunked attention、prefix attention 的系统意义是什么？

**考察目的**

考你是否能把 attention pattern 和 serving/kernels 联系起来。

**第一性原理**

attention 成本来自每个 Q 看多少 K/V。减少可见范围或复用 prefix，可以减少计算和 KV 读取。

**回答**

sliding window：

```text
每个 token 只看最近 W 个 token
降低长上下文 decode KV 读取
适合局部依赖强的模型
```

chunked attention：

```text
把长序列分块处理
改善 prefill 调度和 memory peak
```

prefix attention：

```text
多请求共享相同 prefix
复用 KV cache
减少重复 prefill
```

**发散思考**

追问：

```text
prefix cache 如何路由？
sliding window 对模型质量有什么影响？
chunked prefill 如何影响 TTFT/TPOT？
```

一句话：

> 这些 attention pattern 的系统意义是减少每 token 可见 KV 或复用已有 KV，从而降低计算、显存和延迟。

---

## B26. speculative decoding 为什么能提升吞吐？什么时候不提升？

**考察目的**

考你是否理解 decode 的 sequential bottleneck，以及 draft/verify 的 tradeoff。

**第一性原理**

标准 decode 每步生成一个 token，严格串行。speculative decoding 用小模型一次草拟多个 token，大模型并行验证，从而减少大模型调用次数。

**回答**

流程：

```text
draft model 生成 k 个候选 token
target model 一次验证这些 token
接受一段 prefix
拒绝后回退
```

收益条件：

```text
draft model 便宜
acceptance rate 高
target verify 能并行处理多个 token
额外调度和 KV 管理成本低
```

不提升的情况：

```text
draft 太弱，acceptance rate 低
draft 太强，成本接近 target
batch 很大时 target 已经高效
structured decoding 导致接受率低
系统调度开销过高
```

**发散思考**

追问：

```text
draft 长度怎么选？
spec decode 和 continuous batching 冲突吗？
acceptance rate 如何影响 speedup？
```

一句话：

> speculative decoding 用便宜草稿换更少大模型串行步；收益取决于 draft 成本和接受率。

---

## B27. draft model 太弱或太强分别有什么问题？

**考察目的**

考你是否理解 speculative decoding 的成本模型。

**第一性原理**

期望收益大致取决于：

```text
accepted tokens per target call / extra draft cost
```

**回答**

draft 太弱：

```text
acceptance rate 低
经常只接受 0/1 个 token
target call 减少不明显
还多了 draft overhead
```

draft 太强：

```text
acceptance rate 高
但 draft 本身成本大
总成本不一定下降
```

理想 draft：

```text
足够便宜
分布接近 target
易于 batch
能快速生成多个 token
```

**发散思考**

追问：

```text
multi-draft 是否有用？
draft 是否共享 tokenizer？
draft KV cache 如何管理？
```

一句话：

> draft 模型要在便宜和准确之间取平衡；太弱不接受，太强不省钱。

---

## B28. structured decoding 为什么会影响 batching？

**考察目的**

考你是否知道输出约束不是只影响 logits，也影响 serving 系统。

**第一性原理**

structured decoding 对每个 request 的可选 token 集不同，会让 batch 内请求的 sampling/logits processing 变得异构。

**回答**

影响：

```text
1. 每个请求有不同 grammar state
2. token mask 不同
3. logits processor 无法完全统一
4. speculative decoding acceptance rate 可能下降
5. batch 内分支长度不同
6. CPU control plane 开销增加
```

优化：

```text
grammar state 放 GPU
batch 内按 grammar 类型分组
缓存 token mask
减少 CPU-GPU 往返
```

**发散思考**

追问：

```text
JSON schema decoding 如何做高效？
grammar mask 在 CPU 还是 GPU？
structured decoding 和 router 有什么关系？
```

一句话：

> structured decoding 增加了 batch 内异构性，会影响 logits processing、sampling、spec decode 和调度效率。

---

## B29. TTFT、TPOT、throughput、goodput 分别怎么定义？

**考察目的**

考你是否能用正确指标讨论 serving。

**第一性原理**

不同指标衡量不同用户体验和系统效率：

```text
首 token 等多久
后续 token 多快
系统总产出多少
满足 SLO 的有效产出多少
```

**回答**

```text
TTFT：
    time to first token，从请求进入到首 token 返回。
    主要受排队、prefill、调度影响。

TPOT：
    time per output token，decode 阶段平均每 token 时间。
    主要受 decode batching、KV bandwidth、scheduler 影响。

Throughput：
    tokens/sec 或 requests/sec。
    只看产出总量。

Goodput：
    满足 SLO 的 tokens/sec 或 requests/sec。
    比 throughput 更适合生产系统。
```

**发散思考**

追问：

```text
为什么 throughput 高但 goodput 低？
PD 分离如何分别优化 TTFT 和 TPOT？
scheduler 应该优化平均值还是 tail？
```

一句话：

> TTFT 看首响，TPOT 看持续生成，throughput 看总产出，goodput 看满足 SLO 的有效产出。

---

## B30. prefill chunking 解决什么问题？

**考察目的**

考你是否理解长 prompt prefill 会阻塞 decode。

**第一性原理**

长 prompt prefill 是大块 compute，如果一次性占用 GPU，会让 decode 请求等待，TPOT 和 tail latency 变差。

**回答**

prefill chunking 把长 prompt 分成多个 chunk：

```text
chunk 1 prefill
插入 decode steps
chunk 2 prefill
插入 decode steps
...
```

收益：

```text
1. 降低 decode 被长 prefill 阻塞
2. 改善 tail latency
3. 更好 continuous batching
4. 降低单次 memory peak
```

代价：

```text
1. prefill 总时间可能变长
2. scheduler 更复杂
3. chunk size 需要调参
```

**发散思考**

追问：

```text
chunk size 怎么选？
chunked prefill 和 PD 分离是否都需要？
chunking 如何影响 TTFT？
```

一句话：

> prefill chunking 用更细粒度调度长 prompt，避免大 prefill 阻塞 decode。

---

## B31. prefix cache 为什么需要 cache-aware routing？

**考察目的**

考你是否知道 prefix cache 命中取决于请求打到哪里。

**第一性原理**

prefix KV cache 存在某个 worker 上。如果相同 prefix 的请求被路由到别的 worker，就无法复用，除非跨 worker 迁移 KV。

**回答**

cache-aware routing 会考虑：

```text
该 worker 是否已有 prefix cache
worker 当前负载
cache 是否过期
KV cache 空间
SLO
```

简单策略：

```text
有 cache 且负载可接受 → 路由过去
无 cache 或负载过高 → 选择其他 worker
```

难点是 cache affinity 和 load balance 冲突。

**发散思考**

追问：

```text
cache 命中率和 tail latency 如何 tradeoff？
是否值得迁移 KV cache？
prefix cache key 如何设计？
```

一句话：

> prefix cache 只有在路由命中持有 cache 的 worker 时才有价值，因此 router 必须 cache-aware。

---

## B32. KV cache block size 怎么选？

**考察目的**

考你是否理解 PagedAttention 的 paging tradeoff。

**第一性原理**

block size 越小，碎片越少，但 metadata 和调度开销越高；block size 越大，管理开销低，但内部碎片增加。

**回答**

小 block：

```text
+ 显存浪费少
+ 适合短请求和变长请求
- block table 更大
- kernel address translation 开销高
```

大 block：

```text
+ metadata 少
+ kernel 更规整
- 最后一个 block 内部碎片多
- prefix sharing 粒度粗
```

选择取决于：

```text
平均 prompt/output 长度
batch size
KV head dim
kernel address translation 成本
prefix sharing 需求
```

**发散思考**

追问：

```text
block size 如何影响 attention kernel？
KV block 是否应该固定 token 数？
多模型 serving 是否需要不同 block size？
```

一句话：

> KV block size 是显存碎片、metadata、kernel 复杂度和 cache sharing 粒度的折中。

---

## B33. PagedAttention 如何解决 KV cache 碎片？

**考察目的**

考你是否理解 PagedAttention 借鉴虚拟内存。

**第一性原理**

请求长度动态增长，如果 KV cache 要求物理连续，就会产生外部碎片和过度预留。

**回答**

PagedAttention 把每个请求的 KV cache 切成固定 token 数的 KV blocks。逻辑上 sequence 连续，物理上 block 可以不连续，通过 block table 映射。

vLLM 文档也说明 PagedAttention 的核心是把每个请求的 KV cache 分成 KV Blocks，允许它们存储在非连续物理内存中，从而按需分配并消除 memory fragmentation。([vLLM][27])

收益：

```text
1. 减少外部碎片
2. 减少预留浪费
3. 支持 prefix sharing
4. 支持 continuous batching
```

**发散思考**

追问：

```text
PagedAttention 是否有 address translation overhead？
block table 放哪里？
prefix sharing 如何引用计数？
```

一句话：

> PagedAttention 把 KV cache 管理从连续数组变成虚拟内存式分页。

---

## B34. continuous batching 如何处理 fairness？

**考察目的**

考你是否知道吞吐最大化可能导致某些请求被饿死。

**第一性原理**

scheduler 每步选择哪些请求进入 batch。如果只选最容易处理的请求，长请求或低优先级请求可能长期等待。

**回答**

fairness 机制：

```text
1. waiting time aging
2. per-request token budget
3. prefill/decode quota
4. priority queue
5. max batch delay
6. SLO-aware scheduling
7. 防止长 prompt 永久被切碎
```

**发散思考**

追问：

```text
fairness 和 throughput 如何冲突？
短请求优先是否合理？
如何避免 decode 被 prefill 饿死？
```

一句话：

> continuous batching 要在 GPU 利用率和请求公平性之间做调度权衡。

---

## B35. decode batch 太大为什么会伤 latency？

**考察目的**

考你是否理解吞吐和延迟不是同一个目标。

**第一性原理**

batch 越大，单步处理 token 越多，吞吐可能提高；但每个 step 的计算/访存时间也变长，单请求 TPOT 和 tail latency 可能变差。

**回答**

大 decode batch 的问题：

```text
1. 每 step latency 增大
2. KV cache 总读取量增大
3. memory bandwidth 饱和
4. sampling/logits processing 更慢
5. 请求间异构性增加
6. 新请求插入等待更久
```

**发散思考**

追问：

```text
max_num_batched_tokens 怎么调？
batch size 和 token budget 区别？
goodput 为什么比 throughput 更重要？
```

一句话：

> decode batch 大能提高吞吐，但会增加每步延迟；生产系统要按 SLO 找最优 batch，不是越大越好。

---

## B36. 为什么 serving 系统里 queueing delay 经常比 kernel latency 更重要？

**考察目的**

考你是否理解系统瓶颈不只在 kernel。

**第一性原理**

用户看到的延迟是：

```text
总延迟 = 排队 + 调度 + prefill + decode + postprocess + 网络
```

kernel latency 只是其中一部分。

**回答**

高负载下，排队延迟会快速放大。即使单次 decode kernel 很快，如果 scheduler 无法及时安排请求，TTFT 和 tail latency 仍然很差。

影响 queueing 的因素：

```text
arrival rate
batch policy
prefill/decode 混部
SLO
worker 数量
router 策略
KV cache 空间
```

**发散思考**

追问：

```text
如何做 admission control？
如何用排队论解释 tail latency？
为什么 goodput 会掉？
```

一句话：

> serving 优化不是只优化 kernel，而是优化整个请求队列和资源分配。

---

## B37. PD 分离下 KV cache 怎么传？传 tensor、传 block handle，还是重算？

**考察目的**

考你是否理解 PD 分离的主要代价就是 KV transfer。

**第一性原理**

prefill 产生 KV，decode 需要 KV。分离后必须解决 KV ownership 和传输。

**回答**

方案：

```text
1. 传完整 KV tensor：
   简单，但带宽开销大。

2. 传 KV block：
   适合 PagedAttention，按 block 迁移。

3. 传 handle / remote reference：
   decode 远程访问或延迟拉取，复杂。

4. 重算 prefix：
   避免传输，但浪费 compute，只适合短 prefix 或带宽极差场景。
```

选择取决于：

```text
prompt 长度
网络带宽
decode 预计长度
KV cache 大小
P/D worker 拓扑
```

**发散思考**

追问：

```text
NVLink 内和 IB 跨节点是否策略不同？
KV 传输能否和 decode overlap？
prefix cache 如何参与？
```

一句话：

> PD 分离的收益来自两阶段独立优化，代价集中在 KV cache 传输和所有权管理。

---

## B38. disaggregated serving 什么时候不划算？

**考察目的**

考你是否能讲 tradeoff，不是所有系统都应该 PD 分离。

**第一性原理**

拆分只有当收益大于通信和调度成本时才有意义。

**回答**

不划算情况：

```text
1. prompt 很短，prefill 很小
2. decode 很短，KV 传输不摊销
3. P/D 之间带宽低
4. worker 数量少，调度复杂度不值得
5. prefix cache 本地命中很高
6. workload 稳定，不存在明显 prefill/decode 干扰
```

**发散思考**

追问：

```text
如何判断是否需要 PD 分离？
哪些指标能证明收益？
PD 分离是否会伤 TTFT？
```

一句话：

> PD 分离适合 prefill/decode 特性差异大且干扰严重的场景；短请求或低带宽环境可能不划算。

---

## B39. MoE serving 中 expert load imbalance 怎么处理？

**考察目的**

考你是否理解 MoE 的瓶颈往往不是 expert compute，而是 token 分布不均。

**第一性原理**

router 选择 expert 后，某些 expert 可能收到大量 token，形成热点，其他 expert 空闲。

**回答**

处理方法：

```text
1. load balancing loss
2. capacity factor
3. token dropping
4. expert replication
5. expert parallel re-sharding
6. router bias 修正
7. batch 内 padding / sorting
8. 热点 expert 动态迁移或复制
```

**发散思考**

追问：

```text
capacity factor 越大越好吗？
token dropping 对质量影响？
expert replication 如何路由？
```

一句话：

> MoE serving 的关键是让 token 分布、expert compute、A2A 通信三者平衡。

---

## B40. EP 中 token dropping、capacity factor、padding 各有什么代价？

**考察目的**

考你是否知道 MoE dispatch 不是纯通信问题，还有质量和计算浪费。

**第一性原理**

每个 expert 每 batch 能处理的 token 数有限。超过 capacity 就要处理溢出。

**回答**

capacity factor：

```text
capacity = expected_tokens_per_expert × factor
```

factor 大：

```text
+ 少 drop
- padding 和计算浪费多
```

factor 小：

```text
+ 计算更紧凑
- token drop 风险高
```

token dropping：

```text
+ 控制最坏负载
- 损失模型质量
```

padding：

```text
+ 让 expert GEMM 形状规整
- 计算无效 token
```

**发散思考**

追问：

```text
训练和推理时 capacity 策略是否一样？
如何减少 padding waste？
top-2 routing 比 top-1 复杂在哪里？
```

一句话：

> capacity/padding/drop 是 MoE 在负载均衡、计算效率和模型质量之间的三角权衡。

---

## B41. all-to-all 和 all-reduce 在推理系统中的瓶颈差异是什么？

**考察目的**

考你是否理解不同 collective 的通信模式和系统风险。

**第一性原理**

```text
all-reduce：
    每个 rank 有同形 tensor，做聚合。

all-to-all：
    每个 rank 给每个 rank 发送不同数据。
```

**回答**

all-reduce 常见于 tensor parallel：

```text
通信量规整
实现成熟
延迟和带宽可预测
```

all-to-all 常见于 MoE token dispatch 或 Ulysses SP：

```text
数据量可能不均匀
metadata 复杂
容易受 load imbalance 影响
tail rank 决定整体时间
```

**发散思考**

追问：

```text
all-to-all 为什么更怕 token imbalance？
NVLink 和 IB 下 collective 策略差异？
NCCL 对不同 collective 怎么优化？
```

一句话：

> all-reduce 是规整聚合，all-to-all 是不规则重排；后者更容易被负载不均和 tail latency 拖慢。

---

## B42. NVLink 域内和 IB 跨节点的 parallelism 策略为什么不同？

**考察目的**

考你是否能把并行策略和硬件拓扑绑定。

**第一性原理**

通信带宽/延迟决定哪些 parallelism 可行：

```text
高带宽低延迟：可以频繁同步
低带宽高延迟：必须减少同步或 overlap
```

**回答**

NVLink/NVSwitch 域内：

```text
适合 tensor parallel
频繁 all-reduce/all-gather 可接受
KV 迁移更便宜
```

IB 跨节点：

```text
适合 pipeline parallel、data parallel、coarser-grained SP
需要减少同步频率
需要更强 overlap
```

Ring Attention 跨节点可能比一次性 AllGather 更容易 overlap，但仍要看 compute 是否覆盖通信。

**发散思考**

追问：

```text
TP 为什么通常放节点内？
EP 跨节点时 all-to-all 怎么优化？
PD 分离跨节点是否划算？
```

一句话：

> 并行策略要贴着拓扑设计：NVLink 内可以细粒度同步，IB 跨节点要粗粒度、少同步、强 overlap。

---

## B43. router 如何做 cache affinity 和 load balance 的权衡？

**考察目的**

考你是否理解 router 的核心不是简单最小负载，而是多目标优化。

**第一性原理**

cache hit 降低 prefill/KV 成本，但过度追求 cache affinity 会把请求打爆某个 worker。

**回答**

router scoring 可以综合：

```text
score = cache_bonus
      - load_penalty
      - queue_penalty
      - KV_memory_penalty
      - SLO_risk
```

策略：

```text
1. cache hit worker 负载低 → 路由过去
2. cache hit worker 过载 → 找次优 cache 或冷路由
3. 大 prompt 更偏 cache affinity
4. 短请求更偏低 queue latency
```

**发散思考**

追问：

```text
cache 命中率和 p99 latency 如何同时优化？
是否需要迁移 cache？
router 是否需要预测 decode 长度？
```

一句话：

> router 要在“复用已有 cache”和“避免热点排队”之间做动态权衡。

---

## B44. 为什么 OpenAI-compatible API 不是生产 serving 的全部？

**考察目的**

考你是否知道 API 只是入口，生产系统还需要调度、隔离、观测、控制面。

**第一性原理**

用户请求进入后，还需要：

```text
鉴权
限流
排队
路由
batching
KV 管理
失败恢复
日志
计费
SLO
```

**回答**

OpenAI-compatible API 解决：

```text
接口兼容
client 易接入
```

但不解决：

```text
worker 生命周期
cache-aware routing
PD 分离
multi-tenant isolation
priority scheduling
observability
admission control
成本控制
```

**发散思考**

追问：

```text
为什么 router/gateway 是单独组件？
production serving 如何处理 backpressure？
API 层和 scheduler 层如何交互？
```

一句话：

> API 是用户入口，serving 的难点在 control plane、scheduler、cache、资源隔离和 SLO。

---

## B45. CUDA Graph 在推理中解决什么问题？什么时候不好用？

**考察目的**

考你是否理解 launch overhead 和动态 shape 的矛盾。

**第一性原理**

GPU kernel 很快时，CPU launch overhead、调度 overhead 会占比变大。CUDA Graph 可以捕获固定执行图，减少反复 launch 成本。

**回答**

适合：

```text
固定 batch / shape
重复执行相同 kernel sequence
decode step 形态稳定
```

收益：

```text
减少 CPU launch overhead
降低调度抖动
提高小 batch latency
```

不好用：

```text
dynamic batch
变长 sequence
动态 sampling/grammar
频繁 shape 变化
不同请求混合
```

解决方式：

```text
多 graph cache
padding 到固定 bucket
hybrid graph + eager
```

**发散思考**

追问：

```text
continuous batching 和 CUDA Graph 是否冲突？
graph cache 如何设计？
capture 时内存地址是否固定？
```

一句话：

> CUDA Graph 用静态执行图减少 launch overhead，但动态 serving 需要 bucket/cache/hybrid 才能用好。

---

## B46. 多租户 serving 如何隔离 KV cache、batch 和优先级？

**考察目的**

考你是否理解生产系统中不同租户不能互相拖垮。

**第一性原理**

共享 GPU 时，资源包括：

```text
compute
HBM
KV cache
batch slots
queue capacity
network bandwidth
```

**回答**

隔离方法：

```text
1. per-tenant quota
2. KV cache reservation / limit
3. priority queue
4. max tokens per tenant
5. admission control
6. preemption / cancellation
7. rate limiting
8. batch 分组或加权公平调度
```

**发散思考**

追问：

```text
如何防止一个长上下文请求占满 KV？
premium 用户如何保证 p99？
多租户是否应该共享 prefix cache？
```

一句话：

> 多租户 serving 要隔离的不只是请求数，还有 KV cache、batch token budget、优先级和 tail latency。

---

## B47. SLO-aware scheduler 应该优化平均 latency 还是 tail latency？

**考察目的**

考你是否理解生产系统关注 p95/p99，而不仅是平均值。

**第一性原理**

用户体验和 SLA 通常由 tail latency 决定。平均延迟好但 p99 爆炸，系统仍然不可用。

**回答**

SLO-aware scheduler 应该优化：

```text
满足 SLO 的 goodput
p95/p99 latency
deadline miss rate
```

而不是单纯平均 latency 或 throughput。

策略：

```text
deadline-aware priority
aging
short-job-first 的变体
prefill/decode quota
admission control
动态 batch size
```

**发散思考**

追问：

```text
short request 优先是否会饿死 long request？
goodput 如何定义？
高负载下是排队还是拒绝请求？
```

一句话：

> 生产 scheduler 目标是最大化满足 SLO 的有效吞吐，而不是漂亮的平均延迟。

---

## B48. 多模型 serving 下如何做 admission control？

**考察目的**

考你是否知道超载时不能无限排队。

**第一性原理**

当 arrival rate 超过服务能力，排队会无限增长，所有请求 tail latency 都会恶化。admission control 是在系统崩溃前拒绝或降级。

**回答**

维度：

```text
当前 queue length
预计 prefill tokens
预计 decode tokens
KV cache 可用量
tenant quota
SLO deadline
model priority
```

策略：

```text
1. reject
2. degrade 到小模型
3. 降低 max output tokens
4. 延迟低优先级请求
5. 限制长上下文请求
6. 动态扩容 worker
```

**发散思考**

追问：

```text
如何估计 decode 长度？
reject 是否影响 goodput？
多模型共享 GPU 时如何避免大模型挤掉小模型？
```

一句话：

> admission control 的本质是用有限拒绝换系统稳定和 SLO。

---

## B49. weight-only quantization 为什么常用于 decode？

**考察目的**

考你是否理解 decode 中权重读取和 KV 读取是主要带宽压力。

**第一性原理**

decode 每步 batch 小，GEMM 可能更接近 GEMV，权重读带宽占比高。weight-only quantization 减少权重 bytes。

**回答**

weight-only 量化：

```text
W4A16 / W4A8 等
权重量化，activation 保持高精度或较高精度
```

优势：

```text
1. 模型显存下降
2. 权重带宽下降
3. calibration 相对简单
4. 不需要处理 activation outlier 的全部复杂性
```

劣势：

```text
1. dequant overhead
2. scale metadata
3. batch 大时可能不如 W8A8
```

**发散思考**

追问：

```text
为什么训练更常用 FP16/BF16，而推理可 W4？
weight-only 对 prefill 是否同样收益？
```

一句话：

> weight-only quantization 特别适合 decode 小 batch，因为它直接减少反复读取权重的带宽和显存。

---

## B50. W8A8 和 W4A16 的性能瓶颈分别是什么？

**考察目的**

考你是否能比较不同量化策略。

**第一性原理**

不同量化方案减少的 bytes 不同，引入的 dequant/scale 成本也不同。

**回答**

W8A8：

```text
weight 和 activation 都 INT8
适合矩阵乘走 INT8 Tensor Core
难点是 activation outlier 和 scale 管理
```

W4A16：

```text
weight INT4，activation FP16/BF16
主要减少权重带宽和显存
难点是 weight dequant 和 packing/unpacking
```

性能瓶颈：

```text
W8A8：
    activation quant/dequant、outlier、scale、INT8 kernel 支持

W4A16：
    INT4 unpack/dequant、scale 读取、低 batch 下 kernel overhead
```

**发散思考**

追问：

```text
为什么 W4A16 不一定比 W8A8 快？
W8A8 为什么需要 SmoothQuant？
W4A8 为什么更难？
```

一句话：

> W8A8 难在 activation，W4A16 难在 INT4 dequant 和 packing；端到端是否更快取决于 kernel 和 batch。

---

## B51. per-tensor、per-channel、per-group scale 的 tradeoff？

**考察目的**

考你是否理解量化 scale 粒度。

**第一性原理**

scale 粒度越细，拟合分布越好，但 metadata 和计算成本越高。

**回答**

per-tensor：

```text
一个 tensor 一个 scale
最快最简单
精度最差
```

per-channel：

```text
每个 output/input channel 一个 scale
精度好
metadata 适中
```

per-group：

```text
每 G 个元素一组 scale
低比特常用
精度和性能折中
```

**发散思考**

追问：

```text
group size 如何选？
scale 是否参与 Tensor Core？
scale 放 epilogue 还是 mainloop？
```

一句话：

> scale 粒度是精度、metadata、dequant 成本之间的权衡。

---

## B52. INT4 packing 为什么让 kernel 更复杂？

**考察目的**

考你是否知道 INT4 不是普通 dtype，而是 packed representation。

**第一性原理**

INT4 一个值 4 bit，通常多个值打包在一个 byte/word 里。计算前要按硬件要求解包或按特定 layout 喂 Tensor Core。

**回答**

复杂点：

```text
1. 两个 INT4 packed 到一个 byte
2. 内存 layout 要适配 Tensor Core
3. 需要 bit extract / reorder
4. scale 和 zero point 对齐 group
5. vectorized load 需要对齐
6. dequant 和 MMA pipeline 要融合
```

**发散思考**

追问：

```text
INT4 unpack 在 CUDA core 还是 Tensor Core？
weight prepacking 有什么作用？
group size 如何影响 packing？
```

一句话：

> INT4 的难点是 packed layout 和 dequant pipeline，不是单纯把 float 换成 int4。

---

## B53. activation outlier 为什么比 weight outlier 更麻烦？

**考察目的**

考你是否理解动态数据和静态数据的区别。

**第一性原理**

weight 是固定的，可以离线统计、重排、缩放；activation 随输入变化，分布动态且可能有极端 outlier。

**回答**

activation outlier 麻烦在：

```text
1. 输入相关，不能完全离线固定
2. outlier 会拉大量化 scale
3. 对 W8A8 影响大
4. batch/token 之间分布差异明显
5. 动态 scale 成本高
```

SmoothQuant 的动机就是 activation 难量化、weight 相对容易量化，因此把困难从 activation 迁移到 weight。([arXiv][16])

**发散思考**

追问：

```text
为什么 LayerNorm 后仍有 outlier？
activation clipping 是否可行？
SmoothQuant 如何利用 weight 静态性？
```

一句话：

> weight outlier 可以离线处理，activation outlier 动态出现，会直接破坏 W8A8 的 scale 效率。

---

## B54. SmoothQuant 和 AWQ 能不能组合？

**考察目的**

考你是否理解两者作用对象不同。

**第一性原理**

SmoothQuant 主要平滑 activation，以做 W8A8；AWQ 主要保护重要 weight channel，以做低比特 weight-only。

**回答**

理论上可以组合，但要小心 scale 相互作用：

```text
SmoothQuant：
    X / s, W * s

AWQ：
    根据 activation 统计对 salient weight channel scaling
```

组合时需要重新校准：

```text
activation range
weight range
group scale
error metric
```

否则一个 scaling 可能破坏另一个方法的假设。

**发散思考**

追问：

```text
先 SmoothQuant 还是先 AWQ？
组合后 scale 能否 merge？
对 W4A8 是否有意义？
```

一句话：

> 可以组合，但必须重新推导和校准 scale；不能简单把两个 PTQ recipe 叠起来。

---

## B55. GPTQ、AWQ、SmoothQuant、QuaRot、QServe 核心差别是什么？

**考察目的**

考你是否能按“解决哪个量化难点”分类。

**第一性原理**

量化方法可以按目标分：

```text
weight-only 精度
activation outlier
硬件效率
KV cache
旋转/变换分布
```

**回答**

```text
GPTQ：
    weight-only，使用二阶近似/重构思想降低量化误差。

AWQ：
    activation-aware，保护 salient weight channel。

SmoothQuant：
    W8A8，把 activation outlier 迁移到 weight。

QuaRot：
    用旋转变换让 activation/weight 分布更量化友好。

QServe/QoQ：
    W4A8KV4，强调算法和 serving kernel/system co-design，降低 dequant overhead。
```

**发散思考**

追问：

```text
哪类方法更适合 cloud serving？
哪类更适合 edge？
为什么系统 co-design 很重要？
```

一句话：

> 不同量化方法不是谁绝对更好，而是分别解决 activation、weight、KV、dequant overhead 和硬件映射问题。

---

## B56. KV cache quantization 为什么比 weight quantization 更容易影响长上下文？

**考察目的**

考你是否理解 KV error 会在 attention 中反复使用。

**第一性原理**

weight 是每层固定参与一次计算；KV cache 中历史 token 会在未来很多 decode step 中反复被 attention 读取。

**回答**

KV quantization 敏感原因：

```text
1. 长上下文中每个历史 KV 被多次使用
2. K 误差影响 attention score
3. V 误差影响 weighted sum
4. softmax 对 score 误差敏感
5. token 越长，累积影响越明显
```

**发散思考**

追问：

```text
K 和 V 哪个更敏感？
KV scale 应该如何校准？
SmoothAttention 解决什么？
```

一句话：

> KV cache 量化误差会在长上下文 decode 中被反复放大，尤其 K 的误差会影响 softmax 分布。

---

## B57. FP8 和 INT8 的系统区别是什么？

**考察目的**

考你是否知道二者都是 8-bit，但数值语义不同。

**第一性原理**

INT8 是整数，需要 scale 映射；FP8 是低精度浮点，有指数位和尾数位，动态范围不同。

**回答**

INT8：

```text
定点量化
依赖 scale / zero point
适合整数 Tensor Core
```

FP8：

```text
浮点格式，如 E4M3/E5M2
动态范围更灵活
通常也需要 scale
更适合训练/Transformer Engine 这类动态范围场景
```

**发散思考**

追问：

```text
E4M3 和 E5M2 区别？
FP8 是否不需要 scale？
Transformer Engine 做什么？
```

一句话：

> INT8 是整数映射问题，FP8 是低精度浮点表示问题；二者都需要 scale，但误差形态不同。

---

## B58. FP4/NVFP4 的 scale 设计为什么关键？

**考察目的**

考你是否理解 4-bit 浮点动态范围和精度都极小。

**第一性原理**

bit 数越少，表示范围越窄。没有合适 scale，数值要么 overflow，要么 underflow，要么大量值挤在少数 bins。

**回答**

FP4/NVFP4 通常需要 block scaling：

```text
每个小 block 一个 scale
block 内值用 FP4 表示
```

scale 太粗：

```text
outlier 拉大 scale
普通值精度差
```

scale 太细：

```text
metadata 多
读取 scale 成本高
kernel 复杂
```

**发散思考**

追问：

```text
block size 如何影响精度和性能？
FP4 更适合 weight、activation 还是 KV？
scale 是否能融合到 epilogue？
```

一句话：

> FP4 的有效性很大程度取决于 block scale；scale 设计不好，4-bit 带来的带宽收益会被精度或 overhead 抵消。

---

# C. Agent 设计

## C1. Pi 的设计理念

**考察目的**

考你是否理解 agent framework 不一定要复杂。Pi 这类 minimal coding agent 的核心是强执行环境，而不是堆很多抽象层。

**第一性原理**

agent 最小闭环是：

```text
observe
→ decide
→ tool execution
→ observe result
→ repeat
```

如果工具足够强，比如 shell、文件系统、代码执行，agent 可以自己构造临时工具。

**回答**

Pi 的理念可以概括为：

```text
minimal harness
powerful execution
少量强工具
让 agent 通过写代码和运行代码扩展能力
```

Armin Ronacher 对 Pi 的介绍把它称为 OpenClaw 中的 minimal coding agent，并强调这种 minimal agent harness 的价值；OpenClaw 文档也说明它通过 pi SDK 直接嵌入 `AgentSession`。([Armin Ronacher's Thoughts and Writings][28])

**发散思考**

追问：

```text
为什么工具少反而可能更强？
shell 是否是通用工具？
minimal agent 和复杂 multi-agent framework 哪个更可靠？
```

一句话：

> Pi 的核心是少量强工具加简单 agent loop，而不是复杂 agent graph。

---

## C2. Claude Code 的设计理念

**考察目的**

考你是否理解 coding agent 的复杂度不在主循环，而在权限、上下文、工具、状态和恢复。

**第一性原理**

核心 loop 很简单：

```text
user task
→ model decides action
→ tool executes
→ result appended
→ repeat
```

真正难的是让这个 loop 安全、可控、可恢复。

**回答**

Claude Code 的设计可以概括为：

```text
simple while-loop
strong permission system
context compaction
tool orchestration
session persistence
subagent delegation
MCP / hooks / extensibility
```

用户给的论文分析 Claude Code 源码后指出，其核心是 simple while-loop，同时周围有 safety、context management、extensibility 等支撑系统。([arXiv][29])

**发散思考**

追问：

```text
为什么 agent loop 简单但系统复杂？
权限系统怎么设计？
context compaction 如何不丢状态？
```

一句话：

> Claude Code 不是复杂 agent graph，而是简单 loop 加强大的工程护栏。

---

## C3. 如果你来设计一个 Agent 框架，怎么设计？

**考察目的**

考你是否能从系统角度设计 agent runtime，而不是说几个 buzzword。

**第一性原理**

agent 是带工具的状态机：

```text
state
→ model policy
→ action
→ environment transition
→ new state
```

**回答**

我会设计六层：

```text
1. Model layer：
   small/strong/specialized model routing

2. Tool runtime：
   schema、timeout、权限、side effect、retry、structured output

3. Permission layer：
   read、write、run command、network、external side effect 分级

4. Context layer：
   task state、plan、diff、test result、failure history、compaction

5. Execution sandbox：
   workspace、rollback、secret redaction、network policy、test runner

6. Observability / eval：
   tool calls、latency、cost、diff、success、unsafe action、user intervention
```

核心 loop：

```text
while not done:
    observe
    plan
    act
    verify
    update state
```

**发散思考**

追问：

```text
什么时候需要 multi-agent？
如何防 prompt injection？
如何评估 agent？
如何做 long-horizon recovery？
```

一句话：

> 我不会先堆多 agent，而会先做可靠 runtime：工具强类型、权限清晰、上下文可压缩、执行可回滚、结果可验证。

---

## C4. ReAct loop 和 plan-and-execute loop 有什么区别？

**考察目的**

考你是否理解 agent 编排模式。

**第一性原理**

agent 做任务时有两类需求：

```text
边观察边行动
先规划再执行
```

**回答**

ReAct：

```text
Thought → Action → Observation → Thought → ...
```

优点：

```text
适合动态环境
能根据工具结果调整
```

缺点：

```text
容易短视
长任务可能失去全局结构
```

plan-and-execute：

```text
先生成计划
再逐步执行
必要时 replan
```

优点：

```text
适合长任务
结构更清楚
```

缺点：

```text
初始计划可能过时
计划太长容易空想
```

**发散思考**

追问：

```text
什么时候需要 replan？
如何避免计划幻觉？
coding agent 更适合哪种？
```

一句话：

> ReAct 强在动态反馈，plan-and-execute 强在长任务结构；实际系统通常混合使用。

---

## C5. 为什么 agent framework 的核心不是多 agent，而是 tool runtime？

**考察目的**

考你是否理解 agent 成败取决于能否可靠行动，而不是角色扮演数量。

**第一性原理**

agent 的动作通过工具改变环境。如果工具不可靠、无权限边界、无结构化输出，多 agent 只会放大错误。

**回答**

tool runtime 需要：

```text
schema
权限
timeout
retry
idempotency
dry run
structured output
audit log
sandbox
```

multi-agent 只有在这些情况才有意义：

```text
任务可并行
需要独立上下文
需要权限隔离
需要 reviewer
```

**发散思考**

追问：

```text
multi-agent 什么时候是噪声？
subagent 如何限制预算？
tool schema 写不好有什么后果？
```

一句话：

> 先把工具执行做可靠，再谈多 agent；否则只是多个模型一起不可靠。

---

## C6. prompt injection 在 RAG/tool/browser agent 中怎么防？

**考察目的**

考你是否理解外部内容不能当指令。

**第一性原理**

agent 输入有不同信任等级：

```text
system/developer instruction
user instruction
trusted tool output
untrusted external content
```

外部网页、邮件、文档只能作为 data，不能覆盖上层指令。

**回答**

防御：

```text
1. 明确信任边界
2. 外部内容加 sandbox tag
3. tool permission gating
4. 高风险动作 human approval
5. secret redaction
6. 不把外部内容直接拼成系统指令
7. 引用/溯源
8. allowlist 工具
```

**发散思考**

追问：

```text
RAG 文档里写“忽略之前指令”怎么办？
浏览器 agent 如何防下载恶意脚本？
邮件 agent 如何防自动转账/发信？
```

一句话：

> prompt injection 的核心防线是信任分层：外部内容是数据，不是指令。

---

## C7. Agent 的 memory 应该存什么，不应该存什么？

**考察目的**

考你是否理解 memory 不是把聊天记录无限存。

**第一性原理**

memory 的价值是减少重复上下文，但错误 memory 会污染未来决策。

**回答**

应该存：

```text
稳定偏好
项目约定
长期目标
已确认事实
重要决策
任务状态 checkpoint
```

不应该存：

```text
临时猜测
未验证信息
敏感 secret
一次性噪声
用户不希望保留的信息
```

**发散思考**

追问：

```text
memory 如何过期？
如何让用户编辑 memory？
项目 memory 和个人 memory 如何分开？
```

一句话：

> memory 应该存稳定、可验证、长期有用的信息，而不是无限记录所有上下文。

---

## C8. context compaction 怎么保证不丢关键状态？

**考察目的**

考你是否理解长任务中上下文窗口是稀缺资源。

**第一性原理**

compaction 不是总结聊天，而是压缩任务状态。

**回答**

应该保留：

```text
目标
约束
已完成动作
文件 diff
测试结果
失败尝试
当前假设
剩余风险
下一步
```

形式：

```text
raw transcript
→ event log
→ state summary
→ checkpoint
```

校验：

```text
compaction 后让模型基于 summary 复述当前状态
关键 artifacts 独立保存
重要决策带引用
```

**发散思考**

追问：

```text
什么时候触发 compaction？
如何处理被压缩掉的工具输出？
是否需要可回放 event log？
```

一句话：

> compaction 要保留 state，而不是保留对话；目标是让 agent 继续任务不丢关键约束。

---

## C9. 如何设计 agent 的 permission model？

**考察目的**

考你是否能让 agent 安全地做事。

**第一性原理**

权限按 side effect 风险分层。越不可逆、越外部化，越需要确认。

**回答**

权限层：

```text
read-only
write workspace
run local command
network access
modify external service
send email / calendar
deploy / payment / delete
```

策略：

```text
默认拒绝高风险
最小权限
human approval
dry run
审计日志
secret redaction
per-tool allowlist
```

**发散思考**

追问：

```text
rm -rf 如何拦截？
发邮件前如何确认？
用户说“你自己决定”是否等于授权？
```

一句话：

> permission model 的目标是让 agent 能行动，但不能越权或执行不可逆危险操作。

---

## C10. tool schema 写不好会造成什么问题？

**考察目的**

考你是否知道工具接口就是 agent-computer interface。

**第一性原理**

模型通过 schema 理解工具能力。schema 模糊会导致错误调用、遗漏参数、误解副作用。

**回答**

坏 schema 问题：

```text
1. 参数含义不清
2. 缺少约束
3. 输出不可解析
4. side effect 不明确
5. 错误信息不可诊断
6. 模型不知道何时使用
```

好 schema：

```text
明确输入
明确输出
明确失败模式
明确权限
给 examples
结构化返回
```

**发散思考**

追问：

```text
工具描述应该短还是长？
错误信息如何设计？
是否需要 tool eval？
```

一句话：

> tool schema 是 agent 的 API 文档；写不好，模型就会稳定地错误使用工具。

---

## C11. MCP 的价值是什么？风险是什么？

**考察目的**

考你是否理解 agent 工具生态和安全风险。

**第一性原理**

MCP 让 agent 以标准协议接入外部工具和数据源。标准化降低接入成本，也扩大攻击面。

**回答**

价值：

```text
统一工具协议
跨应用复用
降低集成成本
支持动态工具发现
```

风险：

```text
恶意工具
权限过大
prompt injection
secret 泄露
工具输出不可信
供应链攻击
```

防御：

```text
tool allowlist
权限分级
sandbox
审计
用户确认
签名/来源验证
```

**发散思考**

追问：

```text
MCP server 是否都可信？
如何限制工具访问文件系统？
工具输出能否作为指令？
```

一句话：

> MCP 的价值是工具标准化，风险是把 agent 的攻击面扩展到整个工具生态。

---

## C12. coding agent 为什么需要 sandbox 和 diff？

**考察目的**

考你是否理解代码修改必须可验证、可回滚。

**第一性原理**

coding agent 会读写文件、运行命令。错误操作可能破坏工作区或泄露信息。

**回答**

sandbox 提供：

```text
隔离文件系统
限制网络
限制命令
保护 secret
可清理环境
```

diff 提供：

```text
可审查修改
可回滚
可测试前后差异
支持 human review
```

**发散思考**

追问：

```text
agent 是否可以直接提交代码？
如何处理 destructive command？
测试失败后如何回滚？
```

一句话：

> sandbox 控制行动范围，diff 让修改可审查、可验证、可撤销。

---

## C13. subagent 什么时候有用，什么时候是噪声？

**考察目的**

考你是否能避免“多 agent 崇拜”。

**第一性原理**

subagent 有价值的条件是：

```text
任务可分解
子任务有清晰输入输出
并行收益大于协调成本
```

**回答**

有用场景：

```text
代码库搜索
独立实验分支
安全 reviewer
文档整理
多方案并行比较
权限隔离
```

噪声场景：

```text
任务本身简单
角色定义模糊
subagent 互相聊天
没有明确交付物
上下文传递成本过高
```

**发散思考**

追问：

```text
subagent 输出如何验证？
subagent 是否能调用高风险工具？
如何限制预算？
```

一句话：

> subagent 只有在能并行、能隔离、能产出可验证结果时才有价值。

---

## C14. 如何评估一个 agent 框架？

**考察目的**

考你是否能用工程指标评估 agent，而不是看 demo。

**第一性原理**

agent 是执行系统，评估要看成功率、成本、安全、恢复能力。

**回答**

指标：

```text
task success rate
cost per success
latency / time to completion
tool error recovery rate
unsafe action rate
human intervention rate
regression rate
context compaction failure rate
```

评测集：

```text
短任务
长任务
工具失败
prompt injection
代码修改
真实项目任务
权限边界测试
```

**发散思考**

追问：

```text
SWE-bench 是否足够？
如何做线上 shadow eval？
如何评估安全性？
```

一句话：

> agent eval 要评估长期闭环执行能力，而不是单轮回答质量。

---

## C15. SWE-bench 类任务和真实工程任务的差别是什么？

**考察目的**

考你是否知道 benchmark 和 production 有 gap。

**第一性原理**

benchmark 通常边界清楚、目标明确；真实工程任务上下文不完整、约束多、风险高。

**回答**

差异：

```text
真实任务有隐含需求
代码库更脏
测试不完整
需要和人沟通
有权限和安全问题
有部署风险
任务可能跨多天
```

SWE-bench 价值：

```text
可量化
可复现
适合比较 coding ability
```

不足：

```text
不能覆盖产品判断、长期状态、安全权限、多人协作
```

**发散思考**

追问：

```text
如何构造更真实 agent eval？
benchmark 高分是否代表生产可用？
```

一句话：

> SWE-bench 测代码修复能力，但真实 coding agent 还要处理上下文、沟通、权限、安全和长期状态。

---

## C16. agent 如何处理长任务中的失败恢复？

**考察目的**

考你是否理解 agent 不能只会一路执行，还要能从失败中恢复。

**第一性原理**

长任务一定会遇到工具失败、假设错误、测试失败、上下文丢失。系统必须可回滚、可诊断、可重试。

**回答**

机制：

```text
checkpoint
event log
workspace diff
test result history
retry policy
fallback model
human escalation
state summary
rollback
```

失败分类：

```text
工具失败
环境失败
模型误判
权限不足
外部服务失败
测试失败
```

**发散思考**

追问：

```text
什么时候自动重试，什么时候问人？
如何避免重复失败？
如何从 compaction 错误恢复？
```

一句话：

> 长任务 agent 需要像可靠分布式系统一样记录状态、支持回滚、分类失败并恢复。

---

## C17. agent 如何做可观测性和审计？

**考察目的**

考你是否理解 agent 行动必须可追踪，尤其涉及文件、邮件、部署、外部服务。

**第一性原理**

agent 每个动作都可能产生副作用。可观测性是 debugging，审计是责任边界。

**回答**

记录：

```text
model
prompt hash
tool calls
tool inputs/outputs
权限决策
latency
token/cost
文件 diff
外部 side effect
user approvals
final outcome
```

展示：

```text
timeline
diff view
tool trace
error trace
cost view
approval log
```

**发散思考**

追问：

```text
如何保护隐私同时记录日志？
secret 如何 redaction？
审计日志是否可篡改？
```

一句话：

> agent 的每个外部动作都应该可追踪、可解释、可回滚或可问责。

---

## C18. 什么时候应该让 agent 自动执行，什么时候必须 human-in-the-loop？

**考察目的**

考你是否能设计自动化边界。

**第一性原理**

决策依据是风险：

```text
可逆性
外部影响
金额/法律/隐私风险
用户意图明确度
```

**回答**

可自动：

```text
只读搜索
本地草稿
无副作用代码分析
运行测试
生成 diff
```

需要确认：

```text
删除文件
发送邮件
支付
部署
修改生产系统
泄露/访问敏感数据
执行高风险 shell 命令
```

**发散思考**

追问：

```text
用户一次性授权是否足够？
如何做 scoped approval？
如何防止 prompt injection 诱导高风险动作？
```

一句话：

> 可逆、低风险、局部动作可自动；不可逆、外部化、高风险动作必须 human-in-the-loop。

---

# 最后：这套题的统一面试答法

你可以把所有问题都归结到一个框架：

```text
1. 数据在哪里？
2. 数据要搬多少？
3. 是否能复用？
4. 是否能 overlap？
5. 计算单元需要什么粒度？
6. 同步在哪里发生？
7. 精度误差在哪里累积？
8. 多卡通信是否可隐藏？
9. serving 调度是否匹配 workload？
10. agent 行动是否可验证、可回滚、可控？
```

面对任何一道题，先说：

```text
这个问题表面问 X，
本质考的是 Y，
从第一性原理看瓶颈是 Z，
所以机制 A/B/C 是为了解决这个瓶颈，
代价是 D，
适用场景是 E。
```

这就是 AI Infra 面试里最强的表达方式：不是背名词，而是从数据流、瓶颈和系统 tradeoff 推导答案。

[1]: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html?utm_source=chatgpt.com "1. NVIDIA Ampere GPU Architecture Tuning Guide"
[2]: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/?utm_source=chatgpt.com "NVIDIA Hopper Architecture In-Depth"
[3]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell.html?utm_source=chatgpt.com "Blackwell Specific — NVIDIA CUTLASS Documentation"
[4]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?utm_source=chatgpt.com "2. Profiling Guide — NsightCompute 13.2 documentation"
[5]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/?utm_source=chatgpt.com "CUDA C++ Programming Guide (Legacy)"
[6]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html?utm_source=chatgpt.com "4.13. L2 Cache Control — CUDA Programming Guide"
[7]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html?utm_source=chatgpt.com "Efficient GEMM in CUDA"
[8]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[9]: https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
[10]: https://arxiv.org/abs/2407.08608?utm_source=chatgpt.com "[2407.08608] FlashAttention-3: Fast and Accurate Attention ..."
[11]: https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/?utm_source=chatgpt.com "Accelerating HPC Applications with NVIDIA Nsight ..."
[12]: https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html?utm_source=chatgpt.com "Contents — PTX ISA 9.2 documentation"
[13]: https://forums.developer.nvidia.com/t/what-is-the-difference-cuda-limit-persisting-l2cachesize-access-policy-max-windowsize-persisting-l2cache-maxsize/275542?utm_source=chatgpt.com "cuda Limit Persisting L2CacheSize，access Policy Max ..."
[14]: https://arxiv.org/abs/2309.14509?utm_source=chatgpt.com "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models"
[15]: https://arxiv.org/abs/2310.01889?utm_source=chatgpt.com "Ring Attention with Blockwise Transformers for Near-Infinite Context"
[16]: https://arxiv.org/abs/2211.10438?utm_source=chatgpt.com "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
[17]: https://arxiv.org/abs/2306.00978?utm_source=chatgpt.com "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
[18]: https://arxiv.org/abs/2405.04532?utm_source=chatgpt.com "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving"
[19]: https://arxiv.org/abs/2405.04434?utm_source=chatgpt.com "DeepSeek-V2: A Strong, Economical, and Efficient Mixture ..."
[20]: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin?utm_source=chatgpt.com "DistServe: Disaggregating Prefill and Decoding for ..."
[21]: https://www.usenix.org/conference/osdi22/presentation/yu?utm_source=chatgpt.com "Orca: A Distributed Serving System for Transformer-Based ..."
[22]: https://docs.sglang.ai/basic_usage/openai_api_completions.html?utm_source=chatgpt.com "OpenAI APIs - Completions"
[23]: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/sgl_model_gateway.md?utm_source=chatgpt.com "sglang/docs/advanced_features/sgl_model_gateway.md at ..."
[24]: https://github.com/sgl-project/sglang/issues/22558?utm_source=chatgpt.com "[RFC] Native gRPC Server for SGLang in Rust #22558"
[25]: https://arxiv.org/abs/2309.06180?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model ..."
[26]: https://arxiv.org/abs/2506.02523?utm_source=chatgpt.com "Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention"
[27]: https://docs.vllm.ai/en/v0.6.1/automatic_prefix_caching/details.html?utm_source=chatgpt.com "Implementation — vLLM"
[28]: https://lucumr.pocoo.org/2026/1/31/pi/?utm_source=chatgpt.com "Pi: The Minimal Agent Within OpenClaw"
[29]: https://arxiv.org/html/2604.14228v1?utm_source=chatgpt.com "The Design Space of Today's and Future AI Agent Systems"
