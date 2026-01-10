- [[#0. 引言]]
- [[#1 存储分析]]
- [[#2 Warp 分析]]
    - [[#2.1 Load-Warp]]
    - [[#2.2 MMA-Warp]]
    - [[#2.3 Softmax-Warp]]
- [[#3. 流水线分析]]

# 0. 引言

在即将推出的 FA4 中，FA 的作者在其 main 分支上，基于 cutlass 的示例代码，实现了一版高性能的 FMHA 算子。代码可见[具体链接](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py)。这里我们带着上文的理解，开始对比 FA4 的实现中，做了哪些改动，实现了哪些新的特性？概括地说，主要包含了以下的改动和特性。

> [!important]
> 
> 1. **功能覆盖对比**
>     
>     - CUTLASS：MHA + causal 掩码 + 部分变长
>     
>     - FA 新增：GQA/MQA、变长序列、分页 KV、滑动窗口、learnable sink、score_mod 钩子、LSE 输出
>     
> 
> 1. **流水线与同步**
>     
>     - CUTLASS：pipeline 句柄驱动，教学直观
>     
>     - FA：手写 mbarrier 布局，支持 overlap_sO_sQ、uneven_kv_smem 等优化
>     
> 
> 1. **调度器**
>     
>     - CUTLASS：静态调度器，持久化/非持久化
>     
>     - FA：多种调度器（常规/持久化/LPT/变长），注入更多上下文
>     
> 
> 1. **张量布局与读写**
>     
>     - 两者均用 TMA(GMEM→SMEM)、TMEM、tcgen05 MMA
>     
>     - FA：O 写回支持 TMA 与向量化拷贝双路径，V 预转置，适配 varlen/page_table；gemm 支持源操作数从 tmem 读取
>     

总得来说，由于 FA 是一个完整的库，所以不仅支持了 MHA，还支持了 **GQA、MQA** 两种模式，还有 Deepseek 需要的变体，以及一些推理优化的特性，比如支持 `page-attention`，滑动窗口、`learnable_sink`等特性。下面会以 DeepSeek-V3 系列的 MLA 的计算逻辑进行分析。并按照**存和算**两大部分进行拆解，并最后将其合在一起，看看是如何进行排布的。

# 1 存储分析

第一步，我们先不看细节性的代码，先看看代码里，是如何分配 SMEM，TMEM，RMEM 的？

**SMEM：**

```Python
@cute.struct
class SharedStorage:
# m_barriers for pipelines
mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total] # 15 * 8B = 120B
# Tmem holding buffer
tmem_holding_buf: Int32
# Smem tensors
# store row max and row sum
sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2] # row_max 和 row_sum ，128 * 2 * 2 * 4B = 512B = 0.5 KB
sO: cute.struct.Align[
cute.struct.MemRange[self.o_dtype, sO_size], # 大小为 0 ，O 放在 TMEM 上
self.buffer_align_bytes, # 1024b = 128B
]
sQ: cute.struct.Align[
cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)], # 128 * 192  * 2B * 2 = 96 KB
self.buffer_align_bytes,
]
sK: cute.struct.Align[
# cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)], # 128 * 192 * 2B * 2 + 128 * 128 * 2B = 128 KB
self.buffer_align_bytes,
]
```

可以看到，SMEM 已经用的比较满了，共计用了 224.5 KB 的大小，很接近 228KB 的上限了。这也算是一个 SMEM 瓶颈。SMEM主要包含：两片 Q 的切片，三份 KV 的切片（因为大小受限，这里是 2 块 192 \* 128, 一块 128 \* 128）。且需要注意，这里 Large+Small+Large 的拆分，中间的 Small并不是一成不变，因为不是一直都是 Large+Small+Large 的排列，而是会随着流水线的进行，变成 Small+Large+Small 的组合。下面 load 阶段会着重分析。

**TMEM**：

```Python
self.tmem_s_offset = [0, self.n_block_size] # e.g., 0, 128， S 分两块，每块 128 列
self.tmem_o_offset = [
self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
for i in range(self.q_stage)
] # e.g., 256, 384，O 也分两块，每块 128 列
self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded # 计算 O的最后一个分片的尾部
assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
self.tmem_s_to_p_offset = self.n_block_size // 2
self.tmem_p_offset = [
self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
] # 0, 128
```

可以看到，这里 TMEM 的分配就回到了 C 风格，主要还是 TMEM 是一个新出的硬件，NV 官方支持不够，需要用手动偏移值来区分分块。可以看到，这里 TMEM 也用满了。S 占两片\[128 \* 128], O 占两片 \[128 * head_dim_v_padded]。在 Deepseek 情况下，这里已经占满了全部 512 列的存储，没有任何空间浪费。同时这里的 P 会复用 S 的空间，且复用是按 S 的一半来复用的。

**RMEM：**

```Python
if self.head_dim_padded < 96:
	self.num_regs_softmax = 200
	self.num_regs_correction = 64
	self.num_regs_other = 48
else:
	# self.num_regs_softmax = 192 if self.is_causal or self.is_local else 184
	self.num_regs_softmax = 200
	# self.num_regs_softmax = 176
	# self.num_regs_correction = 96
	# self.num_regs_correction = 80
	# self.num_regs_correction = 64 if self.is_causal or self.is_local else 80
	self.num_regs_correction = 64
	# self.num_regs_other = 32
	# self.num_regs_other = 64
	# self.num_regs_other = 80
	self.num_regs_other = 48
	# self.num_regs_other = 96 if self.is_causal or self.is_local else 80
	# self.num_regs_other = 64 if self.is_causal or self.is_local else 80
	self.num_regs_empty = 24
```

可以看到，在 Deepseek 例子下，会走 else 分支，softmax 会使用 200 个寄存器，correction 使用 64 个，other 占用 48，empty 占用 24。比如 Softmax，这里分配的是一个线程占有 200 个寄存器，这 200 是怎么来的呢？前文说到了，一个 softmax 处理一个中间结果 S，S 的shape 是 \[128, 128]， 那么这里 4 个warp 处理，正好一个线程处理一行，也就是 128 个数，占据 128 个寄存器；同时，P 的写回是S 的一半，所以寄存器需要 64 个。加起来是 192，同时还需要一些辅助中间值，比如 row_max 等，故而申请到 **200** 个。同时比如 correction 这个 WG，启动 4 个 warp，来处理一个 O 的结果，所以一个线程处理 128 个数，且 O 是 BF16 格式，那么就使用 **64** 个寄存器就 OK。同样地，我们来分析 others，主要压力来自于 MMA，每次 UMMA，处理一个 \[128,32] * \[32, 128]的矩阵乘，所以分配到每个线程来看，矩阵 A 需要16 个，B 需要 16 个，总体需要 32 个寄存器，加上一些其他变量，**48** 是比较安全的分配值。一点疑惑：empty 为啥要分 **24**，从代码上没看出来为啥。

# 2 Warp 分析

结合上文的存储分析后，我们对数据流的来处和去处都有了一个比较清晰的认识了：

- 输入和输出：
    
    - Q：`Global ->(Load) SMEM ->(MMA) RMEM`
    
    - K: `GLobal ->(Load) SMEM ->(MMA) RMEM`
    
    - V: `Global ->(Load) SMEM ->(MMA) RMEM`
    
    - O: `TMEM ->(Correction) RMEM -> TMEM ->(Store) GLobal`
    

- 中间结果：
    
    - S: `TMEM ->(Softmax) RMEM`
    
    - P: `RMEM -> TMEM`
    

那么，其实 Warp 的划分，就是负责担任其中的”箭头“，将输入的数据，进行计算或者搬运，不管怎么说，都是将一个 shape 的数，存到指定目的下指定 shape。认识到这一点，对后面的 warp 划分就很好理解了。

### 2.1 Load-Warp

我们来看看 Load 这个 warp 是如何做的。首先，定义 Q 和 KV 的 state。

```Python
q_producer_phase = Int32(1) # Q只有两片，所以一个 int32 表示就行，后面跟 1 进行 XOR 做翻转
kv_producer_state = cutlass.pipeline.make_pipeline_state(
cutlass.pipeline.PipelineUserType.Producer, self.kv_stage # 调用cute的 Producer 的流水线生成三个 stage
)
```

然后，开始计算全局内存中，这一次 tile 的部分的位置。

```Python
m_block, head_idx, batch_idx = work_tile.tile_idx # 分 tile 的维度，每个 CTA 处理一个 bs，一个 head，一个 m_block(128)的输入数据
seqlen = SeqlenInfoCls(batch_idx)
mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx] #  根据 bs 来筛选，得到[S_q,head_dim,head_num],然后根据 head_idx 取出[S_q,head_dim,1] 注意 Q和O 的 shape 换过，变成了[S_q, head_dim, head_num, bs]
gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)) # 先选择mma_tiler_qk 的0 和 2 维度，得到[128,head_dim]的 shape，根据这个去切分 mQ_cur, mQ_cur 变成[S_q/128,1,head_num]的外围shape，再选择[:,0]这个坐标下的tensor（只是视角变换）,也就是拿到这一次该 load 进来的 Q 的分片大小和对应坐标。
head_idx_kv = (
head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx # ==1
)
if const_expr(mPageTable is None):
	if const_expr(not seqlen.has_cu_seqlens_k):
		mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)] \#最简单的情况，直接取前两个维度就行
	else:
	mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
	mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
	gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
	gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
```

之后，开始计算每个 thread 的视角下的分片，得到 MMA 函数需要的各种参数

```Python
tSgQ = thr_mma_qk.partition_A(gQ)
tSgK = thr_mma_qk.partition_B(gK)
tOgV = thr_mma_pv.partition_B(gV)
load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ # tma_atom_Q是descriptor，0代表不广播，cute.make_layout(1)代表 CAT 的 shape，这里只有一个 CTA，所以简单 make1 就行；后面分别是目的地址和原地址，通过封装的这个 tma_get_copy_fn 得到一个 tma 的函数
)
tKsK, tKgK = cpasync.tma_partition(
tma_atom_K, # [MMA,MMA_K,MMA_D,PIPE]
0, # no multicast
cute.make_layout(1),
cute.group_modes(sK, 0, 3), # 把前三个维度合并，变成[_,stage]
cute.group_modes(tSgK, 0, 3), # 把前三个维度合并，变成[_,block]
)
tVsV, tVgV = cpasync.tma_partition(
tma_atom_V,
0, # no multicast
cute.make_layout(1),
cute.group_modes(sV, 0, 3),
cute.group_modes(tOgV, 0, 3),
)
```

然后，通过复用一些本次 tile 固定的参数，来调用实际的 copy 函数，并传递必要的参数。Q 的加载通过 mbar_load_q_empty/full 两组 barrier 和一个 phase（相位）在生产者（load warp）与消费者（MMA warp）之间握手；K/V 由于每次传输字节不同，生产者侧不使用 pipeline_kv 的 producer API，而是直接 mbarrier：每次拷贝前 mbarrier_wait(empty)，后 arrive_and_expect_tx(full, bytes)，消费者侧统一用 pipeline_kv.consumer_* 对应的 barrier 等待与释放。

```Python
def load_Q(
self,
load_Q_fn: Callable,
mbar_full_ptr: cute.Pointer,
mbar_empty_ptr: cute.Pointer,
block: Int32,
stage: int,
phase: Int32,
):
	cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
	with cute.arch.elect_one():
		cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, self.tma_copy_bytes["Q"])
	load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)

@cute.jit
def load_KV(
self,
tma_atom: cute.CopyAtom,
tXgX: cute.Tensor,
tXsX: cute.Tensor,
mbar_full_ptr: cute.Pointer,
mbar_empty_ptr: cute.Pointer,
block: Int32,
producer_state: cutlass.pipeline.PipelineState,
K_or_V: Literal["K", "V"],
page_idx: Optional[Int32] = None,
):
	assert K_or_V in ("K", "V")
	stage, phase = producer_state.index, producer_state.phase
	cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
	if const_expr(K_or_V == "K" and self.uneven_kv_smem):
		# Before this round, the smem location was occupied by V, which is smaller than
		# K. So we need to wait for the stage after that (stage 1) to be empty as well.
		if stage == 0:
			cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)
	with cute.arch.elect_one():
		cute.arch.mbarrier_arrive_and_expect_tx(
			mbar_full_ptr + stage, self.tma_copy_bytes[K_or_V]
		)
	tXsX_cur = tXsX[None, stage]
	if const_expr(self.uneven_kv_smem):
	# Since this is the producer_state, the phase starts at 1, so we have to invert it
		tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1) # 中间的小块会来回覆盖, 第一轮中，phase==1，这里传入 0，代表 stage1 往右移动
		# Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
		# phase =1 代表是生产者的阶段，=0 是消费者的阶段
	tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
	cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)

@cute.jit
def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
	if const_expr(self.uneven_kv_smem):
		# smem layout is [smem_large, smem_small, smem_large], and the current stride is
		# (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
		# phase == 0, or left by offset if phase == 1.
		offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase) # 针对stage1这块，phase=0，右移 32（放 V），phase=1，左移 32（放 K）
		return cute.make_tensor(sX.iterator + offset, sX.layout)
	else:
		return sX

load_Q = partial( # 通过偏函数绑定部分重复参数，简化使用
self.load_Q,
load_Q_fn,
mbar_ptr + self.mbar_load_q_full_offset,
mbar_ptr + self.mbar_load_q_empty_offset,
phase=q_producer_phase,
)

# We have to use mbarrier directly in the load for KV instead of replying on
# pipeline_kv, because we could have different number of TMA bytes for K and V
load_K = partial(
self.load_KV,
tma_atom_K,
tKgK,
tKsK,
mbar_ptr + self.mbar_load_kv_full_offset,
mbar_ptr + self.mbar_load_kv_empty_offset,
K_or_V="K",
)

load_V = partial(
self.load_KV,
tma_atom_V,
tVgV,
tVsV,
mbar_ptr + self.mbar_load_kv_full_offset,
mbar_ptr + self.mbar_load_kv_empty_offset,
K_or_V="V",
)
```

最后，利用前文的一堆准备工作做好的简化的 load 函数，开始执行实际的 load 动作。

```Python
n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block) # 计算本次m_block 对应的 N 方向的有效范围，需要考虑变长/滑动窗口/因果等
load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
page_idx = (
	mPageTable[batch_idx, n_block_max - 1]
	if const_expr(mPageTable is not None)
	else None
)
load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # K0，最右侧
kv_producer_state.advance()
if const_expr(self.q_stage == 2):
	load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1
q_producer_phase ^= 1 # 本轮的 Q 的缓冲区全部用完了
load_V(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # V0，最右侧
kv_producer_state.advance()
for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
	n_block = n_block_max - 2 - i
	page_idx = (
		mPageTable[batch_idx, n_block] if const_expr(mPageTable is not None) else None
	)
	# if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("n_block = {}, page_idx = {}", n_block, page_idx)
	load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Ki
	kv_producer_state.advance()
	load_V(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Vi
	kv_producer_state.advance()
tile_scheduler.prefetch_next_work()
tile_scheduler.advance_to_next_work()
work_tile = tile_scheduler.get_current_work()
```

这里我们来模拟一下程序执行的step，来看看加载过程. Q0和 Q1 只需要两次加载即可，后续计算只用读取这里就行；然后 KV 会不断在在 3 块 KVstage 之间轮转。

```Shell
load_Q0; q_stage=0  | [128,192] | [Q0,--,--,-,--]
load_K0; kv_stage=0 | [128,192] | [Q0,--,K0,-,--]
load_Q1; q_stage=1  | [128,192] | [Q0,Q1,K0,-,--]
q_phase = 1^1 = 0
load_V0; kv_stage=1 | [128,128] | [Q0,Q1,K0,V0,--]
load_K1; kv_stage=2 | [128,192] | [Q0,Q1,K0,V0,K1] # 这里是 Large+Small+Large
kv_phase = 1^1 = 0
load_V1; kv_stage=0 | [128,128] | [Q0,Q1,V1,V0,K1]
load_K2; kv_stage=1 | [128,192] | [Q0,Q1,V1,K2,K1]
load_V2; kv_stage=2 | [128,128] | [Q0,Q1,V1,K2,V2] # 这里是 Small+Large+Small
kv_phse = 0^1 = 1
```

### 2.2 MMA-Warp

盘算完了存储，我们就该来看看另一个重头戏--计算过程了。这里 FA4 的作者也是代码写的非常漂亮，该复用的地方都抽出来单独放。  
首先，mma warp 会做好一些预备工作，比如准备好输入的寄存器视角，以及用到的屏障的准备。

```Python
tSrQ = tiled_mma_qk.make_fragment_A(sQ) # 对 smem 上的 Q 切出寄存器视角
tSrK = tiled_mma_qk.make_fragment_B(sK) # 对 smem上的K切出寄存器视角
tOrV = tiled_mma_pv.make_fragment_B(sV) # 对 smem 上的 O 切出寄存器视角
if const_expr(self.q_stage == 2):
	tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1]) # Q分两片，这里区分开
else:
	tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 0])

qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

gemm_Si = [
	partial(
		sm100_utils.gemm_ptx_partial, # 这里调用一个PTX封装的gemm函数
		qk_mma_op,
		self.tmem_s_offset[stage],
		tSrQs[stage],
		sA=sQ[None, None, None, stage],
		zero_init=True,
	)
	for stage in range(2)
]
gemm_Pi = [
	partial(
		sm100_utils.gemm_ptx_partial,
		pv_mma_op,
		self.tmem_o_offset[stage if self.q_stage == 2 else 0],
		tOrPs[stage],
		sA=None,
	)
	for stage in range(2)
]

mma_q_consumer_phase = Int32(0) # 同Load-Warp一样，这里的q的phase 手动维护就行
mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
	cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage # kv的屏障，用cutedsl官方的即可
)
P_full_O_rescaled_phase = Int32(0) # 整个P都全部缩放后的phase，手动维护
```

接下来，让我们来看看上面的封装出来的`gemm_ptx_partial`，是什么语义？在blackwell_helpers 这个文件中，作者列出了一些 asm 手搓的包装好的函数，我们这里就先大概看一下，这些 ptx 都干了什么？总体来讲，这里封装主要是为了：1.提升性能，更细粒度地插入控制流，比如是否要进行累加等，不用在更上层去控制了；2.实现从 tmem 读取源操作数 A，且实现 3/4 的等待和释放，如此细粒度的屏障控制，显然简单的 cute.gemm 是无法胜任的。

```Python
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    # sA_offset: Int32 = 0,
    # acc_offset: Int32 = 0,
    tA_addr: Optional[Int32] = None,
) -> None:
    # acc_tmem_addr += acc_offset
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    if const_expr(not is_ts):
        sA_swizzle = parse_swizzle_from_pointer(sA.iterator)
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K
                if const_expr(op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
                else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = parse_swizzle_from_pointer(sB.iterator)
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if const_expr(not is_ts)
        else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))]
    offset_a_diff = [offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
        # ) + sA_offset
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(), \#a矩阵的llvm 中的 ir 表示，直接
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\\n\\t"
            ".reg .pred leader_thread;\\n\\t" # 声明谓词寄存器，标记被选为 leader 的线程
            ".reg .pred p;\\n\\t" # 谓词寄存器，首步的累加开关
            ".reg .b32 idesc;\\n\\t" # 32bit的 mma描述符
            ".reg .b32 tmem_acc;\\n\\t" # 32bit的tmem的目标地址寄存器
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\\n\\t" # AB的 smem描述符的低 32 位
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\\n\\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\\n\\t" # AB的描述符的高 32 位，包括 swizzle
            ".reg .b64 smem_desc_a, smem_desc_b;\\n\\t" # 完整的 64bit 的描述符信息
            "elect.sync _|leader_thread, -1;\\n\\t" # 所有线程都参与选举
            f"mov.b32 idesc, {hex(idesc)};\\n\\t" # 加载mma的描述符
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\\n\\t"
            f"mov.b32 tmem_acc, $3;\\n\\t" # tmem的写入地址移入寄存器
            "mov.b32 smem_desc_a_lo_start, $0;\\n\\t" # 前32位A地址移入
            "mov.b32 smem_desc_b_lo_start, $1;\\n\\t" # 前32位B地址移入
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\\n\\t" # A的后32位高地址移入
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\\n\\t" # B的后32位高地址移入
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\\n\\t" # 拼接出完整的A的描述符
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\\n\\t" # 拼接出B的描述符
            "setp.ne.b32 p, $2, 0;\\n\\t" # 如果 not zero_init==true，则设置p寄存器为1，即开启累加
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\\n\\t" # 发射umma 指令，类型为 f16，且控制首步是否累加
            + "".join(
                (
                    # f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\\n\\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\\n\\t"
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\\n\\t" # 低位偏移
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\\n\\t" # 低位偏移
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\\n\\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\\n\\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\\n\\t" # 默认开启累加
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\\n",
            # "r,r,r",
            "r,r,r,r", # 用到了四个寄存器的约束
            has_side_effects=True, # 禁止编译器优化
            is_align_stack=False, # 不需要栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT, # 汇编风格指定为 unix 的 AT&T
        )
    else:
        # For TS gemm, somehow tCrA.iterator.toint() returns 0 no matter what, so we need to
        # explicitly pass in the tA_addr for correctness.
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            # Int32(cute.arch.make_warp_uniform(tCrA[None, None, 0].iterator.toint())).ir_value(),
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \\n\\t"
                "LAB_WAIT: \\n\\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \\n\\t"
                "@P1 bra DONE; \\n\\t" \#P1置位后才结束等待
                "bra     LAB_WAIT; \\n\\t"
                "DONE: \\n\\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            # [
            #     # acc.iterator.toint().ir_value(),
            #     Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
            #     Int32(smem_desc_start_b_lo).ir_value(),
            #     Int32(not zero_init).ir_value(),
            # ],
            input_args,
            "{\\n\\t"
            ".reg .pred leader_thread;\\n\\t"
            ".reg .pred p;\\n\\t"
            ".reg .b32 idesc;\\n\\t"
            ".reg .b32 tmem_acc;\\n\\t"
            ".reg .b32 tmem_a;\\n\\t"
            ".reg .b32 smem_desc_b_lo_start;\\n\\t"
            ".reg .b32 smem_desc_b_lo;\\n\\t"
            ".reg .b32 smem_desc_b_hi;\\n\\t"
            ".reg .b64 smem_desc_b;\\n\\t"
            "elect.sync _|leader_thread, -1;\\n\\t"
            f"mov.b32 idesc, {hex(idesc)};\\n\\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\\n\\t"
            f"mov.b32 tmem_acc, $3;\\n\\t"
            f"mov.b32 tmem_a, $0;\\n\\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\\n\\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\\n\\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\\n\\t"
            "setp.ne.b32 p, $2, 0;\\n\\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\\n\\t" # 同样首步是否累加需要读取前面的判断
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\\n\\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\\n\\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\\n\\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\\n\\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, 1;\\n\\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\\n\\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2])
                    if const_expr(mbar_ptr is None)
                    else cute.size(tCrA.shape[2]) // 4 * 3, # 如果有mbar，就先计算前3/4
                )
            )
            + mbar_wait_str # 等待屏障释放
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\\n\\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\\n\\t"
                        f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\\n\\t"
                    )
                    for k in range(cute.size(tCrA.shape[2]) // 4 * 3, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
```

之后，就是主要的循环部分的代码了。主要逻辑为：先用 K0 对 Q0/（Q1）做一轮 QK（写出 S0/S1 并通知 softmax），随后在“seqlen_kv”循环里交错执行 PV 与后续 QK：每次先消耗一个 V 做 P×V 累加 O，再消费一个新的 K 继续 Q×K 产出下一块 S。期间用多组 mbarrier 同步 S、P/O、Q 的生产/消费，并用 KV 管线消费接口推进 K/V 的到达与释放。最后释放 Q，处理最后一个 V 的 PV 并通知 O 完整，进入下个 tile。

> [!important]
> 
> 这里其实用到了很细粒度的同步，比如 P 和 O 的同步。
> 
> P 只需要获取前 3/4 部分，就可以开始计算了，不用等 softmax 全部算完写回，这一步在上面的 gemm_ptx 中有写到。我猜主要目的是作者发现，gemm 启动时，上一轮的 softmax 还没好，如果一直等全部数据写回，会增加流水线里面的空泡；那么考虑到部分数据的依赖性，可以在 $P(i-1)$ 完成3/4 时启动 gemm 进行计算，这样可以避免空泡。
> 
> O 的释放过程，被隐藏的数据依赖所保证了：开头的计算 P0V0 时，会要求 O0 的 scale 完成，但是在写回 O0 后，并没有手动地释放掉这个屏障，难道不会导致 correction 一直空等吗？原因是 softmax 的warp 中，会在收到这一轮的 S0 后，开始计算，并在迭代后通知 correction warp，开始读取 O0 并计算 scale。然后在下一轮 MMA 开始前，会等待 O0 的mbar，从而保证了数据之间依赖不错位，同时减少了一部分开销（**这部分开销有多大？可以做实验确认**）
> 
>   

```Python
while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if n_block_min < n_block_max:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_load_q_full_offset + stage, mma_q_consumer_phase
                )
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(
                            sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        )
                    gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    # 4. release S0 / S1
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                O_should_accumulate = False
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(2):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile has finished.
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                            P_full_O_rescaled_phase,
                        )
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                        # 4. release accumulated O0_partial / O1_partial
                        # Don't need to signal O_full to the correction warps anymore since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # with cute.arch.elect_one():
                        #     tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                        # 5. release V(i-1)
                        if const_expr(stage == 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrK[None, None, None, Ki_index], zero_init=True)
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                        # 3. release S0
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.q_stage):
                        tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(2):
                    # 2. acquire corrected Oi_partial and Pi
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase
                    )
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not O_should_accumulate,
                        mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warp, the softmax warp has just finished compute
                    # the row sum of the current tile. It does not guarantee that the 1st tile
                    # of the next work tile has been computed yet.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop
```

## 2.3 Softmax-Warp

我们先来看看比较细节地，执行每一步 softmax 计算的具体函数，softmax_step 定义如下所示。

```Python
@cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width # mma_tiler_qk 的 shape 是[m_block,n_block,head_dim],所以这里计算的是如果按 float32 来存取数据，n_block 实际是多少
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2])) # 准备S 的寄存器视图
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1))) # scale 是[m,1]的大小
        tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32))) # P是[m,n/32*v_dtype]大小

        # Wait for Si
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase) # 等待S的屏障shi'fang
        tSrS_t2r = cute.make_fragment(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype) # 
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                aux_tensors,
                fastdiv_mods,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            # tSrScale_r2t = cute.make_fragment(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
            # if thread_idx == 0: cute.printf("softmax acc_scale stage %d: %f, row_max = %f\n", stage, acc_scale, row_max)
        # Notify correction wg that row_max is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        # if thread_idx == 0 and stage == 0: cute.print_tensor(tSrS_t2r)
        # print(tSrS_t2r)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_wait(
                mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, s0_s1_sequence_phase
            )
        tSrP_r2t_f32 = cute.make_fragment(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),
            tSrS_t2r.layout,
        )
        # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            e2e=mask_fn is None and self.head_dim_padded <= 128,
            e2e_freq=self.e2e_freq,
        )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage) * 4)
        # print(tSrP_r2t_f32, tStP_r2t)
        # cute.copy(thr_tmem_store, tSrP_r2t_f32, tStP_r2t)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2]) // 4 * 3):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        for i in cutlass.range_constexpr(
            cute.size(tStP_r2t.shape[2]) // 4 * 3, cute.size(tStP_r2t.shape[2])
        ):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
        )
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.arch.exp2(acc_scale_)
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1
```

  

# 3. 流水线分析

![[FA4--pipeline.drawio.svg]]

根据前文的分析，我们从存储到计算，仔细分析了布局，计算过程等等细节，这里如上图所示，给出一个模拟的流水线过程。可以看到，作者在写 FA4 的代码时，应该当时充分考虑到了Tensor Core 的计算算力和 CUDA Core 以及带宽之间的不匹配的问题，因此选择使用 一个 warp group 专职加载，两个 warp group 专职计算 softmax，来尽力不要“拖累” Tensor Core上发射的 UMMA 指令。且图里也反应了 softmax 和 mma 之间的细微同步：softmax 写回 3/4 后就可以开始计算了，从而进一步缩小空泡，提高流水线效率。