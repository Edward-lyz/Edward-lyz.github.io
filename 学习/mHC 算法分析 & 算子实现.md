# 1. 背景

基于 Deepseek 新开的论文中所讲，mHC 改进了 HC 的训练过程，且提到新的模型使用了 mHC 进行训练，那么站在推理的角度，针对 mHC 使用的三个融合算子，有必要先用 CuTile 搓一个前向的 demo 出来。用于了解算法内容，以及为后续的调优提供一定的基础。
> [!info] arxiv.org  
>  
> [https://arxiv.org/pdf/2512.24880](https://arxiv.org/pdf/2512.24880)  
# 2. 计算过程分析

从算子角度看，mHC 的前向计算并不复杂，本质上是：
> [!important]
> 
> 用一个 GEMM + 若干约束函数，生成三组 mixing 系数，再用这些系数去重新组织 residual stream。
为了方便分析，先约定几个维度含义（与论文、实际实现保持一致）：

- **B**：`Batch × SeqLen`
- **n**：`mHC` 的扩张系数（通常为 4）
- **C**：原始 `hidden size`
- **X_t ∈ R^{B × nC}**：n 路 residual 拼接后的输入

---

### 2.1 一次 GEMM，生成全部未约束系数
mHC 首先通过一次大的线性变换，生成所有 mixing 的“原始参数”：
```Shell
[H̃̃_pre , H̃̃_post , H̃̃_res] = X_t · φ_t
```
其中：
- `φ_t ∈ R^{nC × (n^2 + 2n)}`
- 输出可以拆分为：
    - `H̃̃_pre ∈ R^{B × n}`
    - `H̃̃_post ∈ R^{B × n}`
    - `H̃̃_res ∈ R^{B × n^2}`（reshape 为 `B × n × n`）
**注意**：
这里并不是三次 GEMM，而是**一次 GEMM 输出一整个大向量**，只是逻辑上拆成三段。

---
### 2.2 RMSNorm 风格的 scale 计算
与此同时，对同一个输入 `X_t` 计算一个 RMSNorm 风格的缩放因子：
```Python
r = sqrt( sum(X_t^2) / (nC) )
```
这个 `r` 是：
- 每个 sample 一个 scalar（shape 为 `[B, 1]`）
- 后续会用于统一缩放三组系数

---
### 2.3 scale / alpha / bias 修正
三组系数会被统一做一次线性修正：
```Shell
H̃_pre  = (α_pre  * H̃̃_pre ) / r + b_pre
H̃_post = (α_post * H̃̃_post) / r + b_post
H̃_res  = (α_res  * H̃̃_res ) / r + b_res
```
其中：
- `α_pre / α_post / α_res` 是 scalar，可训练
- `b_pre, b_post, b_res` 是 bias
- 这一步仍然是逐元素计算，没有跨 sample / 跨通道依赖

---
### 2.4 对系数施加约束（mHC 的关键）
mHC 与 HC 的一个本质区别在于：**所有 mixing 系数都被限制在可控范围内**。
具体约束如下：
- **Pre mixing** → 保证非负
    ```Plain
    H_pre = sigmoid(H̃_pre)
    ```
- **Post mixing** → 非负，且上界为 2
    ```Plain
    H_post = 2 * sigmoid(H̃_post)
    ```
- **Residual mixing** → 投影为近似双随机矩阵（行和 = 1，列和 = 1）
    ```Plain
    H_res = Sinkhorn-Knopp(H̃_res)
    ```

`Sinkhorn-Knopp` 的过程是：
1. 以 `exp(H̃_res)` 为初始矩阵
2. 交替进行行归一化、列归一化
3. 重复固定次数（论文中约 20 次）

---
### 2.5 将系数真正应用到 residual stream
得到 `H_pre / H_post / H_res` 后，mHC 才真正开始“改写 residual”。
1. **Pre：n → 1**
```Shell
F_pre = Σ_i H_pre[i] · X_t[i]
```
得到一个标准的 `[B × C]` 输入，送入 Attention / FFN。
1. **中间层计算**
```Shell
F_out = F(F_pre)
```
这一步与 mHC 无关，通常是传统的 transformer 层
1. **Post & Res：重组 n 路 residual**
```Shell
X_res  = H_res · X_t
X_post = H_post^T · F_out
X_{t+1} = X_res + X_post
```
最终输出仍然是 `[B × nC]`，供下一层使用。

---
# 3. 算子融合 / 实现分析
如果严格按上述步骤逐算子实现，mHC 在工程上是**不可接受的**。
论文中提到的三类融合算子，实际上都是为了解决非常具体的性能问题。
### 3.1 Fusion 1：GEMM + RMSNorm reduce
**问题来源**
- `X_t ∈ R^{B × nC}` 非常大
- 既要做 GEMM，又要做 `sum(X_t^2)`
- 如果分成两个 kernel，`X_t` 会被完整读两遍
**融合思路**
在 GEMM 的同时，顺手做 reduce：
```Plain
for x in X_t:
    gemm_acc += x * φ
    norm_acc += x * x
```
一次访存，同时得到：
- `[H̃̃_pre, H̃̃_post, H̃̃_res]`
- `r`
**这是 mHC 最核心、也最必须的融合**。没有这一层，mHC 的内存开销会直接失控。
---
### 3.2 Fusion 2：scale / alpha / bias / sigmoid
这一阶段的特点是：
- 张量规模极小（`n^2 + 2n`）
- 计算完全是逐元素
- kernel launch 成本远大于计算本身
因此做法很直接：能在一个 kernel 里算完的，就不要拆。
常见实现方式是：
- **在 Fusion 1 的输出 buffer 上**
- 直接完成：
    - `/r`
    - `alpha`
    - `+ bias`
    - `sigmoid / 2*sigmoid`
这一类融合的逻辑是计算量很小，但是多次launch 带来的开销会超过算子本身耗时，融合是为了**避免 launch 变慢**。

---

### 3.3 Fusion 3：Sinkhorn-Knopp 单 kernel 迭代
`Sinkhorn-Knopp` 的 `n×n` 矩阵非常小（n 通常为 4），但需要多次强依赖迭代。
如果按 naive 方式实现：
```Python
for t in range(20):
    row_norm kernel
    col_norm kernel
```
会导致：
- 大量 kernel launch
- 大量 device-level 同步
**正确做法**是：
- 一个 kernel
- 将 n×n 矩阵放在寄存器 / shared memory
- 在 kernel 内部完成 20 次迭代
---
### 3.4 应用阶段融合：H_res / H_post + residual merge
这是最容易被忽略，但对带宽影响极大的部分。
如果分步实现：
1. `X_res = H_res · X`
2. `X_post = H_post^T · F_out`
3. `X_next = X_res + X_post`
会产生大量中间读写，内存流量非常不友好。论文中的做法是：

> [!important]
> 将 H_res 的应用、H_post 的应用、residual merge 合成一个 kernel.逻辑上等价于：

```Python
for c in C:
    for i in n:
        acc = Σ_j H_res[i,j] * X[j,c]
        acc += H_post[i] * F_out[c]
        X_next[i,c] = acc
```

这样可以做到：
- `X` 只读一次
- `X_next` 只写一次
- 所有中间结果不落地
从内存访问角度看，这是 mHC 能用于实际训练/推理的大优化点。
---
### 3.5 小结（实现视角）
从算子实现角度看，mHC 的前向可以抽象为：
1. **“大而贵”的融合 GEMM**
2. **“小而碎”的系数后处理 kernel**
3. **“多次迭代”的 Sinkhorn kernel**
4. **一个“极致带宽优化”的 residual 应用 kernel**
后续使用 CUDA 实现 demo，本质上就是围绕这四类 kernel 做拆解与验证
---
# 4. 具体代码实践
## 4.1 GEMM + RMS_Norm
这个算子改动不大，主要是将官方示例中的持久化调度的 GEMM 的 Kernel 抄过来基本不动，在每个CTA 计算对应分块的 GEMM 时，在计算的末尾，加入一段按行计算 RMS 的结果即可。
```Python

for k_tile in range(k_tiles):
	# Compute MMA
	# ···
	if bid_n == 0:
		a_fp32 = ct.astype(a, ct.float32)
		rms_acc = rms_acc + ct.sum(a_fp32 * a_fp32, axis=1, keepdims=False)
if bid_n == 0:
	denom = ct.full((TILE_SIZE_M,), K * 1.0, dtype=ct.float32)
	mean = ct.truediv(rms_acc, denom)
	rstd = ct.rsqrt(mean)
	ones_row = ct.full((TILE_SIZE_M,), 1.0, dtype=ct.float32)
	r = ct.truediv(ones_row, rstd)
	r = ct.reshape(r, (TILE_SIZE_M, 1))
	r = ct.astype(r, R.dtype)
	ct.store(R, index=(bid_m, 0), tile=r)
```
## 4.2 Fused_Sigmoid
这部分代码也比较简单, 主要是 IO 瓶颈，融合后计算量也没有增加多少。
```Python
@ct.kernel
def mhc_scale_bias_sigmoid_kernel(
Y,
R,
n: int,
alpha_pre: float,
alpha_post: float,
alpha_res: float,
Bias,
TILE_SIZE_N: ConstInt,
):
"""Fused scale/bias/sigmoid kernel for mHC (in-place on Y)."""
row = ct.bid(0)
col = ct.bid(1)
​
offsets = ct.arange(TILE_SIZE_N, dtype=ct.int32)
col_ids = col * TILE_SIZE_N + offsets
​
y = ct.load(Y, index=(row, col), shape=(1, TILE_SIZE_N), padding_mode=ct.PaddingMode.ZERO)
bias = ct.load(Bias, index=(col,), shape=(TILE_SIZE_N,), padding_mode=ct.PaddingMode.ZERO)
bias = ct.reshape(bias, (1, TILE_SIZE_N))
r = ct.load(R, index=(row, 0), shape=(1, 1), padding_mode=ct.PaddingMode.ZERO)
r = ct.reshape(r, (1, 1))
​
one = ct.full((TILE_SIZE_N,), 1.0, dtype=ct.float32)
zero = ct.full((TILE_SIZE_N,), 0.0, dtype=ct.float32)
mask_pre = ct.where(ct.less(col_ids, n), one, zero)
mask_post = ct.where(ct.less(col_ids, 2 * n), one, zero)
mask_post = mask_post - mask_pre
mask_res = one - mask_pre - mask_post
​
scale = alpha_pre * mask_pre + alpha_post * mask_post + alpha_res * mask_res
scale = ct.reshape(scale, (1, TILE_SIZE_N))
​
y_fp32 = ct.astype(y, ct.float32)
bias_fp32 = ct.astype(bias, ct.float32)
linear = ct.truediv(y_fp32 * scale, r) + bias_fp32
sigmoid_linear = _sigmoid(linear)
two_sigmoid = sigmoid_linear * 2.0
​
mask_pre = ct.reshape(mask_pre, (1, TILE_SIZE_N))
mask_post = ct.reshape(mask_post, (1, TILE_SIZE_N))
mask_res = ct.reshape(mask_res, (1, TILE_SIZE_N))
​
out = linear * mask_res + sigmoid_linear * mask_pre + two_sigmoid * mask_post
out = ct.astype(out, Y.dtype)
ct.store(Y, index=(row, col), tile=out)
​
```
## 4.3 迭代 SkinHorn 
  其实也比较简单，主要要注意的点就是避免来回读写, 做到原位读写
```Python
@ct.kernel
def mhc_sinkhorn_kernel(
Y,
n: ct.Constant[int],
):
"""Sinkhorn-Knopp normalization for residual block (in-place on Y)."""
row = ct.bid(0)
total = n * n
offsets = ct.arange(total, dtype=ct.int32)
offsets = offsets + 2 * n
​
mat = ct.gather(Y, (row, offsets), latency=1)
mat = ct.reshape(mat, (n, n))
mat = ct.astype(mat, ct.float32)
mat = ct.exp(mat)
​
for _ in range(20):
	row_sum = ct.sum(mat, axis=1, keepdims=True)
	mat = ct.truediv(mat, row_sum)
	col_sum = ct.sum(mat, axis=0, keepdims=True)
	mat = ct.truediv(mat, col_sum)
​
mat = ct.reshape(mat, (total,))
mat = ct.astype(mat, Y.dtype)
ct.scatter(Y, (row, offsets), mat, latency=1)
```
## 4.4 融合apply 
```Python
@ct.kernel
def mhc_apply_residual_kernel(
X,
F_out,
Y,
Out,
C: int,
n: ct.Constant[int],
TILE_SIZE_C: ConstInt,
):
"""Apply H_res and H_post to residual stream (in-place on Out)."""
row = ct.bid(0)
c_tile = ct.bid(1)
c_tiles = ct.cdiv(C, TILE_SIZE_C)
​
f_tile = ct.load(
F_out,
index=(row, c_tile),
shape=(1, TILE_SIZE_C),
padding_mode=ct.PaddingMode.ZERO,
)
f_tile = ct.astype(f_tile, ct.float32)
​
for i in range(n):
	acc = ct.full((1, TILE_SIZE_C), 0.0, dtype=ct.float32)
	for j in range(n):
		hij_offset = 2 * n + i * n + j
		hij_idx = ct.full((1,), hij_offset, dtype=ct.int32)
		hij = ct.gather(Y, (row, hij_idx), latency=1)
		hij = ct.reshape(hij, (1, 1))
		hij = ct.astype(hij, ct.float32)
		​
		x_tile = ct.load(
		X,
		index=(row, j * c_tiles + c_tile),
		shape=(1, TILE_SIZE_C),
		padding_mode=ct.PaddingMode.ZERO,
		)
		x_tile = ct.astype(x_tile, ct.float32)
		acc = acc + x_tile * hij
​
	hpost_offset = n + i
	hpost_idx = ct.full((1,), hpost_offset, dtype=ct.int32)
	hpost = ct.gather(Y, (row, hpost_idx), latency=1)
	hpost = ct.reshape(hpost, (1, 1))
	hpost = ct.astype(hpost, ct.float32)
	acc = acc + f_tile * hpost
	acc = ct.astype(acc, Out.dtype)
	ct.store(Out, index=(row, i * c_tiles + c_tile), tile=acc)
​
​
```
## 性能表现
由于上面的Cutile 实现基本都是根据算子语义直接翻译过来写的，没有根据输入的数据规模进行调优，我们假设 DS 的新模型在前向计算时，保持 seq_len 等于 2048，bs 等于 4，那么 X 的第一个维度就是 8192，同时假设 hidden_states 的维度依然是 7168，那么 X 的第二个维度就是 n * 7168, 根据论文所言，n 一般等于 4，那么就是28672. 
我们只考虑FP8 下的性能表现作为参照，先给出一版初步实现的性能数据：
```Shell
[bench] mhc_scale_bias_sigmoid M=8192 backend=cutile dtype=torch.float8_e4m3fn ms=0.0075
mhc-scale-bias-sigmoid-performance-float8_e4m3fn-GBps:
        M     CuTile
0  8192.0  56.891691
[bench] mhc_apply_residual M=8192 backend=cutile dtype=torch.float8_e4m3fn ms=0.3228
mhc-apply-residual-performance-float8_e4m3fn-GBps:
        M       CuTile
0  8192.0  1638.022544
[bench] mhc_sinkhorn M=8192 backend=cutile dtype=torch.float8_e4m3fn ms=0.0877
mhc-sinkhorn-performance-float8_e4m3fn-TFLOPS:
        M    CuTile
0  8192.0  0.044838
[bench] mhc_gemm_rmsnorm M=8192 backend=cutile dtype=torch.float8_e4m3fn ms=0.6534
mhc-gemm-rmsnorm-performance-float8_e4m3fn-TFLOPS:
        M     CuTile
0  8192.0  17.973055
```
综合来看，总共四个融合算子，其中`mhc-scale-bias-sigmoid`这个算子表现符合预期，耗时 `7.5 us`，可以给到优秀，性能能和 CUDA 手搓基本持平；`mhc_apply_residual` 算子不尽人意，耗时过长，从实现的带宽上也能看出性能，可以给到差评；`mhc_sinkhorn` 算子也如此，不过是 20 次的归一化迭代而已，实际耗时 87us，给到下等实现的评价；至于计算的重头戏，`gemm-rms` 算子，由于任务分配还是按照切分输出来划分，在 N 很小而 K 很大时，实际就会比较吃亏，性能严重受损。耗时几乎是不可接受的。
# 5. 性能优化

在论文/代码的典型设定里：
- `n=4`
- `M = Batch × SeqLen = 8192`
- `C=7168`，所以 `K=nC=28672`
- `N=n^2+2n=24`（只有 24 个输出通道）

也就是说：`K` 很大、`N` 极小，而且后续的 `Sinkhorn / apply` 都是“小矩阵 + 大吞吐”的典型组合。为了避免性能受限，我们应该避免：
1) 并行度不够（`GEMM` 只开得出很少的 CTA）
2) 中间结果落地导致的额外带宽（`gather/scatter`、多次读写 `X/Y`）
下面所有优化，基本都围绕这两个点做取舍。这里先放出我们的优化后的 bench 性能数据（X 是 BF16，W 是 TF32，其余输入是 FP32 格式），以贴近真实的模型使用情况。
```Shell
mhc-split-gemm-rms-performance-bfloat16-GBps:
        M      CuTile     PyTorch     DeepGemm
0  8192.0  5061.57394  313.423308  5064.876475
mhc-gemm-rms-scale-performance-bfloat16-GBps:
        M       CuTile     PyTorch
0  8192.0  4513.131525  388.723762
mhc-sinkhorn-performance-bfloat16-GBps:
        M     CuTile   PyTorch
0  8192.0  19.152307  2.584275
mhc-apply-residual-performance-bfloat16-GBps:
        M       CuTile     PyTorch
0  8192.0  6602.945269  337.325322
```

## 5.1 `gemm-rms`：针对 `N` 极小的 GEMM 重新做并行度
**问题来源**
- `N=24` 导致沿 `N` 维能切出来的 tile 非常少（`TILE_SIZE_N` 再小也就 1~2 个 tile）
- `K=28672` 又很大，单个 CTA 的工作量很重，但 CTA 数量很少 → SM 吃不满

**优化思路**
- [x] 切换成 split-K 算法：把并行度从 `M×N` 拓展到 `M×N×SPLIT_K`，每个 CTA 只算一段 `K`，最后再做一次归并
- [x] host 侧传入 `w_nt`（`N×K`）避免“在线转置”：kernel 内只做 tile 级别的寄存器转置，避免全量转置/非连续读
- [x] swizzle 任务映射：用 `_compute_bid(..., GROUP_SIZE_M)` 改写 `tile_id → (bid_m, bid_n)` 的映射，减少 L2 partition camping（同一时刻过多 CTA 打到相邻地址）
- [x] 去除 warp-divergency：避免在 Kernel 里写if-else

**对应实现**
```Python
def mhc_split_gemm_rms_kernel(
    X,
    W,
    Y_acc,
    R_acc,
    M: int,
    N: int,
    K: int,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_K: ConstInt,
    SPLIT_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
    """Split-K fused GEMM + RMS compute kernel for mHC.
    Key optimization: All blocks compute RMS to avoid wasting registers.
    Each block computes partial RMS for its K-tile range, which are later
    summed in the finalize kernel.
    """
    tile_id = ct.bid(0)
    bid_k = ct.bid(1)
    zero_pad = ct.PaddingMode.ZERO
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    bid_m, bid_n = _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M)
    k_tiles = ct.cdiv(K, TILE_SIZE_K)
    k_tiles_per_split = ct.cdiv(k_tiles, SPLIT_K)
    k_tile_start = bid_k * k_tiles_per_split
    k_tile_end = ct.minimum(k_tile_start + k_tiles_per_split, k_tiles)
    rms_acc = ct.full((TILE_SIZE_M,), 0.0, dtype=ct.float32)
    accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
    mma_dtype = ct.tfloat32 if (X.dtype == ct.float32 or W.dtype == ct.float32) else X.dtype
    for k_tile in range(k_tile_start, k_tile_end):
        a = ct.load(
            X,
            index=(bid_m, k_tile),
            shape=(TILE_SIZE_M, TILE_SIZE_K),
            padding_mode=zero_pad,
            allow_tma=True,
        )
        b = ct.load(
            W,
            index=(k_tile, bid_n),
            shape=(TILE_SIZE_K, TILE_SIZE_N),
            padding_mode=zero_pad,
            allow_tma=True,
        )
        a_mma = ct.astype(a, mma_dtype)
        b_mma = ct.astype(b, mma_dtype)
        accumulator = ct.mma(a_mma, b_mma, acc=accumulator)
        a_fp32 = ct.astype(a, ct.float32)
        rms_acc = rms_acc + ct.sum(a_fp32 * a_fp32, axis=1, keepdims=False)
    bid_m_k = bid_m + bid_k * num_bid_m
    ct.store(Y_acc, index=(bid_m_k, bid_n), tile=accumulator)
    # Store RMS partial results - will be summed across bid_n in finalize kernel
    # Using bid_n as additional dimension for partial sums
    ct.store(R_acc, index=(bid_m_k, bid_n), tile=ct.reshape(rms_acc, (TILE_SIZE_M, 1)))
```

**效果**
- 单论开头的 gemm+rms计算，耗时约为 `90us`。
- 对比 DeepGeem：相同的shape 下，Deepgemm 的实现中，同样使用 split-k 算法加速计算，其实现耗时为 `90us`。
- 综合来看，可以说耗时基本接近，和 deepgemm 的实现基本持平，用很少的高级语言+一部分的高性能 Kernel 优化思想即可追平同样的耗时和同样的带宽利用率。
## 5.2 把 scale/bias/sigmoid 融进 split-K 的 finalize（减少一次全量读写）

这一步的直觉是：split-K 本来就需要一个 finalize kernel 把 `Y_acc/R_acc` 归并成最终 `Y/R`，那么把下面这些“逐元素后处理”塞进去，收益很稳定：
- `/ r`（RMSNorm 的 `r` 已经在 finalize 里算出来了）
- `alpha`、`bias`
- `sigmoid / 2*sigmoid`（只作用在 `pre/post` 的前 `2n` 个通道）

**对应实现**
```Python
def mhc_finalize_scale_bias_sigmoid_kernel(
    Y_acc,
    R_acc,
    Y,
    R,
    n: int,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    Bias,
    M: int,
    N: int,
    K: int,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    SPLIT_K: ConstInt,
):
    """Finalize split-K + fused scale/bias/sigmoid kernel for mHC."""
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    y_accum = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
    r_accum = ct.full((TILE_SIZE_M, 1), 0.0, dtype=ct.float32)
    # Sum across split_k dimension
    for split_idx in range(SPLIT_K):
        bid_m_k = bid_m + split_idx * num_bid_m
        y_tile = ct.load(
            Y_acc,
            index=(bid_m_k, bid_n),
            shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        y_accum = y_accum + y_tile
        # RMS is independent of bid_n; each bid_n block stores the same partial RMS.
        # Loading the current bid_n avoids over-counting when num_bid_n > 1.
        r_tile = ct.load(
            R_acc,
            index=(bid_m_k, bid_n),
            shape=(TILE_SIZE_M, 1),
            padding_mode=ct.PaddingMode.ZERO,
        )
        r_tile = ct.astype(r_tile, ct.float32)
        r_accum = r_accum + r_tile
    denom = ct.full((TILE_SIZE_M, 1), K * 1.0, dtype=ct.float32)
    mean = ct.truediv(r_accum, denom)
    rstd = ct.rsqrt(mean)
    ones = ct.full((TILE_SIZE_M, 1), 1.0, dtype=ct.float32)
    r = ct.truediv(ones, rstd)
    if bid_n == 0:
        r_out = ct.astype(r, R.dtype)
        ct.store(R, index=(bid_m, 0), tile=r_out)
    offsets = ct.arange(TILE_SIZE_N, dtype=ct.int32)
    col_ids = bid_n * TILE_SIZE_N + offsets
    bias = ct.load(Bias, index=(bid_n,), shape=(TILE_SIZE_N,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.reshape(bias, (1, TILE_SIZE_N))
    one = ct.full((TILE_SIZE_N,), 1.0, dtype=ct.float32)
    zero = ct.full((TILE_SIZE_N,), 0.0, dtype=ct.float32)
    mask_pre = ct.where(ct.less(col_ids, n), one, zero)
    mask_post = ct.where(ct.less(col_ids, 2 * n), one, zero)
    mask_post = mask_post - mask_pre
    mask_res = one - mask_pre - mask_post
    scale = alpha_pre * mask_pre + alpha_post * mask_post + alpha_res * mask_res
    scale = ct.reshape(scale, (1, TILE_SIZE_N))
    linear = ct.truediv(y_accum * scale, r) + ct.astype(bias, ct.float32)
    sigmoid_linear = _sigmoid(linear)
    two_sigmoid = sigmoid_linear * 2.0
    mask_pre = ct.reshape(mask_pre, (1, TILE_SIZE_N))
    mask_post = ct.reshape(mask_post, (1, TILE_SIZE_N))
    mask_res = ct.reshape(mask_res, (1, TILE_SIZE_N))
    out = linear * mask_res + sigmoid_linear * mask_pre + two_sigmoid * mask_post
    out = ct.astype(out, Y.dtype)
    ct.store(Y, index=(bid_m, bid_n), tile=out)
```

**效果**
- 作为前一个算子的收尾算子+融合 scale，其耗时不大，将 gemm+rms+scale 作为一个整体来看，实现的带宽仍能达到 `4513 GBps`.

## 5.3 sinkhorn：把 `n×n` 当成寄存器里的小矩阵来算
`n=4` 时 `n×n=16`，对每个 sample 而言数据量极小，真正的瓶颈反而是：
- 多次 kernel launch（行归一化/列归一化拆开会非常亏）
- `gather/scatter` 带来的非连续访问和额外指令

**优化思路**
- [x] 避免使用 `gather/scatter`：把 residual block 视作连续的 16 个元素，直接 `load → reshape → store`
- [x] 使用 `exp2` 快速近似 `exp`：`exp(x) = exp2(x * log2(e))`，通常在 GPU 上更友好

**对应实现**
```Python
def mhc_sinkhorn_kernel(
    Y,
    n: ct.Constant[int],
):
    """Sinkhorn-Knopp normalization for residual block (in-place on Y)."""
    row = ct.bid(0)
    total = n * n
    mat = ct.load(Y, index=(row, 0), shape=(1, total))
    mat = ct.reshape(mat, (n, n))
    mat = ct.astype(mat, ct.float32)
    mat = ct.exp2(mat * LOG2E)
    for _ in range(20):
        row_sum = ct.sum(mat, axis=1, keepdims=True)
        mat = ct.truediv(mat, row_sum)
        col_sum = ct.sum(mat, axis=0, keepdims=True)
        mat = ct.truediv(mat, col_sum)
    mat = ct.reshape(mat, (1, total))
    mat = ct.astype(mat, Y.dtype)
    ct.store(Y, index=(row, 0), tile=mat)
```

**效果**
- 这个算子计算量和 IO 量都不大，主要还是看 IO 的实现效率，循环 20 次耗时 `26us`左右，已经接近了手搓 cuda 的性能了。

## 5.4 apply：明确是带宽问题，用“连续 load + 向量 FMA”把中间态压扁
apply 阶段的数学很简单：
- `X_res = H_res · X`
- `X_post = H_post ⊙ F_out`
- `Out = X_res + X_post`

但 `X/Out` 都是 `[B, n, C]` 的大张量，真正的优化目标是：
> [!important]
> 让 `X` 只被连续读取一次，让 `Out` 只被连续写一次；不要让 `H_post/H_res` 的访问打断吞吐。

**优化思路**
- [x] 手写 FMA，不用 MMA：`n=4` 时 `4×4` 的 mixing 太小，走 `mma` 只会引入额外搬运/对齐成本；直接做 `acc += h * x` 更划算
- [x] 避免 `gather/scatter`：把 `y_post/y_res` 先 `narrow/view` 成 `[B, n]` 和 `[B, n, n]`，kernel 内直接 `ct.load` 连续块
- [x] 沿 `C` 做分块（`TILE_SIZE_C`）：每个 CTA 负责 `[n, TILE_SIZE_C]` 的输出 tile，让访问模式尽量全是连续向量 load/store

**对应实现**
```Python
def mhc_apply_residual_kernel(
    X,
    F_out,
    Y_post,
    Y_res,
    Out,
    C: int,
    n: ct.Constant[int],
    TILE_SIZE_C: ConstInt,
):
    """Apply H_res and H_post to residual stream (in-place on Out)."""
    # Shapes:
    # - X: [B, n, C] view of residual stream
    # - F_out: [B, C]
    # - Y_post: [B, n]
    # - Y_res: [B, n, n]
    # - Out: [B, n, C]
    row = ct.bid(0)
    c_tile = ct.bid(1)
    compute_dtype = ct.float32 if (X.dtype == ct.float32 or F_out.dtype == ct.float32 or Y_post.dtype == ct.float32) else X.dtype
    f_tile = ct.load(
        F_out,
        index=(row, c_tile),
        shape=(1, TILE_SIZE_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    f_tile = ct.astype(f_tile, compute_dtype)
    h_post = ct.load(
        Y_post,
        index=(row, 0),
        shape=(1, n),
        padding_mode=ct.PaddingMode.ZERO,
    )
    h_post = ct.reshape(h_post, (n, 1))
    h_post = ct.astype(h_post, compute_dtype)
    h_res = ct.load(
        Y_res,
        index=(row, 0, 0),
        shape=(1, n, n),
        padding_mode=ct.PaddingMode.ZERO,
    )
    h_res = ct.reshape(h_res, (n, n))
    h_res = ct.astype(h_res, compute_dtype)
    acc = ct.full((n, TILE_SIZE_C), 0.0, dtype=compute_dtype)
    for j in range(n):
        x_row = ct.load(
            X,
            index=(row, j, c_tile),
            shape=(1, 1, TILE_SIZE_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_row = ct.reshape(x_row, (1, TILE_SIZE_C))
        x_row = ct.astype(x_row, compute_dtype)
        h_col = ct.extract(h_res, (0, j), shape=(n, 1))
        x_row = ct.broadcast_to(x_row, (n, TILE_SIZE_C))
        h_col = ct.broadcast_to(h_col, (n, TILE_SIZE_C))
        prod = h_col * x_row
        acc = acc + prod
    h_post = ct.broadcast_to(h_post, (n, TILE_SIZE_C))
    f_tile = ct.broadcast_to(f_tile, (n, TILE_SIZE_C))
    x_post = h_post * f_tile
    out_tile = acc + x_post
    out_tile = ct.astype(out_tile, Out.dtype)
    out_tile = ct.reshape(out_tile, (1, n, TILE_SIZE_C))
    ct.store(Out, index=(row, 0, c_tile), tile=out_tile)
```

**效果**
- 见上文的bench 数据，其实现的带宽极高，能达到`6602 GBps`, 说明其优化空间已经不大。
