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

- **B**：Batch × SeqLen

- **n**：mHC 的扩张系数（通常为 4）

- **C**：原始 hidden size

- **X_t ∈ R^{B × nC}**：n 路 residual 拼接后的输入

---

### 2.1 一次 GEMM，生成全部未约束系数

mHC 首先通过一次大的线性变换，生成所有 mixing 的“原始参数”：

```Plain
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

```Plain
r = sqrt( sum(X_t^2) / (nC) )
```

这个 `r` 是：

- 每个 sample 一个 scalar（shape 为 `[B, 1]`）

- 后续会用于统一缩放三组系数

---

### 2.3 scale / alpha / bias 修正

三组系数会被统一做一次线性修正：

```Plain
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

mHC 与 HC 的一个本质区别在于：

**所有 mixing 系数都被限制在可控范围内**。

具体约束如下：

- **Pre mixing**
    
    ```Plain
    H_pre = sigmoid(H̃_pre)
    ```
    
    → 保证非负
    

- **Post mixing**
    
    ```Plain
    H_post = 2 * sigmoid(H̃_post)
    ```
    
    → 非负，且上界为 2
    

- **Residual mixing**
    
    ```Plain
    H_res = Sinkhorn-Knopp(H̃_res)
    ```
    
    → 投影为近似双随机矩阵（行和 = 1，列和 = 1）
    

`Sinkhorn-Knopp` 的过程是：

1. 以 `exp(H̃_res)` 为初始矩阵

1. 交替进行行归一化、列归一化

1. 重复固定次数（论文中约 20 次）

---

### 2.5 将系数真正应用到 residual stream

得到 `H_pre / H_post / H_res` 后，mHC 才真正开始“改写 residual”。

1. **Pre：n → 1**

```Plain
F_pre = Σ_i H_pre[i] · X_t[i]
```

得到一个标准的 `[B × C]` 输入，送入 Attention / FFN。

1. **中间层计算**

```Plain
F_out = F(F_pre)
```

这一步与 mHC 无关，通常是传统的 transformer 层

1. **Post & Res：重组 n 路 residual**

```Plain
X_res  = H_res · X_t
X_post = H_post^T · F_out
X_{t+1} = X_res + X_post
```

最终输出仍然是 `[B × nC]`，供下一层使用。

---

# 3. 算子融合 / 实现分析

如果严格按上述步骤逐算子实现，mHC 在工程上是**不可接受的**。

论文中提到的三类融合算子，实际上都是为了解决非常具体的性能问题。

---

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

- 在 Fusion 1 的输出 buffer 上

- 直接完成：
    
    - `/ r`
    
    - `alpha`
    
    - `+ bias`
    
    - `sigmoid / 2*sigmoid`
    

这一类融合的逻辑是计算量很小，但是多次launch 带来的开销会超过算子本身耗时，融合是为了**避免 launch 变慢**。

---

### 3.3 Fusion 3：Sinkhorn-Knopp 单 kernel 迭代

`Sinkhorn-Knopp` 的 `n×n` 矩阵非常小（n 通常为 4），

但需要多次强依赖迭代。

如果按 naive 方式实现：

```Python
for t in range(20):
    row_norm kernel
    col_norm kerne
```

会导致：

- 大量 kernel launch

- 大量 device-level 同步

**正确做法**是：

- 一个 kernel

- 将 n×n 矩阵放在寄存器 / shared memory

- 在 kernel 内部完成 20 次迭代

- backward 时直接重算中间状态

---

### 3.4 应用阶段融合：H_res / H_post + residual merge

这是最容易被忽略，但对带宽影响极大的部分。

如果分步实现：

1. `X_res = H_res · X`

1. `X_post = H_post^T · F_out`

1. `X_next = X_res + X_post`

会产生大量中间读写，内存流量非常不友好。

论文中的做法是：

> [!important]
> 
> 将 H_res 的应用、H_post 的应用、residual merge 合成一个 kernel

逻辑上等价于：

```Plain
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

1. **一个“大而贵”的融合 GEMM**

1. **两个“小而碎”的系数后处理 kernel**

1. **一个“强迭代”的 Sinkhorn kernel**

1. **一个“极致带宽优化”的 residual 应用 kernel**

后续使用 CUDA 实现 demo，本质上就是围绕这四类 kernel 做拆解与验证

---

# 4. 具体代码实践

## 4.1 GEMM + RMS_Norm

  

# 5. 进一步优化思路