# 【学习】TurboQuant 数学推导

---

## 第零章 背景

大语言模型推理时，要把之前见过的所有 token 信息存下来，存成 Key-Value Cache，塞在显存里。模型越大、上下文越长，这块缓存就越吃显存。

向量量化干的事情就是把这些高维浮点向量压成几个 bit 的编码，用的时候再解压回来。压缩有损失，但损失能有多小？

TurboQuant 说：损失可以只比理论极限大约 2.7 倍。

要搞懂这个结论怎么来的，得先过一遍用到的数学工具。

---

## 第一章 概率论基本概念

### 1.1 随机变量

随机变量就是一个”结果不确定”的数。掷骰子，结果 $X$ 可能是 1 到 6 中的任何一个。掷之前你不知道会是哪个，但你知道每个结果出现的可能性——这就是它的概率分布。

两种类型：
- 离散的：只能取有限个或可数个值，比如骰子的 1~6
- 连续的：可以取某个区间内的任意实数，比如”随机从 0 到 1 之间取一个数”

连续随机变量的分布用概率密度函数（probability density function, PDF）$f(x)$ 来描述。$X$ 落在区间 $[a, b]$ 内的概率等于

$$
P(a \leq X \leq b) = \int_a^b f(x) \, dx \tag{1}
$$

密度函数本身不是概率，可以大于 1。但全实数轴上积分必须为 1：

$$
\int_{-\infty}^{+\infty} f(x) \, dx = 1 \tag{2}
$$

### 1.2 期望（均值）

期望 $\mathbb{E}[X]$ 是随机变量的”加权平均值”。

离散情况（骰子）：

$$
\mathbb{E}[X] = \sum_{i} x_i \cdot P(X = x_i) = \frac{1+2+3+4+5+6}{6} = 3.5 \tag{3}
$$

连续情况：

$$
\mathbb{E}[X] = \int_{-\infty}^{+\infty} x \cdot f(x) \, dx \tag{4}
$$

直觉上：重复实验无穷多次，所有结果求平均，就是期望。

期望有个好用的性质——线性：

$$
\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] \tag{5}
$$

不管 $X$ 和 $Y$ 是否独立，这个等式都成立。

### 1.3 方差与标准差

方差度量随机变量偏离均值的程度：

$$
\text{Var}(X) = \mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right] \tag{6}
$$

由 (5) 和 (6) 展开，有个等价公式：

$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \tag{7}
$$

标准差是方差的平方根：$\sigma = \sqrt{\text{Var}(X)}$，和原始数据单位相同。

### 1.4 协方差与独立性

两个随机变量之间的协方差：

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\,\mathbb{E}[Y] \tag{8}
$$

- $\text{Cov}(X, Y) > 0$：$X$ 大的时候 $Y$ 也倾向于大
- $\text{Cov}(X, Y) = 0$：线性上无关联（但不一定独立）
- $\text{Cov}(X, Y) < 0$：$X$ 大的时候 $Y$ 倾向于小

独立比协方差为零更强：$X$ 和 $Y$ 独立意味着知道 $X$ 的值对预测 $Y$ 完全没帮助。独立一定推出协方差为零，反过来不一定。

当 $X$ 和 $Y$ 独立时：

$$
\mathbb{E}[XY] = \mathbb{E}[X] \cdot \mathbb{E}[Y], \quad \text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) \tag{9}
$$

---

## 第二章 几个关键概率分布

### 2.1 正态分布（高斯分布）

正态分布在自然界到处都是，中心极限定理给出了原因：大量独立随机因素叠加，结果就近似正态。

$X \sim N(\mu, \sigma^2)$ 的密度函数：

$$
f(x) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \tag{10}
$$

关键参数：$\mu$ 是均值，$\sigma^2$ 是方差。$\mu = 0$, $\sigma = 1$ 时叫标准正态分布 $N(0, 1)$，密度是关于原点对称的钟形曲线。

后面反复要用的一个结论——标准正态的绝对值的期望：

由 (10) 取 $\mu=0, \sigma=1$ 并对 $|z|$ 积分，

$$
\mathbb{E}[|Z|] = \int_0^{+\infty} z \cdot \frac{2}{\sqrt{2\pi}} e^{-z^2/2} \, dz = \sqrt{\frac{2}{\pi}} \tag{11}
$$

推导：因为 $|Z|$ 的密度是 $f_{|Z|}(z) = 2 \cdot \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$（$z \geq 0$），积分里令 $u = z^2/2$，$du = z\,dz$，就变成 $\frac{2}{\sqrt{2\pi}} \int_0^\infty e^{-u} du = \frac{2}{\sqrt{2\pi}} = \sqrt{2/\pi}$。

### 2.2 多维正态分布

$d$ 维正态向量 $g \sim N(\mathbf{0}, I_d)$ 的意思是：$g = (g_1, \ldots, g_d)$，每个 $g_i$ 独立地服从 $N(0,1)$。

由 (10) 和 (9)，各分量独立的联合密度为各分量密度之积：

$$
f(g) = \prod_{i=1}^d \frac{1}{\sqrt{2\pi}} e^{-g_i^2/2} = \frac{1}{(2\pi)^{d/2}} e^{-\|g\|^2/2} \tag{12}
$$

<aside>

注意到密度只依赖 $\|g\|$（到原点的距离），和方向无关——**球对称性**（spherical symmetry）。推球面均匀分布时要用。

</aside>

### 2.3 $\chi^2$ 分布

如果 $g_1, \ldots, g_k$ 独立地服从 $N(0, 1)$，那么

$$
Q = g_1^2 + g_2^2 + \cdots + g_k^2 \sim \chi^2(k) \tag{13}
$$

叫做自由度为 $k$ 的 $\chi^2$ 分布。

基本性质：
- $\mathbb{E}[Q] = k$
- $\text{Var}(Q) = 2k$

$\chi^2$ 分布和 Gamma 分布的关系：$\chi^2(k) = \text{Gamma}(k/2, 1/2)$。

### 2.4 Gamma 分布

$\text{Gamma}(\alpha, \beta)$ 的密度：

$$
f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} \, x^{\alpha - 1} \, e^{-\beta x}, \quad x > 0 \tag{14}
$$

这里 $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t} dt$ 是 Gamma 函数。它是阶乘在实数上的推广：$\Gamma(n) = (n-1)!$。

<aside>

两个特殊值后面常用：$\Gamma(1/2) = \sqrt{\pi}$，$\Gamma(1) = 1$。

</aside>

### 2.5 Beta 分布

Beta 分布定义在 $[0, 1]$ 区间上，密度为：

$$
f(x) = \frac{1}{B(\alpha, \beta)} \, x^{\alpha - 1} (1 - x)^{\beta - 1}, \quad x \in [0, 1] \tag{15}
$$

<aside>

其中 $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$ 是 Beta 函数。

</aside>

它和 Gamma 分布之间有一个后面绑不开的关系：

> **定理**
> 
> 
> $U \sim \text{Gamma}(\alpha, \theta)$
> 
> $V \sim \text{Gamma}(\beta, \theta)$
> 
> $$
> \frac{U}{U + V} \sim \text{Beta}(\alpha, \beta) \tag{16}
> $$
> 
> $U + V$
> 
> $U/(U+V)$
> 

推导球面坐标分布的时候直接要用这个定理。

证明思路（可跳过）：对 $(U, V)$ 做变量替换 $W = U + V$, $T = U/(U+V)$，算 Jacobian，会发现联合密度分离成 $W$ 和 $T$ 各自的边缘密度的乘积，$T$ 的部分正好是 $\text{Beta}(\alpha, \beta)$。

---

## 第三章 高维球面上的均匀分布

### 3.1 单位球面的定义

$d$ 维空间中的单位球面：

$$
S^{d-1} = \{x \in \mathbb{R}^d : \|x\|_2 = 1\} \tag{17}
$$

上标是 $d-1$ 因为球面本身是 $d-1$ 维的”曲面”。比如 $d=3$ 时，$S^2$ 就是我们日常说的球面。

### 3.2 如何生成球面均匀分布

取 $g \sim N(\mathbf{0}, I_d)$（$d$ 维标准正态向量），令：

由 (12) 的球对称性，归一化后方向均匀：

$$
y = \frac{g}{\|g\|_2} \tag{18}
$$

则 $y$ 在 $S^{d-1}$ 上均匀分布。

为什么？因为 $g$ 的密度具有球对称性——对任意正交矩阵 $U$（即 $U^\top U = I$），$Ug$ 和 $g$ 有相同的分布。所以 $Ug / \|Ug\| = Ug / \|g\| = U(g/\|g\|) = Uy$ 和 $y$ 分布相同。这意味着 $y$ 的分布在任何旋转下都不变——这就是球面均匀分布的定义。

### 3.3 球面坐标的边缘分布

到 TurboQuant 的第一个核心数学结果了。

**引理（坐标的 Beta 分布）**：$y$ 在 $S^{d-1}$ 上均匀分布时，$y_1$（选哪个坐标都一样）的密度为：

$$
f_{y_1}(t) = \frac{\Gamma(d/2)}{\Gamma(1/2)\,\Gamma((d-1)/2)} \cdot (1 - t^2)^{(d-3)/2}, \quad t \in [-1, 1] \tag{19}
$$

**完整推导**：

用 $y = g/\|g\|$ 的构造。第一个坐标 $y_1 = g_1 / \|g\|$。

**第一步**：看 $y_1^2$。

由 (18)，

$$
y_1^2 = \frac{g_1^2}{\|g\|^2} = \frac{g_1^2}{g_1^2 + g_2^2 + \cdots + g_d^2} \tag{20}
$$

令 $U = g_1^2$ 和 $V = g_2^2 + \cdots + g_d^2$，则：

- $g_1 \sim N(0,1)$，所以 $U = g_1^2 \sim \chi^2(1) = \text{Gamma}(1/2, 1/2)$
- $g_2^2 + \cdots + g_d^2$ 是 $d-1$ 个独立 $\chi^2(1)$ 的和，所以 $V \sim \chi^2(d-1) = \text{Gamma}((d-1)/2, 1/2)$
- $U$ 和 $V$ 独立（因为 $g_1$ 和 $g_2, \ldots, g_d$ 独立）

由 (16) 代入 (20)：

$$
y_1^2 = \frac{U}{U + V} \sim \text{Beta}\!\left(\frac{1}{2}, \frac{d-1}{2}\right) \tag{21}
$$

**第二步**：从 $y_1^2$ 的分布推出 $y_1$ 的分布。

由 (15) 和 (21)，$y_1^2$ 的密度为：

$$
f_{y_1^2}(s) = \frac{1}{B(1/2, (d-1)/2)} \, s^{-1/2} (1-s)^{(d-3)/2}, \quad s \in [0,1] \tag{22}
$$

对 (22) 做变量替换 $s = t^2$（$t \in [-1,1]$），$ds = 2|t|\,dt$：

$$
f_{y_1}(t) = f_{y_1^2}(t^2) \cdot 2|t| \cdot \frac{1}{2} \tag{23}
$$

最后那个 $\frac{1}{2}$ 是因为 $y_1$ 的分布关于 0 对称（正负各占一半），而 $y_1^2$ 只覆盖了 $s \geq 0$。展开后 $|t|$ 和 $s^{-1/2} = |t|^{-1}$ 恰好抵消：

由 (22) 代入 (23) 并化简，

$$
f_{y_1}(t) = \frac{1}{B(1/2, (d-1)/2)} \cdot (1 - t^2)^{(d-3)/2} \tag{24}
$$

对 (24) 代入 $B(1/2, (d-1)/2) = \frac{\Gamma(1/2) \Gamma((d-1)/2)}{\Gamma(d/2)}$，得到：

$$
\boxed{f_{y_1}(t) = \frac{\Gamma(d/2)}{\Gamma(1/2)\,\Gamma((d-1)/2)} \cdot (1 - t^2)^{(d-3)/2}, \quad t \in [-1, 1]} \tag{25}
$$

### 3.4 高维下趋向正态分布

$d$ 很大时 $y_1$ 长什么样？

**方差**：由对称性，$\|y\|^2 = y_1^2 + \cdots + y_d^2 = 1$，取期望得 $d \cdot \mathbb{E}[y_1^2] = 1$：

$$
\text{Var}(y_1) = \mathbb{E}[y_1^2] = \frac{1}{d} \tag{26}
$$

（$\mathbb{E}[y_1] = 0$ 由对称性得到。）

想想看：单位球面上一个点有 $d$ 个坐标，平方和为 1，所以每个坐标大约是 $\pm 1/\sqrt{d}$ 的量级。

由 (25)，密度函数中的核心项 $(1 - t^2)^{(d-3)/2}$ 可以改写为：

$$
(1 - t^2)^{(d-3)/2} = \exp\!\left(\frac{d-3}{2} \ln(1 - t^2)\right) \tag{27}
$$

当 $|t|$ 较小时（$|t| \ll 1$），对 (27) 用 $\ln(1 - t^2) \approx -t^2$：

$$
\approx \exp\!\left(-\frac{d-3}{2} t^2\right) \approx \exp\!\left(-\frac{d}{2} t^2\right) \tag{28}
$$

这正是 $N(0, 1/d)$ 密度的核心部分 $e^{-t^2/(2 \cdot 1/d)}$。

高维下 $y_1$ 集中在 $|t| \sim 1/\sqrt{d}$ 附近，远离 $\pm 1$，所以近似成立得很好。

### 3.5 坐标间的近似独立

严格来说，$y_1, \ldots, y_d$ 不独立，因为被 $\sum y_j^2 = 1$ 约束住了。但协方差很小：

由 (8)，

$$
\text{Cov}(y_j, y_k) = \mathbb{E}[y_j y_k] - \mathbb{E}[y_j]\mathbb{E}[y_k] = \mathbb{E}[y_j y_k] \tag{29}
$$

由对称性，$\mathbb{E}[y_j y_k] = c$ 对所有 $j \neq k$ 是同一个常数。而

由 (26) 和 (29)，

$$
\mathbb{E}\!\left[\left(\sum_j y_j\right)^2\right] = \sum_j \mathbb{E}[y_j^2] + \sum_{j \neq k} \mathbb{E}[y_j y_k] = d \cdot \frac{1}{d} + d(d-1) \cdot c = 1 + d(d-1)c \tag{30}
$$

另一方面，$\sum_j y_j$ 的分布关于 0 对称（因为 $y$ 和 $-y$ 等概率），但 $\mathbb{E}[(\sum_j y_j)^2]$ 不一定为 0。通过更精细的计算（利用球面上的积分公式），可以得到：

$$
\mathbb{E}[y_j y_k] = 0 \quad \text{（对称性直接给出）} \tag{31}
$$

实际上，对 $j \neq k$，考虑变换 $y_j \to -y_j$（保持球面均匀分布不变），所以 $\mathbb{E}[y_j y_k] = -\mathbb{E}[y_j y_k]$，推出 $\mathbb{E}[y_j y_k] = 0$。

协方差恰好为 0。但不是真正独立——$\sum y_j^2 = 1$ 这个非线性约束还在。高阶相关性（比如 $\mathbb{E}[y_j^2 y_k^2]$ 和 $\mathbb{E}[y_j^2] \cdot \mathbb{E}[y_k^2]$ 的差）是 $O(1/d^2)$ 量级，高维下可以忽略。

<aside>

这意味着什么？如果坐标真正独立，最优策略就是对每个坐标分别做标量量化。近似独立保证这样做几乎不亏。

</aside>

---

## 第四章 最优标量量化

### 4.1 量化是什么

标量量化就是把一个连续值离散化。你有 $K = 2^b$ 个代表值（centroid / 码字），每个输入值映射到最近的代表值。

一个 $K$-级标量量化器两部分：
- 编码器：把实数轴切成 $K$ 个区间 $R_1, \ldots, R_K$，$x \in R_i$ 编码为索引 $i$
- 解码器：索引 $i$ 映射回代表值 $c_i$

目标是最小化均方误差：

$$
\text{MSE} = \mathbb{E}[(X - Q(X))^2] = \sum_{i=1}^K \int_{R_i} (t - c_i)^2 f(t) \, dt \tag{32}
$$

其中 $f(t)$ 是 $X$ 的密度，$Q(X) = c_i$ 当 $X \in R_i$。

### 4.2 Lloyd-Max 算法

联合优化区间和代表值不好直接求解，但可以交替优化：

**固定区间，求最优代表值**：对 (32) 中的 $c_i$ 求导令其为零：

$$
\frac{\partial}{\partial c_i} \int_{R_i} (t - c_i)^2 f(t) \, dt = -2 \int_{R_i} (t - c_i) f(t) \, dt = 0 \tag{33}
$$

由 (33) 解出：

$$
c_i = \frac{\int_{R_i} t \, f(t) \, dt}{\int_{R_i} f(t) \, dt} \tag{34}
$$

就是区间 $R_i$ 内的条件期望——最能代表这个区间的值。

**固定代表值，求最优区间**：每个点分配到最近的 $c_i$，边界就是相邻 centroid 的中点：

$$
b_i = \frac{c_i + c_{i+1}}{2} \tag{35}
$$

两步交替迭代到收敛，就拿到了针对密度 $f(t)$ 的最优量化器。

### 4.3 对 TurboQuant 的应用

TurboQuant 里 $f(t)$ 是已知的 Beta 边缘分布（或正态近似），所以 Lloyd-Max 跑一次离线计算，把最优的 $2^b$ 个 centroid 存成查找表就行了。之后量化只是最近邻查找，很快。

---

## 第五章 随机旋转的妙用

### 5.1 正交矩阵

正交矩阵 $\Pi$ 满足 $\Pi^\top \Pi = \Pi \Pi^\top = I$，代表旋转（或旋转加反射）。三个性质：

1. 保范数：$\|\Pi x\| = \|x\|$
2. 保内积：$\langle \Pi x, \Pi y \rangle = \langle x, y \rangle$
3. 保距离：$\|\Pi x - \Pi y\| = \|x - y\|$

### 5.2 Haar 分布的随机正交矩阵

“随机旋转” $\Pi$ 从正交群上的 Haar 测度中采样。Haar 测度是正交群上唯一的”均匀分布”——对任意固定正交矩阵 $U$，$U\Pi$ 和 $\Pi$ 的分布相同。

实际生成方法：取 $A \in \mathbb{R}^{d \times d}$，每个元素 $A_{ij} \sim N(0,1)$ 独立，对 $A$ 做 QR 分解得到 $A = QR$，取 $\Pi = Q$。

### 5.3 为什么随机旋转有用

任意单位向量 $x \in S^{d-1}$，旋转后 $y = \Pi x$ 在球面上均匀分布。

不管输入 $x$ 是什么，旋转后 $y$ 的统计性质完全一样。这就是”数据无关”（data-oblivious）——一个预计算的 codebook 适用于所有输入，不需要根据数据来调。

$\Pi$ 保距离：

$$
\|x - \tilde{x}\|^2 = \|\Pi x - \Pi \tilde{x}\|^2 = \|y - \tilde{y}\|^2 \tag{36}
$$

旋转后量化再旋转回来，MSE 和直接量化一样。

### 5.4 “高维向量天然正交，还需要旋转吗？”

这个问题值得专门说清楚。

高维空间里从球面上随机取两个向量，它们的内积 $\langle x, z \rangle$ 高度集中在 0 附近——维度越高，越接近正交。这是对的。但这个性质说的是向量**之间**的关系，而量化面对的是一个**单独的向量内部**的问题：怎么把 $x$ 的 $d$ 个坐标压成 $bd$ bit。

旋转解决的是：这 $d$ 个坐标的分布可能很糟糕。

**一个极端例子**

$x = (1, 0, 0, \ldots, 0) \in S^{d-1}$。不旋转直接量化：

- $x_1 = 1$ 需要精确编码
- $x_2 = \cdots = x_d = 0$ 全是零，$b(d-1)$ bit 浪费了
- 只有 $b$ bit 在干活

旋转后 $y = \Pi x$：

- 每个 $y_j \approx N(0, 1/d)$，信息均匀分散到所有坐标
- $bd$ bit 全在干活

现实数据不会像 $(1,0,\ldots,0)$ 这么极端，但 KV cache embedding 的坐标分布往往不均匀——某些维度方差大，某些方差小，维度之间可能有相关性。旋转把这些不均匀全抹平了。

**近似正交不是让旋转多余，而是让旋转生效**

仔细想：

1. 球面均匀分布各向同性 → 旋转后每个坐标承载等量信息
2. 高维下坐标近似独立 → 独立标量量化近最优

第 2 点之所以成立，正是因为高维球面上 $\sum y_j^2 = 1$ 对单个坐标的约束很弱（协方差为 0，高阶相关 $O(1/d^2)$）。维度低的时候（比如 $d = 3$）坐标之间依赖性强，独立标量量化就不行了。

所以高维近似正交性不是让旋转变得多余——它是旋转之后独立标量量化能近最优的原因。

**不旋转行不行？**

如果恰好知道输入分布，当然可以设计数据相关的量化器（比如 PQ 先跑 k-means 学 codebook）。两种路线的对比：

| 方案 | 做法 | 代价 |
| --- | --- | --- |
| 数据相关（PQ 等） | 离线学 codebook | 几十到几百秒的训练，不支持 online |
| 数据无关（TurboQuant） | 随机旋转 + 通用 codebook | 一次 $O(d^2)$ 矩阵乘法，支持 streaming |

KV cache 是逐 token 生成的，做不了离线训练。TurboQuant 选数据无关这条路，旋转是实现这条路的关键操作。

---

## 第六章 TurboQuant_mse：MSE 最优量化

所有拼图都有了，写出 MSE 量化器。

### 6.1 量化流程

**输入**：$x \in S^{d-1}$（单位向量），bit 数 $b$

预计算（只做一次）：
- 随机正交矩阵 $\Pi$
- 针对 Beta 分布的 Lloyd-Max codebook $\{c_1, \ldots, c_{2^b}\}$

**Quant(x)**：
1. $y = \Pi x$
2. 对每个 $j = 1, \ldots, d$：找到最近的 centroid $c_{i_j}$，输出索引 $i_j$（$b$ bit）

**DeQuant(idx)**：
1. 对每个 $j$：$\tilde{y}_j = c_{i_j}$
2. $\tilde{x} = \Pi^\top \tilde{y}$

### 6.2 MSE 上界推导

**第一步：向量 MSE 分解为坐标 MSE**

由 (36)，

$$
\|x - \tilde{x}\|^2 = \|\Pi(x - \tilde{x})\|^2 = \|y - \tilde{y}\|^2 = \sum_{j=1}^d (y_j - \tilde{y}_j)^2 \tag{37}
$$

对 (37) 取期望：

$$
D_{\text{mse}} = \mathbb{E}\!\left[\sum_{j=1}^d (y_j - \tilde{y}_j)^2\right] = \sum_{j=1}^d \mathbb{E}[(y_j - \tilde{y}_j)^2] \tag{38}
$$

由 (25) 各坐标同分布，所以 (38) 中每项相等：

$$
D_{\text{mse}} = d \cdot \mathbb{E}[(y_1 - \tilde{y}_1)^2] \tag{39}
$$

**第二步：标量 MSE 的估计**

$y_1$ 在高维下近似服从 $N(0, 1/d)$。对正态分布 $N(0, \sigma^2)$ 的 Lloyd-Max 最优量化，经典的高分辨率量化理论（Gersho-Gray）给出：

$$
\mathbb{E}[(y_1 - \tilde{y}_1)^2] \approx \frac{\pi\sqrt{3}}{2} \cdot \sigma^2 \cdot 4^{-b} \tag{40}
$$

这里 $\sigma^2 = 1/d$。

**$\pi\sqrt{3}/2$ 这个常数哪来的**：

高分辨率量化理论研究的是当 $K = 2^b$ 很大时的渐近行为。对一维密度 $f(t)$，最优量化器的 MSE 渐近为：

$$
\text{MSE} \sim \frac{1}{12} \cdot K^{-2} \cdot \left(\int f(t)^{1/3} dt\right)^3 \tag{41}
$$

对 (41) 代入 $N(0, \sigma^2)$ 的密度 (10)，$f(t) = \frac{1}{\sqrt{2\pi}\sigma} e^{-t^2/(2\sigma^2)}$，计算积分：

$$
\int_{-\infty}^{+\infty} f(t)^{1/3} \, dt = \int \left(\frac{1}{\sqrt{2\pi}\sigma}\right)^{1/3} e^{-t^2/(6\sigma^2)} \, dt = \left(\frac{1}{\sqrt{2\pi}\sigma}\right)^{1/3} \cdot \sqrt{6\pi}\,\sigma = (6\pi)^{1/2} \cdot \frac{\sigma^{2/3}}{(2\pi)^{1/6}} \tag{42}
$$

将 (42) 代入 (41) 整理（过程比较繁琐），得到：

$$
\text{MSE} \sim \frac{\pi\sqrt{3}}{2} \cdot \sigma^2 \cdot K^{-2} = \frac{\pi\sqrt{3}}{2} \cdot \sigma^2 \cdot 4^{-b} \tag{43}
$$

**第三步：合起来**

由 (39) 和 (40)（取 $\sigma^2 = 1/d$），

$$
D_{\text{mse}} = d \cdot \frac{\pi\sqrt{3}}{2} \cdot \frac{1}{d} \cdot 4^{-b} = \frac{\pi\sqrt{3}}{2} \cdot 4^{-b} \tag{44}
$$

$d$ 消掉了——MSE 上界只跟 bit 数 $b$ 有关，和维度 $d$ 无关。

**小 bit 数的精确值**：高分辨率近似在 $b$ 很小时不够精确，直接对 Beta 分布跑 Lloyd-Max 可以得到精确 MSE：

| $b$ | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| $D_{\text{mse}}$ | 0.36 | 0.117 | 0.03 | 0.009 |

---

## 第七章 内积偏差问题

### 7.1 为什么 MSE 最优 $\neq$ 内积最优

我们关心两件事：
- $\tilde{x}$ 和 $x$ 的距离近（MSE 小）
- $\langle y, \tilde{x} \rangle$ 是 $\langle y, x \rangle$ 的好估计（内积准）

这两个目标不等价。MSE 很小的量化器可能在内积上有系统偏差。

### 7.2 偏差从哪来

考虑最极端的情况：1-bit 量化（$b=1$），每个坐标只有两个代表值 $+c$ 和 $-c$（对称的 Lloyd-Max codebook）。

量化后 $\tilde{y}_j = c \cdot \text{sign}(y_j)$，其中 $\text{sign}(y_j)$ 是 $y_j$ 的符号。

现在考虑内积。取两个向量 $x, z \in S^{d-1}$，量化后的内积估计为：

$$
\langle \Pi z, \tilde{y} \rangle = c \sum_j (\Pi z)_j \cdot \text{sign}(y_j) \tag{45}
$$

令 $u_j = (\Pi z)_j$, $v_j = y_j = (\Pi x)_j$。由于 $\Pi$ 是随机旋转，$(u_j, v_j)$ 近似联合正态。

$$
\mathbb{E}[u_j \cdot \text{sign}(v_j)] \tag{46}
$$

对联合正态 $(U, V) \sim N(0, \Sigma)$，有一个经典公式（后面第八章详细推导）：

$$
\mathbb{E}[U \cdot \text{sign}(V)] = \sqrt{\frac{2}{\pi}} \cdot \frac{\text{Cov}(U, V)}{\sqrt{\text{Var}(V)}} \tag{47}
$$

由 (45)、(46) 和 (47)，

$$
\mathbb{E}[\langle \Pi z, \tilde{y} \rangle] = c \cdot d \cdot \sqrt{\frac{2}{\pi}} \cdot \frac{\text{Cov}(u_1, v_1)}{\sqrt{\text{Var}(v_1)}} \tag{48}
$$

而无偏要求

$$
\mathbb{E}[\langle \Pi z, \tilde{y} \rangle] = \langle \Pi z, y \rangle = \langle z, x \rangle \tag{49}
$$

但实际两边差了一个和 $c$、$\sqrt{2/\pi}$ 相关的乘法因子。1-bit 时这个偏差因子约 $2/\pi \approx 0.637$。MSE 量化器系统性地低估了内积。

### 7.3 偏差随 bit 数增加而减小

直觉上 $b$ 越大，$\tilde{y}_j$ 越接近 $y_j$，非线性偏差越小。$b \to \infty$ 时偏差趋于零。但 2-4 bit 场景下偏差不能忽略。

---

## 第八章 二元正态的 sign 期望公式

这个公式在 QJL 和偏差分析里反复出现，单独推一遍。

### 8.1 问题

设 $(U, V)$ 服从联合正态：

$$
\begin{pmatrix} U \\ V \end{pmatrix} \sim N\!\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \sigma_U^2 & \rho\sigma_U\sigma_V \\ \rho\sigma_U\sigma_V & \sigma_V^2 \end{pmatrix}\right) \tag{50}
$$

求 $\mathbb{E}[V \cdot \text{sign}(U)]$。

### 8.2 推导

联合正态有一个标准的条件分解。$V$ 可以写成：

由 (50) 的条件分布性质，

$$
V = \frac{\rho \sigma_V}{\sigma_U} U + \sqrt{1 - \rho^2} \, \sigma_V \, W \tag{51}
$$

其中 $W \sim N(0, 1)$ 且与 $U$ 独立。

你可以验证这个分解没问题：
- $\mathbb{E}[V] = 0$ ✓
- $\text{Var}(V) = \frac{\rho^2 \sigma_V^2}{\sigma_U^2} \sigma_U^2 + (1-\rho^2)\sigma_V^2 = \sigma_V^2$ ✓
- $\text{Cov}(U, V) = \frac{\rho \sigma_V}{\sigma_U} \text{Var}(U) = \rho \sigma_U \sigma_V$ ✓

将 (51) 代入 $\mathbb{E}[V \cdot \text{sign}(U)]$：

$$
\mathbb{E}[V \cdot \text{sign}(U)] = \frac{\rho \sigma_V}{\sigma_U} \mathbb{E}[U \cdot \text{sign}(U)] + \sqrt{1-\rho^2}\,\sigma_V \, \mathbb{E}[W \cdot \text{sign}(U)] \tag{52}
$$

第二项：$W$ 和 $U$ 独立，$\mathbb{E}[W] = 0$，所以 $\mathbb{E}[W \cdot \text{sign}(U)] = \mathbb{E}[W] \cdot \mathbb{E}[\text{sign}(U)] = 0$。

第一项：$U \cdot \text{sign}(U) = |U|$（数乘以自己的符号就是绝对值）：

由 (11) 和 (52)，

$$
\frac{\rho \sigma_V}{\sigma_U} \mathbb{E}[|U|] = \frac{\rho \sigma_V}{\sigma_U} \cdot \sigma_U \sqrt{\frac{2}{\pi}} = \rho \sigma_V \sqrt{\frac{2}{\pi}} \tag{53}
$$

由 (52) 和 (53)，

$$
\boxed{\mathbb{E}[V \cdot \text{sign}(U)] = \sqrt{\frac{2}{\pi}} \cdot \rho \sigma_V = \sqrt{\frac{2}{\pi}} \cdot \frac{\text{Cov}(U, V)}{\sigma_U}} \tag{54}
$$

后面会再用两次。

---

## 第九章 QJL 变换与无偏内积估计

### 9.1 Johnson-Lindenstrauss 引理的直觉

JL 引理：高维向量随机投影到低维，距离和内积可以被近似保持。QJL 是它的量化版——投影后取符号，每个分量只存 1 bit。

### 9.2 QJL 的定义

取随机矩阵 $S \in \mathbb{R}^{m \times d}$，$S_{ij} \sim N(0,1)$ 独立。

编码：

$$
Q_{\text{qjl}}(x) = \text{sign}(Sx) \in \{-1, +1\}^m \tag{55}
$$

解码：

$$
Q_{\text{qjl}}^{-1}(z) = \frac{\sqrt{\pi/2}}{m} \cdot S^\top z \tag{56}
$$

### 9.3 无偏性的完整证明

要证：$\mathbb{E}[\langle y, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(x)) \rangle] = \langle y, x \rangle$

由 (55) 和 (56) 展开：

$$
\mathbb{E}\!\left[\left\langle y, \frac{\sqrt{\pi/2}}{m} S^\top \text{sign}(Sx) \right\rangle\right] \tag{57}
$$

$S^\top$ 的第 $i$ 列就是 $S$ 的第 $i$ 行 $s_i^\top$。$\text{sign}(Sx)$ 的第 $i$ 个分量是 $\text{sign}(s_i^\top x) = \text{sign}(\langle s_i, x \rangle)$。

所以 (57) 等于：

$$
= \frac{\sqrt{\pi/2}}{m} \sum_{i=1}^m \mathbb{E}\!\left[\langle y, s_i \rangle \cdot \text{sign}(\langle s_i, x \rangle)\right] \tag{58}
$$

对每个 $i$，令 $U = \langle s_i, x \rangle$ 和 $V = \langle s_i, y \rangle$。

因为 $s_i$ 的分量是 i.i.d. $N(0,1)$：
- $U = \sum_k s_{ik} x_k$ 是独立正态变量的线性组合，所以 $U \sim N(0, \|x\|^2)$
- 同理 $V \sim N(0, \|y\|^2)$
- $\text{Cov}(U, V) = \text{Cov}(\sum_k s_{ik} x_k, \sum_l s_{il} y_l) = \sum_k x_k y_k = \langle x, y \rangle$

$(U, V)$ 是联合正态的。对 (58) 中每一项直接套用 (54)：

$$
\mathbb{E}[V \cdot \text{sign}(U)] = \sqrt{\frac{2}{\pi}} \cdot \frac{\text{Cov}(U, V)}{\sigma_U} = \sqrt{\frac{2}{\pi}} \cdot \frac{\langle x, y \rangle}{\|x\|} \tag{59}
$$

将 (59) 代回 (58)（$m$ 项，每项相同）：

$$
\frac{\sqrt{\pi/2}}{m} \cdot m \cdot \sqrt{\frac{2}{\pi}} \cdot \frac{\langle x, y \rangle}{\|x\|} = \frac{\langle x, y \rangle}{\|x\|} \tag{60}
$$

当 $x \in S^{d-1}$（$\|x\| = 1$）时：

$$
= \langle x, y \rangle \quad \blacksquare \tag{61}
$$

$\sqrt{\pi/2}$ 不是凭空出现的——它是 $\sqrt{2/\pi}$ 的倒数，刚好补偿 sign 函数造成的信息损失。

---

## 第十章 TurboQuant_prod：无偏内积量化器

### 10.1 思路

MSE 量化有偏差，QJL 无偏但 MSE 大。把两者拼起来：

1. 先用 MSE 量化吃掉大部分信号
2. 残差用 QJL 做无偏修正

### 10.2 完整算法

**Quant_prod(x)**，总预算 $b$ bit per coordinate：

1. $\text{idx} = \text{Quant}_{\text{mse}}^{(b-1)}(x)$ —— 用 $b-1$ bit 做 MSE 量化
2. $\tilde{x}_{\text{mse}} = \text{DeQuant}_{\text{mse}}(\text{idx})$ —— 重建
3. $r = x - \tilde{x}_{\text{mse}}$ —— 残差
4. $\gamma = \|r\|_2$ —— 残差范数（存为浮点数）
5. $q = \text{sign}(S \cdot r / \gamma)$ —— 对归一化残差做 QJL（1 bit per coordinate）

输出：$(\text{idx}, q, \gamma)$，总共 $(b-1) \cdot d + 1 \cdot d = b \cdot d$ bit（加一个浮点数 $\gamma$）。

**DeQuant_prod(idx, q, γ)**：

1. $\tilde{x}_{\text{mse}} = \text{DeQuant}_{\text{mse}}(\text{idx})$
2. $\tilde{r} = \gamma \cdot \frac{\sqrt{\pi/2}}{d} \cdot S^\top q$ —— QJL 解码并缩放
3. $\tilde{x} = \tilde{x}_{\text{mse}} + \tilde{r}$

### 10.3 无偏性证明

$$
\mathbb{E}[\langle y, \tilde{x} \rangle] = \mathbb{E}[\langle y, \tilde{x}_{\text{mse}} + \tilde{r} \rangle] = \langle y, \tilde{x}_{\text{mse}} \rangle + \mathbb{E}[\langle y, \tilde{r} \rangle] \tag{62}
$$

注意：给定 $\Pi$ 和 $x$，MSE 量化的结果 $\tilde{x}_{\text{mse}}$ 是确定的（没有额外随机性），所以第一项不需要期望。

对 (62) 第二项，$\tilde{r}$ 是对 $r$ 的 QJL 重建（缩放了 $\gamma$），由 (61) 的无偏性：

$$
\mathbb{E}[\langle y, \tilde{r} \rangle] = \langle y, r \rangle = \langle y, x - \tilde{x}_{\text{mse}} \rangle \tag{63}
$$

由 (62) 和 (63)，

$$
\mathbb{E}[\langle y, \tilde{x} \rangle] = \langle y, \tilde{x}_{\text{mse}} \rangle + \langle y, x - \tilde{x}_{\text{mse}} \rangle = \langle y, x \rangle \quad \blacksquare \tag{64}
$$

残差 QJL 提供的修正项把 MSE 量化的偏差抵消干净了。

### 10.4 内积失真上界

由 (64)，误差为 $\langle y, x \rangle - \langle y, \tilde{x} \rangle = \langle y, r - \tilde{r} \rangle$。

$$
D_{\text{prod}} = \mathbb{E}[(\langle y, r - \tilde{r} \rangle)^2] \tag{65}
$$

1. 就是对残差 $r$ 做 QJL 的内积失真。由 QJL 的失真界（$m = d$ 维）：

$$
\leq \frac{C_0}{d} \cdot \|y\|^2 \cdot \|r\|^2 \tag{66}
$$

其中 $C_0$ 是和 QJL 相关的常数。

$\|r\|^2$ 是 MSE 量化的失真，由 (44)：$\mathbb{E}[\|r\|^2] = D_{\text{mse}}^{(b-1)} \leq \frac{\pi\sqrt{3}}{2} \cdot 4^{-(b-1)}$。

将 (44) 代入 (66)：

$$
D_{\text{prod}} \leq \frac{C_0 \cdot \pi\sqrt{3}/2}{d} \cdot \|y\|^2 \cdot 4^{-(b-1)} = \frac{C_1}{d} \cdot \|y\|^2 \cdot 4^{-b} \tag{67}
$$

吸收常数后：

$$
\boxed{D_{\text{prod}} \leq \frac{\pi\sqrt{3}}{2d} \cdot \|y\|^2 \cdot 4^{-b}} \tag{68}
$$

---

## 第十一章 信息论下界

### 11.1 Shannon 率失真理论速览

率失真理论研究的问题：$R$ bit 预算压缩一个随机源，最小失真是多少？

核心结论：$d$ 维正态源 $X \sim N(0, \sigma^2 I_d)$，用 $R = bd$ bit，MSE 失真下界：

$$
D(R) = d \cdot \sigma^2 \cdot 2^{-2R/d} = d \cdot \sigma^2 \cdot 4^{-b} \tag{69}
$$

### 11.2 应用到球面向量

球面均匀分布的每个坐标方差为 $1/d$，行为上接近 $d$ 个独立的 $N(0, 1/d)$ 随机变量。

对 (69) 取 $\sigma^2 = 1/d$，用 $bd$ bit 量化，MSE 下界为：

$$
D_{\text{mse}} \geq d \cdot \frac{1}{d} \cdot 4^{-b} = 4^{-b} \tag{70}
$$

### 11.3 Yao 极小极大原理

内积失真的下界用 Yao 原理建立。直白地说：

> 对于一个”博弈”——量化器设计者要最小化最坏情况失真，对手要选择最难的输入——有以下等式：
> 
> 
> $$
> \min_Q \max_x D(Q, x) = \max_{\text{分布}} \min_{\text{确定性 }Q} \mathbb{E}_x[D(Q, x)] \tag{71}
> $$
> 
> 左边是：最优随机化量化器面对最坏输入的失真
> 右边是：对手选择最难的输入分布，最优确定性量化器的期望失真
> 

通过在 (71) 右边构造特定的输入分布（球面均匀分布），可以证明任何确定性量化器的期望内积失真至少为 $\frac{1}{d} \cdot 4^{-b}$，从而：

由 (70) 和 (71)，

$$
D_{\text{prod}} \geq \frac{1}{d} \cdot 4^{-b} \tag{72}
$$

### 11.4 近最优性总结

| 指标 | 信息论下界 | TurboQuant 上界 | 差距倍数 |
| --- | --- | --- | --- |
| MSE | $4^{-b}$ | $\frac{\pi\sqrt{3}}{2} \cdot 4^{-b} \approx 2.7 \cdot 4^{-b}$ | $\approx 2.7 \times$ |
| 内积 | $\frac{4^{-b}}{d}$ | $\frac{\pi\sqrt{3}}{2} \cdot \frac{4^{-b}}{d} \approx 2.7 \cdot \frac{4^{-b}}{d}$ | $\approx 2.7 \times$ |

上下界的函数形式完全一致——都是 $4^{-b}$ 的指数衰减，只差一个常数。TurboQuant 的量化策略在结构上已经是最优的，指数部分没法再改进。

$b = 1$ 时差距更小（约 $1.45 \times$），极端低 bit 场景下效果尤其好。

---

## 第十二章 全流程总结

TurboQuant 的数学链路：

```
  输入 x ∈ S^{d-1}（单位向量）
       |
       | 随机旋转 Π（Haar 分布）
       v
  y = Πx ∈ S^{d-1}（球面均匀分布）
       |
       | 坐标分解
       v
  y_1, ..., y_d ~ Beta(1/2, (d-1)/2) ≈ iid N(0, 1/d)
       |
       |-- MSE 路径 ──────────────────────────────┐
       |   对每个 y_j 做 Lloyd-Max 标量量化         |
       |   D_mse ≤ (√3·π/2) · 4^{-b}              |
       |                                           |
       |-- 内积路径 ─────────────────────────────┐  |
           第一阶段: (b-1)-bit MSE 量化          |  |
           第二阶段: 对残差做 1-bit QJL          |  |
           无偏性: E[<y,x̃>] = <y,x>   ✓        |  |
           D_prod ≤ (√3·π/2d)·‖y‖²·4^{-b}       |  |
                                                 |  |
  信息论下界: D_mse ≥ 4^{-b}, D_prod ≥ 4^{-b}/d   |
  差距倍数: ≈ 2.7x                                  |
```

三个要点：

1. 随机旋转使一切变简单——不管输入长什么样，旋转后坐标分布已知且近似独立，预计算的标量量化器就够用
2. MSE 最优但有偏，QJL 无偏但 MSE 大——先 MSE 后 QJL 取两家之长
3. 上下界函数形式一致——常数因子意义上这条路到头了

---

## 附录 A：符号速查表

| 符号 | 含义 |
| --- | --- |
| $S^{d-1}$ | $d$ 维空间中的单位球面 |
| $\Pi$ | 随机正交矩阵（Haar 分布） |
| $b$ | 每个坐标的 bit 数 |
| $K = 2^b$ | 量化级数（centroid 个数） |
| $D_{\text{mse}}$ | MSE 失真 $\mathbb{E}[\|x - \tilde{x}\|^2]$ |
| $D_{\text{prod}}$ | 内积失真 $\mathbb{E}[(\langle y, x \rangle - \langle y, \tilde{x} \rangle)^2]$ |
| $\Gamma(\cdot)$ | Gamma 函数，阶乘的连续推广 |
| $B(\alpha, \beta)$ | Beta 函数 $= \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$ |
| $\text{sign}(t)$ | 符号函数：$t > 0$ 时为 $+1$，$t < 0$ 时为 $-1$ |
| QJL | Quantized Johnson-Lindenstrauss 变换 |

## 附录 B：推导里反复用的三个工具

1. **Gamma-Beta 关系**：$U \sim \text{Gamma}(\alpha, \theta)$, $V \sim \text{Gamma}(\beta, \theta)$ 独立 $\Rightarrow$ $U/(U+V) \sim \text{Beta}(\alpha, \beta)$
2. **二元正态 sign 期望**：$\mathbb{E}[V \cdot \text{sign}(U)] = \sqrt{2/\pi} \cdot \text{Cov}(U,V) / \sigma_U$
3. **高分辨率量化渐近**：$N(0, \sigma^2)$ 上 $2^b$级最优量化 MSE $\sim (\pi\sqrt{3}/2) \cdot \sigma^2 \cdot 4^{-b}$

TurboQuant 的全部理论结果就靠这三个。