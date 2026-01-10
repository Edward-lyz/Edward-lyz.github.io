# 1. 引言

1d2d，这是个啥？相信你也和我一样，对这个说法表示疑惑。遇到这个问题，是在做算子性能摸底测试的时候，我在 展示deep_gemm 中的 fp8_gemm_nt 的性能数据时，被问到这么一个问题：你用的 gemm 是 1d1d 的，还是 1d2d的？这个问题可把我问到了，我只是隐约知道，deep_gemm 的 cache 里面，会有 1D1D, 1D2D 等等的名字。我还真没在意这些区别，以及背后的代码逻辑。所以我写下此文，希望能够解答我自己的疑惑。

这里先给出一个结论，1D1D 主要用于backward，以及SM100 的架构，1D2D 主要用于 forward，以及 SM90 的架构。这个主要是影响到 gemm 算子的输入前的量化方式，1D1D 是 per token 的， 而 1D2D 是 per block 的。且这个逻辑就是官方的选择逻辑。1D1D 更适配 SM100， 1D2D 更适配 SM90。下面会详细进行解读。

https://github.com/deepseek-ai/DeepGEMM/blob/38f8ef73a48a42b1a04e0fa839c2341540de26a6/tests/generators.py\#L50

> [!info] Question about restriction on BLK_N on SM100 when type is 1d2d · Issue #145 · deepseek-ai/DeepGEMM  
> // 1D2D kernels&#39; maximum block N is 128 // 1D2D kernels require more friendly block Ns if (kernel_type == KernelType::Kernel1D2D and (block_n &gt; 128 or 128 % block_n !  
> [https://github.com/deepseek-ai/DeepGEMM/issues/145#issuecomment-3135308609](https://github.com/deepseek-ai/DeepGEMM/issues/145#issuecomment-3135308609)  

# 2. 从GEMM 计算过程讲起

我们都知道，GEMM 其实是数学上简单（并没有更优化的，降低复杂度的优美算法），硬件/软件实现上比较麻烦的一个算子类型。受限于芯片的矩阵乘单元（在 GPU 上就是 TensorCore）的输入限制，矩阵乘只能拆成一片片（tile）地进行乘加操作。那么我们知道，受限于显存，GEMM 的输入（weight）都是低精度的，比如 FP8 ，那么为了最后计算结果精度不会太差，中间的累加值会选择 FP32/FP16 进行存储。但是，输出还需要被量化回 FP8 的精度，以便作为后续模型层级的计算输入。在 Deepseek 自己放出的模型中，选择的就是 **block-quant** 的模式。

TODO： 补充模型量化知识