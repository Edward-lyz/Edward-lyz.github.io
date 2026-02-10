# 1. 引言
在前文中，我们用深入的分析和简单的实现，就追平了 DeepGemm 的 mHC 的第一个融合算子的实现。因此，我们可以进一步追问：Cutile 的极限在哪里？因此我们就有这篇文章，证明 Cutile 不仅是 triton 的平替，更是 CUDA 的平替，能够用简单的代码+成熟的优化手段，追平 DeepGemm 的精巧优化。下面，我们就开始复刻DeepGemm 的经典 API：group_gemm，且包括 contiguous / masked 两个语义。

# 算子语义分析 & 对比
