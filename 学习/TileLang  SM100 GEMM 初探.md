# 1. 背景

之前 `TileLang` 的生态还是不够好，针对 B 卡的新架构，在 25 年 10 月底的这个 [PR](https://github.com/tile-ai/tilelang/pull/1108) 之前, 都用的是自行维护的 `TVM` 组件，导致了新架构压根不支持。同时，由于我们之前讨论过 `CuTIle` 这个语言写出来的 `GEMM` 的性能在 `SM100` 架构上的性能表现不尽如人意，叠加最近社区更新了 `SM100` 的 `GEMM` 的示例代码，因此我们来简单上手+学习一下 `TileLang` 语言编写出的 GEMM 的性能如何。

# 2. 实测+分析

简单修改 `tilelang/examples/gemm_sm100/gemm_tcgen5mma.py` 中的输入输出的 `dtype`，以及把 `MNK` 设置到`[320, 32768, 7168]`, 从而测试同样规模下的 GEMM 的性能对比。

性能数据如下

```Bash
Latency: 0.23449599742889404 ms
Flops: 641.0508367230555 TFLOPS
```

转化成的 CUTLASS 的代码如下：

```C
\#include <tl_templates/cuda/instruction/tcgen05mma.h>
\#include <tl_templates/cuda/tcgen_05.h>
\#include <tl_templates/cuda/cuda_fp8.h>
\#include <tl_templates/cuda/gemm.h>
\#include <tl_templates/cuda/copy.h>
\#include <tl_templates/cuda/reduce.h>
\#include <tl_templates/cuda/ldsm.h>
\#include <tl_templates/cuda/threadblock_swizzle.h>
\#include <tl_templates/cuda/debug.h>
\#ifdef ENABLE_BF16
\#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
\#endif

extern "C" __global__ void main_kernel(const fp8_e5_t* __restrict__ A, const fp8_e5_t* __restrict__ B, bfloat16_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const fp8_e5_t* __restrict__ A, const fp8_e5_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ uint C_tmem[1];
  __shared__ uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  tl::Tcgen05SMemDescriptor desc_a;
  tl::Tcgen05SMemDescriptor desc_b;
  float C_local[64];
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(C_tmem[0])), 128);
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
  }
  __syncthreads();
  \#pragma unroll
  for (int i = 0; i < 4; ++i) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((i * 4096) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((int)blockIdx.y) * 917504) + (i * 229376)) + ((((int)threadIdx.x) >> 3) * 7168)) + ((((int)threadIdx.x) & 7) * 16)), (((((int)blockIdx.y) * 2) + (i >> 1)) < 5));
  }
  \#pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_1 * 4096) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), B+((((((int)blockIdx.x) * 917504) + (i_1 * 229376)) + ((((int)threadIdx.x) >> 3) * 7168)) + ((((int)threadIdx.x) & 7) * 16)));
  }
  tl::cp_async_commit();
  \#pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_2 * 4096) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), A+(((((((int)blockIdx.y) * 917504) + (i_2 * 229376)) + ((((int)threadIdx.x) >> 3) * 7168)) + ((((int)threadIdx.x) & 7) * 16)) + 128), (((((int)blockIdx.y) * 2) + (i_2 >> 1)) < 5));
  }
  \#pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_3 * 4096) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 49152), B+(((((((int)blockIdx.x) * 917504) + (i_3 * 229376)) + ((((int)threadIdx.x) >> 3) * 7168)) + ((((int)threadIdx.x) & 7) * 16)) + 128));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 54; ++k) {
    tl::cp_async_wait<1>();
    __syncthreads();
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a, (&(((fp8_e5_t*)buf_dyn_shmem)[((k & 1) * 16384)])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b, (&(((fp8_e5_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 32768)])), 1, 64, 0, 0, 2);
      \#pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::tcgen05mma_ws_ss<tl::DataType::kFloat8_e5m2>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 32)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, ((0 < ki) ? 1 : ((k == 0) ? 0 : 1)), static_cast<uint32_t>(136316048), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
    __syncthreads();
    \#pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((k & 1) * 16384) + (i_4 * 4096)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((((int)blockIdx.y) * 917504) + (i_4 * 229376)) + ((((int)threadIdx.x) >> 3) * 7168)) + (k * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 256), (((((int)blockIdx.y) * 2) + (i_4 >> 1)) < 5));
    }
    \#pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((k & 1) * 16384) + (i_5 * 4096)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), B+((((((((int)blockIdx.x) * 917504) + (i_5 * 229376)) + ((((int)threadIdx.x) >> 3) * 7168)) + (k * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 256));
    }
    tl::cp_async_commit();
    mbar[0].wait((k & 1));
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::initialize_tcgen05_descriptor(desc_a, (&(((fp8_e5_t*)buf_dyn_shmem)[0])), 1, 64, 0, 0, 2);
    tl::initialize_tcgen05_descriptor(desc_b, (&(((fp8_e5_t*)buf_dyn_shmem)[32768])), 1, 64, 0, 0, 2);
    \#pragma unroll
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      tl::tcgen05mma_ws_ss<tl::DataType::kFloat8_e5m2>(uint64_t(desc_a + (ki_1 * 32)), uint64_t(desc_b + (ki_1 * 32)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, 1, static_cast<uint32_t>(136316048), 0, 0, 0, 0);
    }
    tl::tcgen05_mma_arrive((&(mbar[0])));
  }
  mbar[0].wait(0);
  tl::cp_async_wait<0>();
  __syncthreads();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::initialize_tcgen05_descriptor(desc_a, (&(((fp8_e5_t*)buf_dyn_shmem)[16384])), 1, 64, 0, 0, 2);
    tl::initialize_tcgen05_descriptor(desc_b, (&(((fp8_e5_t*)buf_dyn_shmem)[49152])), 1, 64, 0, 0, 2);
    \#pragma unroll
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      tl::tcgen05mma_ws_ss<tl::DataType::kFloat8_e5m2>(uint64_t(desc_a + (ki_2 * 32)), uint64_t(desc_b + (ki_2 * 32)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, 1, static_cast<uint32_t>(136316048), 0, 0, 0, 0);
    }
    tl::tcgen05_mma_arrive((&(mbar[0])));
  }
  mbar[0].wait(1);
  tl::tcgen05_ld_32dp32bNx<64, false>(C_tmem[0], ((((int)threadIdx.x) >> 7) * 64), (&(C_local[0])));
  __syncthreads();
  \#pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    uint2 __1;
    float4 v_ = *(float4*)(C_local + (i_6 * 4));
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((int)threadIdx.x) & 127) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + (i_6 * 4))) = __1;
  }
  __syncthreads();
  \#pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    if (((((int)blockIdx.y) * 2) + (i_7 >> 1)) < 5) {
      tl::st_global_256(&(*(ulonglong4*)(C + (((((((int)blockIdx.y) * 4194304) + (i_7 * 1048576)) + ((((int)threadIdx.x) >> 3) * 32768)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 7) * 16)))), *(ulonglong4*)(((bfloat16_t*)buf_dyn_shmem) + ((i_7 * 4096) + (((int)threadIdx.x) * 16))));
    }
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(C_tmem[0])), 128);
  }
}

```

简单回顾了一下生成的内容，发现性能差的原因是：
1. 不支持 `TMA`操作，不能直接调用 `TMEM` 的读写
2. `Warp-Group` 的特性还不支持

综上，在 2026 年年初，`TileLang` 对 `SM100` 等新硬件的适配还是较差。