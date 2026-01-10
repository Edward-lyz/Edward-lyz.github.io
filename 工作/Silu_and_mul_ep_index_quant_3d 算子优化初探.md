# 1. 引言

这个算子作为 BZZ2 卡上吞吐做到 HZZ2 卡上吞吐的两倍项目中的一个扫尾小算子，优化工程被提上日程。那么作为工程师，首先要做的是明确状态：这个算子目前的表现如何，MFU/MBU 是多少，算子理论是什么瓶颈，目前实现的版本是否满足预期？下一步才是针对性地优化，可以用 NCU 等 perf 软件，查看目前实现的卡点代码在哪里。最后一步才是查看源码，针对性地修改问题。那么按照这个逻辑，本文会逐一展开每个点的分析过程。

# 2. 算子语义

这个算子用于 dp_ep_moe 中的一个环节，该算子的输入接口定义如下：

```Shell
speedgate.silu_and_mul_ep_index_quant_3d(
            input_t,
            scale_t,
            actual_token_num_t,
            x_q,
            x_s,
            num_sm,
        )
    """
    input:  [num_experts, max_tokens, 2 * hidden] bf16
    scale:  [num_experts] bf16
    actual_token_num: [num_experts] int32
    x_q:    [num_experts, max_tokens, hidden] fp8
    x_s:    [num_experts, max_tokens, hidden // 128] fp32 (column-major in last two dims)
    """
```

该算子的语义是：对 [E, T_eff, 2H] 的输入做 silu(x) * y * scale[e].

其中 X 是input 的[e, T_eff, 0:hidden], Y 是 input 的[e, T_eff, hidden:2*hidden]。再按 128 通道分组做 per-group 的 absmax()，生成 [E, T_eff, H] 的 FP8 激活 + [E, T_eff, H/128] 的 fp32 scale。

# 3. 优化思路

查看源代码发现，是任务拆分不均匀导致的。一个 block 的线程要去计算所有的 num_experts 的数据量。导致了其实只有一个 SM 在计算，从而拖慢了计算。

其他优化点不大，因为这个算子是一个明显的 IO 瓶颈，计算部分其实没有什么好做 overlap 或者消除串行的。

# 4. 结果对比

![[image 2.png|image 2.png]]

忽略 MFU 部分，这里除以的是 TensorCore 的算力峰值

可以看到，优化后的时间-tokens 图像，已经符合预期：耗时随着 tokens 增长线性增长。且耗时也几乎砍一半还多。

同时，保证了结果的正确性：

```Shell
============================================================ test session starts ============================================================
platform linux -- Python 3.12.12, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/users/liyanzhen01/PUBLIC/SpeedGate/script
configfile: pytest.ini
plugins: anyio-4.11.0, typeguard-4.4.4
collected 15 items

test_silu_and_mul_index_3d.py 16 8000 3584
Test passed for config: (16, 8000, 3584)
.8 8000 3584
Test passed for config: (8, 8000, 3584)
.4 8000 3584
Test passed for config: (4, 8000, 3584)
.2 8000 3584
Test passed for config: (2, 8000, 3584)
.1 8000 3584
Test passed for config: (1, 8000, 3584)
...........

============================================================= warnings summary ==============================================================
../../../../../../../usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63
  /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
    import pynvml  # type: ignore[import]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================================================= 15 passed, 1 warning in 8.57s =======================================================
```