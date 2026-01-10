# 一、引言

在推理业务里，算子性能基本决定了整网的吞吐和时延。现在常见的做法，是围绕“某个模型 + 某种部署形态”，拿一次 nsys 抓一段 trace，看下热点 kernel 和瓶颈。这种方式能解决眼前的问题，但不太好复用。问题主要有几个：
- 每次都是临时脚本和配置，很难沉淀；
- 只能看到当前这一小撮 shape / 配置的表现；
- 一旦换了硬件、换了后端版本，就很难和历史数据对起来。
- 算子层作为引擎的支持层，不能把性能验证放在引擎整网验证的后面，这些性能测试的工作需要做在前面。

这个小项目想做的事情很简单：把“算子性能验证”变成一套固定流程，有统一的测试骨架、固定的计时方式、固定的写库格式。不同算子、不同后端、不同硬件的结果都能放到同一套数据里，方便后面做对比和回归。初次的代码提交的 PR 见：[https://console.cloud.baidu-int.com/devops/icode/repos/baidu/hac-aiacc/aiak_ds_tool/reviews/118739725/files/base...latest](https://console.cloud.baidu-int.com/devops/icode/repos/baidu/hac-aiacc/aiak_ds_tool/reviews/118739725/files/base...latest)

下面按照：**现状问题 → 设计和实现 → 使用方式 → 总结**，来展开。

---

# 二、现状与问题

从推理引擎的视角看，现在我们在算子性能上主要有这几类问题。

1. 视角过于依赖具体模型
    
    - 在某个模型上做 nsys 分析，只能覆盖到这一小段 shape 和配置；
    
    - 很难回答一个实际常问的问题：在更大的参数空间里，这个算子实现大概是什么水平。
    

1. 跨硬件、跨后端的数据难以对齐
    
    - 算子实现散落在 PyTorch、FlashAttention、FlashInfer、TRT-LLM、DeepGEMM 等各种库里，接口千奇百怪；
    
    - 评测脚本大多是临时写的，每次遇到新的测试需求，就要新建一个脚本测试，不利于多人合作&共享结果，以及长期维护。比如下图，虽然有组织地在维护，但是还是之后编写的同学才好跑测试，没法解耦。
    
    ![[PixPin_2025-11-14_16-54-29.png]]
    

1. 结果不易复现，缺少系统沉淀
    
    - 没有统一的写库和去重机制，历史结果要么在日志里，要么在 Excel 里，要么在个人脚本里；
    
    - 换新硬件、升级后端时，很难在已有基线的基础上快速判断，是变快了还是变慢了。
    

综合起来，现在算子层的调优和验收还是比较靠人。经验多的人能玩得转，新人很难接手，工程资产也堆不起来，简单来说，就是不构成体系。

---

# 三、具体工作与设计

下面把这套东西简称为「算子测试库」。

**核心思路：**在一个统一骨架下，用配置驱动算子功能 / 性能测试，然后按约定写入 SQLite，方便后续脚本化分析和画图。

## **3.1 整体目标**

围绕上面这些问题，这个库要提供的能力可以简单拆成几条：

- 一套通用的测试骨架
    
    从配置加载，到参数展开、输入构造、调用后端、计时、算指标、写库，都走同一条链路。
    

- 多后端、多硬件统一对齐
    
    相同的参数扫描逻辑和计时方式，能同时覆盖 FlashInfer、FlashAttention、DeepGEMM 等库，以及 B200、H200 等不同 GPU。
    

- 统一落库
    
    所有结果（包括环境信息和指标）都写入 SQLite，表结构一致，方便后续用 Python 脚本一把读出来分析。
    

- 版本可追踪
    
    后端版本、精度组合、硬件型号都写在库里并校验，避免出现“这组数据不知道是哪版代码跑出来的”这种情况。
    

## **3.2 架构设计**

**整体结构分成三层：Runner、算子 Tester、后端路由**。

![[aiak_op_profiler.drawio.png]]

### **3.2.1 统一 Runner层**

Runner 就是整条测试链路的入口和调度器。主要做几件事：

- 解析命令行参数，加载 YAML 配置；

- 支持单卡和多卡采集：
    
    - 多卡模式下，父进程按顺序派生子进程，设置好 CUDA_VISIBLE_DEVICES；
    
    - 子进程负责本卡的采样，把每组参数的平均时延写到 JSONL；
    
    - 父进程汇总所有子进程结果，一次性写入 SQLite；
    

- 支持“只采样不落库”模式，通过环境变量控制，方便纯调试场景。

**Runner 尽量只管任务调度和配置，不参与具体算子逻辑。**

### **3.2.2 算子测试器层**

这一层是具体算子的 Tester，比如 GEMM、Grouped GEMM、MHA、MLA 等。一个 Tester 一般会做下面几个步骤：

1. 参数展开
    
    根据 YAML 里的 params，把列表或区间展开成一串具体配置。
    

1. 构造输入
    
    在目标 CUDA 设备上构造输入张量。量化前一般先用 BF16 / FP16 这类精度做原始输入。
    

1. 选择后端实现
    
    根据硬件型号、精度、参数形状等信息，挑选对应的后端 runner。
    

1. 执行和计时
    
    - 先跑一两次试验，早点暴露接口和 shape 问题；
    
    - 然后用统一的 CUDA Event 计时工具多次重复运行，算出平均时延。
    

1. 指标计算
    
    - 根据理论 FLOPs 和 IO 字节数，算出 GFLOPs 和 GB/s；
    
    - 结合硬件峰值算力和带宽，算出 MFU 和 MBU。
    

1. 写入数据库
    
    - 按统一的 schema 写入 SQLite；
    
    - 用每类算子定义好的唯一索引，保证去重和幂等更新。
    

对上层而言，Tester 提供的是输入配置，得到结果的黑箱。对下层而言，把各种后端、精度和量化细节都包在内部。做到高内聚，低耦合。

### **3.2.3 后端路由层**

这层负责解决具体“调谁”的问题：在某个硬件 + 某种精度 + 某个算子的组合下，应该调用哪个库、走哪条实现路径。

它主要做：

- 根据硬件（如 GB200 / H200）、精度（FP4 / FP8 / BF16 / FP16）和参数形状，决定使用哪个库、哪个 kernel；

- 把 FlashInfer、FlashAttention、DeepGEMM、TRT-LLM 等不同库的 API 封成统一的 callable，外层只看到一个 runner；

- 把量化相关代码集中在几个工具模块里，比如 FP4 / FP8 的量化 / 反量化逻辑，不在各处复制粘贴。

这样的好处是：

- 某个第三方库升级或行为变化时，只需要在路由层动少量代码；

- 新接入一个后端库时，Runner 和 Tester 基本可以不动。

## **3.3 数据库和指标**

为了方便长期维护和自动分析，存储和指标这块也有统一约定。

- 存储：以 SQLite 为主
    
    - 每块 GPU 使用一个库文件，例如 data/b200.db；
    
    - 表名由算子、精度码、后端名和版本号拼出来，例如：
        
        - gemm_c8_deep_gemm_2.1.1+c9f8b34。
        
    

- 表结构：维度列 + 指标列
    
    - 维度列：算子的核心参数
        
        - 对 GEMM 来说是 m、n、k、num_groups、tp_size 等；
        
        - 对 MHA 来说是 batch_size、seq_len、num_heads 等。
        
    
    - 指标列：time_us、gflops、gbytes_per_s、mfu、mbu 等；
    
    - 字段名和精度固定，方便跨实验直接拼接。
    

- 唯一索引和去重
    
    - 每种算子定义一组合适的唯一索引字段；
    
    - 写库时用“插入或替换”策略：
        
        - 新实验可以覆盖旧数据；
        
        - 多次重复跑同一配置不会刷出一堆重复行。
        
    

- 理论模型和硬件峰值
    
    - 理论模块给出不同数据类型下的 FLOPs 和 IO 字节数；
    
    - 硬件模块给出不同 GPU 的峰值算力和带宽；
    
    - Tester 组合这两部分数据，算出 GFLOPs / GB/s 以及 MFU / MBU。
    

---

# 四、效果展示

从使用者的角度，希望这套东西最好一键启动：给个配置，跑一条命令，就能拿到能分析的数据。一个典型流程如下。

1. 准备环境
    
    - 使用内置或推荐的 Docker 镜像，里面预装常用后端库：PyTorch、FlashInfer、FlashAttention、DeepGEMM 等；
    
    - 如果要加新后端，或者升级版本，在镜像或虚拟环境里装好对应版本，然后在 YAML 里写清楚版本号即可。
    

1. 写一份 YAML 配置
    
    - 在 configs/<op>/config.yaml 里补几个关键字段：
        
        - backend 和 backend_version（必填，运行时会检查）；
        
        - dtypes（输入 / 输出精度，例如 fp8 / bf16）；
        
        - params（可以是枚举列表，也可以用 {min, max, step} 的方式写区间）。
        
    

1. 运行命令

以 GEMM 为例，如果不指定 --config，会默认读取内置配置。多卡模式下，Runner 会自动按卡位派生子进程采样，在父进程中聚合和写库。比如用下面的配置来启动测试：

```YAML
backend: deep_gemm
backend_version: "2.1.1+c9f8b34"
dtypes: {'input': 'fp8', 'output': 'bf16'}
params:
  M: {min: 2, max: 4, step: 2}
  N: [8192]
  K: [1536]
```

```Bash
python3 -m aiak_op_profiler.runner --gemm
[Runner] spawning child for device=0: /usr/bin/python3 /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/aiak_op_profiler/runner.py --gemm -> /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/data/.tmp/collect_gemm_0.jsonl
[Version OK] backend=deep_gemm version=2.1.1+c9f8b34 matches config.
[Runner] op=gemm backend=deep_gemm version=2.1.1+c9f8b34             dtypes={'input': 'fp8', 'output': 'bf16'} devices=[0] combos=2
[Runner] spawning child for device=1: /usr/bin/python3 /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/aiak_op_profiler/runner.py --gemm -> /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/data/.tmp/collect_gemm_1.jsonl
[Version OK] backend=deep_gemm version=2.1.1+c9f8b34 matches config.
[Runner] op=gemm backend=deep_gemm version=2.1.1+c9f8b34             dtypes={'input': 'fp8', 'output': 'bf16'} devices=[0] combos=2
[Runner] spawning child for device=2: /usr/bin/python3 /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/aiak_op_profiler/runner.py --gemm -> /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/data/.tmp/collect_gemm_2.jsonl
[Version OK] backend=deep_gemm version=2.1.1+c9f8b34 matches config.
[Runner] op=gemm backend=deep_gemm version=2.1.1+c9f8b34             dtypes={'input': 'fp8', 'output': 'bf16'} devices=[0] combos=2
[Runner] spawning child for device=3: /usr/bin/python3 /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/aiak_op_profiler/runner.py --gemm -> /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/data/.tmp/collect_gemm_3.jsonl
[Version OK] backend=deep_gemm version=2.1.1+c9f8b34 matches config.
[Runner] op=gemm backend=deep_gemm version=2.1.1+c9f8b34             dtypes={'input': 'fp8', 'output': 'bf16'} devices=[0] combos=2
[Runner] spawning child for device=4: /usr/bin/python3 /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/aiak_op_profiler/runner.py --gemm -> /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/data/.tmp/collect_gemm_4.jsonl
[Version OK] backend=deep_gemm version=2.1.1+c9f8b34 matches config.
[Runner] op=gemm backend=deep_gemm version=2.1.1+c9f8b34             dtypes={'input': 'fp8', 'output': 'bf16'} devices=[0] combos=2
[Runner] spawning child for device=5: /usr/bin/python3 /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/aiak_op_profiler/runner.py --gemm -> /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/data/.tmp/collect_gemm_5.jsonl
[Version OK] backend=deep_gemm version=2.1.1+c9f8b34 matches config.
[Runner] op=gemm backend=deep_gemm version=2.1.1+c9f8b34             dtypes={'input': 'fp8', 'output': 'bf16'} devices=[0] combos=2
[Runner] spawning child for device=6: /usr/bin/python3 /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/aiak_op_profiler/runner.py --gemm -> /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/data/.tmp/collect_gemm_6.jsonl
[Version OK] backend=deep_gemm version=2.1.1+c9f8b34 matches config.
[Runner] op=gemm backend=deep_gemm version=2.1.1+c9f8b34             dtypes={'input': 'fp8', 'output': 'bf16'} devices=[0] combos=2
[Runner] spawning child for device=7: /usr/bin/python3 /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/aiak_op_profiler/runner.py --gemm -> /home/users/liyanzhen01/PUBLIC/aiak_ds_tool/data/.tmp/collect_gemm_7.jsonl
[Version OK] backend=deep_gemm version=2.1.1+c9f8b34 matches config.
[Runner] op=gemm backend=deep_gemm version=2.1.1+c9f8b34             dtypes={'input': 'fp8', 'output': 'bf16'} devices=[0] combos=2
[Write-Aggregated] gemm_a8_deep_gemm_2.1.1+c9f8b34 {'tp_size': 1, 'num_groups': 1, 'm': 2, 'n': 8192, 'k': 1536, 'time(us)': 265.48, 'MFU': 0.0001, 'MBU': 0.0059}
[Write-Aggregated] gemm_a8_deep_gemm_2.1.1+c9f8b34 {'tp_size': 1, 'num_groups': 1, 'm': 4, 'n': 8192, 'k': 1536, 'time(us)': 259.81, 'MFU': 0.0002, 'MBU': 0.0061}
```

跑完之后：

- 对应 GPU 的 SQLite 库里会多出一批表，或者更新一批行:

![[PixPin_2025-11-14_17-17-31.png]]

- 日志里记录下本次运行的环境信息和元数据:

![[PixPin_2025-11-14_17-07-01.png]]

后面只要在 plot 脚本里：

- 从 SQLite 读出需要的表；

- 按维度筛一下参数；

- 喂给统一的画图脚本，就能把结果画出来。

对新同学/使用者来说，只要搞清楚几个配置字段，就能在这个框架下完成算子性能采集，不需要从零写脚本、对齐接口和计时逻辑。

---

# 五、总结

围绕「算子性能流程规范化」这个目标，算子测试库目前主要做了几件事：

1. 把视角从模型层下沉到算子层

1. 搭了一套统一的测试骨架

1. 建了一份结构化的性能数据库

1. 降低接入和使用成本

基于这套框架，后面无论是接入新硬件、换新的算子实现，还是对现有后端做性能回归，都可以尽量复用现在这条路径，而不是再起一套新的脚本和流程。

# 六、TODO

- [x] 如果别人想新增一个 GPU 硬件以及后端，或者想新增一个 OP 加入测试， 理论上应该最小化修改范围。仓库/项目维护者只用维护 base 类，以及最核心的代码即可。别人二次开发时，只用基于暴露出来的接口进行维护即可。

- [ ] 把 simulator 也整合一下；能复用的就复用起来

- [ ] 现在只适配了 B 卡的后端；H 卡的后端还没接入，需要进一步修改 backend 相关代码。

- [ ] 固定一个镜像版本，且接入 k8s 框架，不再需要拉取代码库才能测试。