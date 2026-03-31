# minfer: Min multimodal llm inference engine

<label for="file">Dev progress:</label>
<progress id="file" value="5" max="100"> </progress>

## 项目简介：
目标是实现一个轻量级的 多模态 llm 推理引擎，支持gguf模型格式，专注于移动端、边缘设备的推理引擎。

主要包含的点有：
1. llm 推理的基本流程
2. 基于page attention的kv-cache优化
3. int8和fp16的支持
4. LoRA的支持

## 文件夹结构
- 3rdparty 第三方依赖
  - googletest # 测试框架
  - libnpy # python对齐的核心框架
  - xsimd  # 跨平台simd指令框架
  - mobilekv # 动态kv管理框架
- code_test 一些实验性代码
- include 引擎的接口头文件
- src 源码
 - core 核心代码部分包含以下几个大类：
    1. 统一的kv cache系统，为 page attention做准备
    2. gguf loader
    3. memory 管理

 - backend 后端计算代码
  - cpu 基于cpu实现的layer，包括simd优化
  - gpu TBD

- test 测试代码
    - 基本测试
    - layer 测试
- benchmark 包含模型的基础速度测试

## TODO
- kv-cache
- 支持fp32格式
- 支持int8格式
- 支持fp16格式

## KV cache支持方案
使用端侧灵活kvcache管理器：https://github.com/zihaomu/mobilekv
在Net层做一个整体的cache的内存申请，然后每一层都进行配置，这个版本的kv cache默认使用ring buffer的版本。基于这个设计去实现。

## Benchmark
项目提供 `minfer_benchmark` 用于评估 LLM 推理速度，并且将 `prefill` 与 `decode` 分开统计。

为什么要拆开：
- `prefill` 会一次处理完整 prompt，吞吐通常用 `prompt tokens / s` 表示。
- `decode` 是逐 token 生成，吞吐通常用 `generated tokens / s` 和 `ms/token` 表示。
- 两者计算形态不同，不能只看一个统一速度值。

### 编译
```bash
cmake -S . -B build
cmake --build build -j
```

### 运行示例
```bash
./build/minfer_benchmark \
  --model /path/to/model.gguf \
  --prompt "Hello world! <s>" \
  --prompt-lens 32,128,512 \
  --decode-tokens 128 \
  --warmup 1 \
  --runs 5 \
  --progress-interval 16
```

可选输出 CSV：
```bash
./build/minfer_benchmark --model /path/to/model.gguf --csv benchmark.csv
```

### 输出指标
- `prefill avg/p50/p90(ms)`：prefill 延迟统计
- `prefill throughput(tok/s)`：prefill token 吞吐
- `decode avg/p50/p90/p99(ms/tok)`：decode 单 token 延迟统计
- `decode throughput(tok/s)`：decode token 吞吐
- `ttft(ms)`：time-to-first-token，约等于 `prefill + 首个 decode step`
- `--progress-interval`：每生成 N 个 token 打印一次 decode 进度，避免长时间无输出


### 算子优化流程

算子也就是kernel，是底层引擎的关键。开发流程如下：
#### CPU算子
实现依赖于`xsimd`库去适配不同的平台指令，覆盖 avx2、avx512、neon、risc-v 等 SIMD 后端。
kernel优化步骤：
1. 生成对应kernel的测试数据
kernel需要严格的测试，对比python用于验证C++实现是否正确。
其中数据生成有python基于numpy，类似于`test/layers/test_data/layer_data_generater.py`或`test/core/test_data/gemm_case_generater.py`。而对应数据在对应python的data文件夹下。

2. C++ 实现和测试
C++对应算子实现放到`src/backend/cpu/kernel`中去。而上一步测试的数据应该在C++的测试样例中被使用，在`test/layers/feedforward_test.cpp`或`test/core/mat_test.cpp`中。

3. benchmark
算子有对应的benchmark脚本，放在`benchmark`，可以是python或者C++实现。
当前仓库已提供 `benchmark/op_benchmark.cpp`，构建后可通过 `./build/minfer_op_benchmark` 对 `rope`、`transpose`、`softmax`、`silu`、`rmsnorm`，以及基础加减乘除广播场景做基准测试。

### GEMM 微内核优化
GEMM是llm模型的最主要的算子，需要的是不是简单的并行，而是针对平台，计算精度，以及原始矩阵规模设定一个完整的并行策略。
