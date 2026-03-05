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
- code_test 一些实验性代码
- include 引擎的接口头文件
- src 源码
 - core 核心代码部分包含以下几个大类：
    1. 统一的kv cache系统，为 page attention做准备
    2. gguf loader
    3. memory 管理
    4. tensor的管理
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
