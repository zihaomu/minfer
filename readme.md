# minfer: min toy inference engine

<label for="file">Dev progress:</label>
<progress id="file" value="5" max="100"> </progress>

## 项目简介：
目标是实现一个轻量级的llm推理引擎，支持gguf模型格式，能够在CPU和GPU上运行。
期望通过这个项目，深入理解llm模型的运行机制和优化方法。
主要包含的点有：
1. llm 推理的基本流程
2. kv-cache优化
3. flash-attn
4. SmoothQuant/AWQ等量化方法
5. LoRA
6. xxx

## 实现规范
1. 所有的weight都是按照`[in_dims, out_dims]`来读取和计算。

## Project Roadmap
- 2025-06-01 🟢 — 每个层都有测试例子，attnention和ffn都能产生正确的结果；
- 2025-09-09 🟢 — 能够load gguf模型，和正确的处理在运行时的模型内存分配；
- 


## TODO
- 
- kv-cache
- 支持fp32格式
- 支持int8格式
- 支持fp16格式