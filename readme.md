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
 
- test 测试代码
- layer 测试。



## TODO
- kv-cache
- 支持fp32格式
- 支持int8格式
- 支持fp16格式