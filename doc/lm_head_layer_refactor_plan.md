# LmHeadLayer 改造计划

## 1. 背景

当前 `LinearLayer_19` 是模型最后的 `lm_head` 投影层，对应 `hidden -> vocab` 的输出映射。

现状特征：

- decode 阶段每生成 1 个 token，都要执行一次 `1 x hidden -> vocab` 投影。
- 在当前基线中，`LinearLayer_19` 长期占 decode 总时延约 `27% ~ 30%`。
- 这部分已经是主热点，继续只做通用 `LinearLayer` 微调，边际收益会越来越小。

当前关键代码路径：

- `src/core/gguf_model/gguf_loader.cpp`：最终输出投影由 `LinearLayerParams` 创建
- `src/backend/cpu/layer/linear_layer.cpp`：`w.gemmNT(x).copyTo(out);`
- `src/backend/cpu/layer/runtime_weight.cpp`：decode `outer == 1` 的 packed GEMV fast path
- `src/backend/cpu/layer/output_layer.cpp`：完整 logits copy
- `src/core/utils.cpp`：`argmax_tokens()` 对完整 logits 做扫描

结论：

- `lm_head` 已经不是“普通线性层”。
- 它需要独立的精度策略、执行策略、输出策略。
- 这次改造的目标，不是简单重命名，而是把 `lm_head` 升级成一个终端层。

---

## 2. 改造目标

本次改造聚焦四件事：

1. 把最终输出投影从通用 `LinearLayer` 中拆出，形成独立 `LmHeadLayer`
2. 给 `LmHeadLayer` 单独的 runtime precision policy，允许它不再被全局精度覆盖
3. 为 decode 提供单独输出 contract，支持 `full_logits / argmax / top-k`
4. 为 `LmHeadLayer` 引入可选的 batch=1 专用执行策略，以及后续的绑核 / NUMA 策略

最终目标不是“某个内核更快一点”，而是：

- 默认兼容现有接口
- decode 路径可按策略走最省路径
- `lm_head` 可以独立演进，不再绑死在通用 `LinearLayer` 抽象上

---

## 3. 设计原则

### 3.1 保持兼容

- 现有 `prefill()` / `step()` 继续保留，默认仍返回完整 logits
- 现有测试、benchmark、example 不因本次改造直接失效

### 3.2 特化必须显式开启

- `argmax-only` / `top-k-only` 必须是显式 decode 模式，不允许悄悄改变默认行为
- 绑核 / NUMA 必须是 opt-in，不允许在不支持的平台强行开启

### 3.3 先拆 contract，再做激进优化

- 先把 `LmHeadLayer` 的边界立住
- 再做 batch=1 专用 kernel、输出裁剪、绑核与 NUMA
- 不把所有风险一次性压到一个提交里

---

## 4. 非目标

本计划当前不包含：

- 重写所有 `LinearLayer`
- 修改 attention / FFN 的数据流
- 破坏现有 full logits API
- 默认开启 `lm_head-only INT8`
- 为所有平台强制接入 `libnuma`

---

## 5. 分阶段计划

## P0. 独立 `LmHeadLayer` 骨架 + 精度策略打通

### 范围

- 新增 `LayerType::LmHead`
- 新增 `LmHeadLayerParams`
- loader 在创建最终输出投影时，改为创建 `LmHeadLayerParams`
- CPU backend 注册 `LmHeadLayer`
- `LmHeadLayer` 第一版先复用当前 packed GEMV 能力，不在这一阶段重写 kernel
- 修正当前“构图时全局精度覆盖层级精度”的问题，让 `lm_head` 支持单独 precision policy

建议改动位置：

- `include/minfer/layer.h`
- `src/core/gguf_model/gguf_loader.cpp`
- `src/backend/cpu/backend_cpu.cpp`
- `src/backend/cpu/layer/`
- `src/core/net.impl.cpp`

### 交付物

- 一个可实例化、可运行、可 profile 的 `LmHeadLayer`
- `LmHeadLayer` 默认行为与当前 `LinearLayer_19` 保持一致
- `LmHeadLayer` 可读取自身 precision policy，不再被全局 runtime precision 无条件覆盖

### 验收标准

1. 构建通过

```bash
cmake --build build -j
```

2. 模型可正常加载与运行，`--layer-profile` 输出中最后一层名称变为 `LmHeadLayer_*`

```bash
./build/minfer_benchmark --layer-profile --threads 4 --prompt-lens 128 --decode-tokens 32 --warmup 1 --runs 1
```

3. 默认 `full_logits` 模式下，功能保持兼容：
- `prefill()` / `step()` 仍返回完整 logits
- greedy token 与改造前一致

4. 关键回归测试通过：

```bash
./build/minfer_test --gtest_filter='Mat_TEST.runtime_weight*:Net_TEST.*:Layer_TEST.quantized_*'
```

5. 默认模式性能不回退超过阈值：
- `prompt_len=128, threads=4/8/16` 下
- `decode avg(ms/tok)` 相比改造前基线回退不得超过 `3%`
- `prefill throughput(tok/s)` 回退不得超过 `2%`

6. 精度策略生效：
- 设置 `lm_head` 独立 precision policy 后，日志或断言可证明最终实例化精度不是被全局值覆盖

---

## P1. Decode 输出 contract 独立化

### 范围

- 新增 decode 专用输出接口，例如：
  - `full_logits`
  - `argmax`
  - `top_k`
- 引入新返回结构，例如 `DecodeResult`
- 保留旧接口不变，新接口只服务 decode 特化路径
- `LmHeadLayer` 可以根据输出策略决定是否需要完整 logits buffer

### 交付物

- 新 decode API
- `LmHeadLayer` 的输出策略枚举
- 兼容路径与特化路径并存

### 验收标准

1. 旧接口兼容：
- 现有 `prefill()` / `step()` 示例与测试不需要修改即可通过

2. 新接口正确：
- `argmax` 模式输出 token 与 `full_logits + argmax_tokens()` 结果完全一致
- `top-k` 模式返回的 token id 和 score 与 full logits 参考结果一致

3. `top-k` 结果满足以下约束：
- 长度等于请求的 `k`
- score 按从高到低排序
- token id 不重复

4. 新接口增加测试覆盖：
- greedy decode 一致性测试
- top-k 正确性测试
- `k=1 / k>1 / vocab tail` 边界测试

5. shortlist 模式不再走完整 logits 对外暴露路径：
- `argmax/top-k` 模式下，不再要求 `OutputLayer` 对完整 vocab logits 做外部 copy
- profile 中 `OutputLayer` 不应再成为 decode 路径可见热点

---

## P2. batch=1 专用 `LmHeadLayer` kernel + epilogue 融合

### 范围

- 为 `LmHeadLayer` 提供 decode 专用入口
- 专门针对 `batch=1, seq=1, hidden -> vocab` 路径做调度
- 将 `argmax` / `top-k` reduce 尽量前移到 `LmHeadLayer` 内部完成
- 在 `full_logits` 模式下保留完整输出
- 必要时引入 `LmHeadExecutionContext`

说明：

- 这一步不是重复造一个通用 GEMV
- 核心是把“投影 + shortlist 归约”做成同一终端阶段

### 交付物

- `LmHeadLayer` decode 专用 fast path
- `argmax/top-k` 的 fused epilogue
- 可选的 `full_logits` fallback

### 验收标准

1. 正确性通过：
- `argmax` 模式 token 与 `full_logits` 参考完全一致
- `top-k` 模式 token 集合与排序与参考一致

2. 性能必须有实质收益，测量口径固定为：

```bash
OMP_PROC_BIND=close OMP_PLACES=cores \
./build/minfer_benchmark --layer-profile --threads {4,8,16} --prompt-lens 128 --decode-tokens 128 --warmup 1 --runs 5
```

3. `argmax` 或 `top-k` 模式下，相比 P0 默认 full logits 基线：
- `LmHeadLayer` decode `Avg(ms)` 下降 `>= 15%`
- 整体 `decode avg(ms/tok)` 下降 `>= 8%`

4. `full_logits` 模式不能被误伤：
- 相比 P0 默认 full logits 基线，性能回退不得超过 `3%`

5. profile 证据成立：
- `LmHeadLayer` 仍是热点，但占比必须下降
- `OutputLayer` 不再承担 shortlist 路径的主要开销

---

## P3. `LmHeadLayer` 执行策略，绑核与 NUMA

### 范围

- 将 `LmHeadLayer` 的执行策略从“只认 OpenMP 线程数”升级为“可选绑核 / NUMA”
- 优先支持外部配置和轻量 runtime policy，例如：
  - `OMP_PROC_BIND=close`
  - `OMP_PLACES=cores`
  - `numactl --cpunodebind=... --membind=...`
- 如果收益明确，再考虑内建 affinity policy

建议分两步：

1. 先做外部验证
- 不改 runtime 调度
- 用 `numactl` / OMP 绑定先证明收益真实存在

2. 再做内建策略
- 只对 `LmHeadLayer` 生效
- 不影响其他层默认行为

### 交付物

- 一套 `LmHeadLayer` 执行策略配置项
- 多平台 fallback 逻辑
- 一份绑定前后 benchmark 对比记录

### 验收标准

1. 功能上必须是 opt-in：
- 未开启绑定策略时，行为与 P2 一致
- 不支持 affinity / NUMA 的机器必须自动回退，不得崩溃

2. 在多核机器上，绑核策略可观测：
- 日志或调试输出能看到启用状态
- 至少能确认线程绑定策略已生效

3. 在多 NUMA 节点机器上，收益门槛如下：
- `decode avg(ms/tok)` 改善 `>= 5%`，或
- `LmHeadLayer Avg(ms)` 改善 `>= 8%`

4. 稳定性约束：
- `p99 decode ms/tok` 不得恶化超过 `10%`
- 连续 3 轮 benchmark 结果方向一致，不允许只赢一次

5. 单 NUMA 节点或不支持场景下：
- 不得引入超过 `3%` 的回退

---

## P4. 可选增强，`lm_head-only` 精度特化

### 范围

- 在 `LmHeadLayer` 独立后，允许只对 `lm_head` 使用更激进的 precision policy
- 候选策略：
  - `force_fp16`
  - `force_int8`
- 其他层保持原精度不变

说明：

- 这一步是可选增强，不作为拆层的前置阻塞项
- 只有在精度风险和收益都被量化后才默认推荐

### 交付物

- `lm_head` 独立 precision policy 配置
- 对应 benchmark 与精度对比报告

### 验收标准

1. 功能正确：
- 仅 `lm_head` 精度变化，其他层 precision 不变

2. 精度可接受：
- greedy decode 的 top-1 一致率达到预设门槛
- 建议门槛：固定 prompt 集上 `>= 99%`

3. 性能收益成立：
- `LmHeadLayer Avg(ms)` 相比 FP16 路径改善 `>= 15%`

4. 不允许静默降质：
- 若一致率不达标，不得作为默认路径启用

---

## 6. 统一 benchmark 与验收口径

所有阶段统一使用以下主口径：

### 6.1 端到端 benchmark

```bash
OMP_PROC_BIND=close OMP_PLACES=cores \
./build/minfer_benchmark --layer-profile --threads 4 --prompt-lens 128 --decode-tokens 128 --warmup 1 --runs 5

OMP_PROC_BIND=close OMP_PLACES=cores \
./build/minfer_benchmark --layer-profile --threads 8 --prompt-lens 128 --decode-tokens 128 --warmup 1 --runs 5

OMP_PROC_BIND=close OMP_PLACES=cores \
./build/minfer_benchmark --layer-profile --threads 16 --prompt-lens 128 --decode-tokens 128 --warmup 1 --runs 5
```

重点指标：

- `decode avg(ms/tok)`
- `decode p90 / p99`
- `LmHeadLayer Avg(ms)` 与 `%`
- `prefill throughput(tok/s)`

### 6.2 runtime-only 微基准

```bash
OMP_PROC_BIND=close OMP_PLACES=cores \
./build/minfer_op_benchmark --gemm-runtime-only --batch 1 --seq-len 1 --hidden 512 --gemm-out 32768 --warmup 20 --iters 200 --threads 4

OMP_PROC_BIND=close OMP_PLACES=cores \
./build/minfer_op_benchmark --gemm-runtime-only --batch 1 --seq-len 1 --hidden 512 --gemm-out 32768 --warmup 20 --iters 200 --threads 8

OMP_PROC_BIND=close OMP_PLACES=cores \
./build/minfer_op_benchmark --gemm-runtime-only --batch 1 --seq-len 1 --hidden 512 --gemm-out 32768 --warmup 20 --iters 200 --threads 16
```

重点指标：

- `runtime fp32/fp16/int8 avg(ms)`
- 线程数变化下的扩展性

### 6.3 正确性口径

建议至少覆盖：

```bash
./build/minfer_test --gtest_filter='Mat_TEST.runtime_weight*:Mat_TEST.gemm_*:Net_TEST.*:Layer_TEST.quantized_*'
```

对于新接口，需新增：

- `LmHeadLayer` full logits / argmax / top-k 一致性测试
- decode 单步 API 测试
- shortlist 边界测试

---

## 7. 推荐执行顺序

推荐严格按以下顺序推进：

1. `P0`，先拆层，立边界，打通 per-layer precision
2. `P1`，把 decode 输出 contract 独立出来
3. `P2`，做 batch=1 专用 `LmHeadLayer` fast path
4. `P3`，在数据证明有效后再做绑核与 NUMA
5. `P4`，最后评估 `lm_head-only` 精度特化是否值得默认开启

原因很直接：

- 如果没有 `LmHeadLayer`，后面的优化都只能继续往通用 `LinearLayer` 里塞例外逻辑
- 如果没有新 decode contract，`argmax/top-k only` 就只能偷偷破坏旧接口
- 如果没有先做基线，绑核 / NUMA 很容易变成“看起来很高级，但收益不稳定”

---

## 8. 完成标准

这份计划最终以以下条件作为“改造完成”判断：

1. `LmHeadLayer` 已独立存在，并成为最终输出投影的标准实现
2. 旧 full logits API 仍然可用
3. 新 decode shortlist API 已上线并有测试覆盖
4. `LmHeadLayer` 在 shortlist 模式下，相比当前 full logits 基线取得稳定性能收益
5. 绑核 / NUMA 只在数据证明有效的平台启用，不把系统调优硬编码成默认行为

这才算“把 lm_head 当作一等公民处理完了”，不是只把热点换了个名字。
