# GEMM 专项 Benchmark Matrix 与里程碑清单

## 1. 目标

只聚焦 `GEMM` 一个算子，围绕下面三个目标建立专项基准与优化路线：

- 不同精度下都有稳定表现：`FP32`、`FP16 weight + FP32 accumulate`、`INT8 rowwise weight + FP32 accumulate`
- 不同维度下都没有明显短板：既覆盖 LLM 常见大矩阵，也覆盖 decode 的小 `M` 与 tail case
- 不只看 micro benchmark，也看整算子入口，确保优化能真实传导到模型推理

当前仓库里 GEMM 的主要事实如下：

- 快路径聚焦 `A[M, K] x B[N, K] -> C[M, N]` 的 `NT` 路线，见 `src/core/mat_gemm.cpp`
- `FP16` / `INT8` 权重 GEMM 已支持，但 `INT8` 目前只支持 `transA=false, transB=true`
- `benchmark/op_benchmark.cpp` 已经提供 `--gemm-only` 与 `--gemm-micro-only` 两个专项入口
- `RuntimeWeight::gemmNT()` 已经有 decode packed 路线，是 decode 小批场景的关键优化入口

## 2. Benchmark 合同

### 2.1 算子合同

本专项统一使用以下 GEMM 合同：

- 输入：`A[M, K]`
- 权重：`B[N, K]`
- 输出：`C[M, N]`
- 在现有 benchmark 中，对应参数换算为：
  - `M = batch * seq_len`
  - `K = hidden`
  - `N = gemm_out`
  - 输入张量形状：`[batch, seq_len, hidden]`
  - 输出张量形状：`[batch, seq_len, gemm_out]`

### 2.2 精度轴

每个场景固定都测三档：

| 精度 | 权重类型 | 累加类型 | 当前入口 |
| --- | --- | --- | --- |
| FP32 | `DT_32F` | FP32 | `gemm(gemm_a, gemm_w, false, true)` |
| FP16 | `DT_16F` | FP32 | `gemm(gemm_a, gemm_w_fp16, false, true)` |
| INT8 | `DT_8S` + per-row scales | FP32 | `gemm(gemm_a, gemm_w_int8, gemm_w_int8_scales, false, true)` |

### 2.3 基准分层

每个形状至少跑两层：

- `micro-kernel` 层：看内核/pack 策略本身，命令使用 `--gemm-micro-only`
- `gemm entry` 层：看完整 GEMM 入口实际表现，命令使用 `--gemm-only`

推荐顺序：

1. 先用 `micro-kernel` 判断是不是内核本身退化
2. 再用 `gemm entry` 判断 pack、调度、线程、广播等路径是否吞掉收益

## 3. 复现要求

每次正式记录结果，必须一并保存以下上下文：

- commit SHA
- CPU 型号 / 核数 / 是否开启 SMT
- 编译模式，建议 `Release`
- 编译选项，尤其是 SIMD 与 OpenMP 开关
- 线程数 `--threads`
- benchmark 二进制：`./build/minfer_op_benchmark`
- warmup 次数、iters 次数
- 固定 seed：当前 benchmark 已内置 `std::mt19937 rng(20260306)`

建议每轮对比统一采用：

- warmup：`10`
- measured iters：`30` 或 `50`
- 每个场景至少重复一次全量 sweep，确认结果不是噪声

## 4. GEMM 专项 Benchmark Matrix

## 4.1 维度族定义

把场景拆成四个维度族：

### A. Decode 小 M

目标：优化单 token / 小批次延迟。

| 族名 | batch | seq_len | M | K(hidden) | N(gemm_out) |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode_m1_h512_ffn | 1 | 1 | 1 | 512 | 2048 |
| decode_m1_h1024_ffn | 1 | 1 | 1 | 1024 | 4096 |
| decode_m1_h2048_ffn | 1 | 1 | 1 | 2048 | 8192 |
| decode_m2_h2048_ffn | 2 | 1 | 2 | 2048 | 8192 |
| decode_m4_h4096_ffn | 4 | 1 | 4 | 4096 | 16384 |
| decode_m8_h4096_ffn | 8 | 1 | 8 | 4096 | 16384 |

说明：

- 重点看 `RuntimeWeight::gemmNT()` 的 decode packed 路线
- 这组结果最直接影响 `decode ms/tok`

### B. Prefill 吞吐

目标：优化大 `M` 时的吞吐。

| 族名 | batch | seq_len | M | K(hidden) | N(gemm_out) |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill_m16_h512_ffn | 1 | 16 | 16 | 512 | 2048 |
| prefill_m32_h1024_ffn | 1 | 32 | 32 | 1024 | 4096 |
| prefill_m64_h2048_ffn | 1 | 64 | 64 | 2048 | 8192 |
| prefill_m128_h2048_ffn | 1 | 128 | 128 | 2048 | 8192 |
| prefill_m256_h4096_ffn | 1 | 256 | 256 | 4096 | 16384 |
| prefill_m512_h4096_ffn | 1 | 512 | 512 | 4096 | 16384 |

说明：

- 重点看 blocked kernel、pack 成本与 OpenMP 切分
- 这组结果最直接影响 `prefill tok/s`

### C. LLM 真实投影族

目标：覆盖不同输入输出比值，而不是只测 `N = 4K`。

| 场景 | batch | seq_len | M | K | N |
| --- | ---: | ---: | ---: | ---: | ---: |
| attn_qkv_h512 | 1 | 128 | 128 | 512 | 1536 |
| attn_o_h512 | 1 | 128 | 128 | 512 | 512 |
| ffn_up_h512 | 1 | 128 | 128 | 512 | 2048 |
| ffn_down_h512 | 1 | 128 | 128 | 2048 | 512 |
| attn_qkv_h4096 | 1 | 128 | 128 | 4096 | 12288 |
| attn_o_h4096 | 1 | 128 | 128 | 4096 | 4096 |
| ffn_up_h4096 | 1 | 128 | 128 | 4096 | 16384 |
| ffn_down_h4096 | 1 | 128 | 128 | 16384 | 4096 |

说明：

- `qkv` 检查 `N = 3K`
- `o_proj` 检查 `N = K`
- `up/gate` 检查 `N = 4K`
- `down_proj` 检查 `K = 4N`

### D. Tail / 非对齐尺寸

目标：避免维度一旦不对齐就出现断崖。

围绕 SIMD lane、`kKernelNR`、blocking 边界做邻域测试：

| 族名 | batch | seq_len | M | K | N |
| --- | ---: | ---: | ---: | ---: | ---: |
| tail_small_63_63_63 | 1 | 63 | 63 | 63 | 63 |
| tail_small_64_64_64 | 1 | 64 | 64 | 64 | 64 |
| tail_small_65_65_65 | 1 | 65 | 65 | 65 | 65 |
| tail_prefill_127_4096_4095 | 1 | 127 | 127 | 4096 | 4095 |
| tail_prefill_128_4096_4096 | 1 | 128 | 128 | 4096 | 4096 |
| tail_prefill_129_4096_4097 | 1 | 129 | 129 | 4096 | 4097 |
| tail_decode_m1_k4096_n4095 | 1 | 1 | 1 | 4096 | 4095 |
| tail_decode_m1_k4096_n4096 | 1 | 1 | 1 | 4096 | 4096 |
| tail_decode_m1_k4096_n4097 | 1 | 1 | 1 | 4096 | 4097 |

说明：

- 这组场景专门看 tail 处理质量
- 目标不是一定跑最快，而是相邻尺寸不要突然掉速

## 4.2 执行优先级

如果时间有限，建议按下面顺序执行：

1. `Decode 小 M`
2. `Prefill 吞吐`
3. `LLM 真实投影族`
4. `Tail / 非对齐尺寸`

## 5. 命令模板

## 5.0 推荐先用 sweep 脚本

仓库中已提供批量脚本：

```bash
benchmark/gemm_bench_sweep.sh --group decode --mode both --threads 32
benchmark/gemm_bench_sweep.sh --group decode --mode runtime --threads 32
benchmark/gemm_bench_sweep.sh --group prefill --mode entry --threads 32
benchmark/gemm_bench_sweep.sh --group tail --filter 4096 --dry-run
```

脚本默认：

- benchmark 二进制：`./build/minfer_op_benchmark`
- 输出目录：`benchmark/results/gemm/<timestamp>`
- 基线报告：`benchmark/results/gemm/<timestamp>/report.md`
- `head-count=1`、`head-count-kv=1`

之所以固定 `head-count=1`，是因为 GEMM 专项 benchmark 不依赖 attention 头配置，而 odd hidden 场景如 `63/65` 会被默认的头数整除校验拦住。

## 5.1 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 5.2 单点 GEMM entry 测试

示例：`batch=1, seq_len=128, hidden=4096, gemm_out=16384`

```bash
./build/minfer_op_benchmark \
  --gemm-only \
  --warmup 10 \
  --iters 30 \
  --threads 32 \
  --head-count 1 \
  --head-count-kv 1 \
  --batch 1 \
  --seq-len 128 \
  --hidden 4096 \
  --gemm-out 16384
```

## 5.3 单点 micro-kernel 测试

```bash
./build/minfer_op_benchmark \
  --gemm-micro-only \
  --warmup 10 \
  --iters 30 \
  --threads 32 \
  --head-count 1 \
  --head-count-kv 1 \
  --batch 1 \
  --seq-len 1 \
  --hidden 4096 \
  --gemm-out 16384
```

## 5.4 推荐 sweep 列表

### Decode sweep

```bash
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 512  --gemm-out 2048
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 1024 --gemm-out 4096
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 2048 --gemm-out 8192
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 2 --seq-len 1 --hidden 2048 --gemm-out 8192
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 4 --seq-len 1 --hidden 4096 --gemm-out 16384
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 8 --seq-len 1 --hidden 4096 --gemm-out 16384
```

### Prefill sweep

```bash
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 16  --hidden 512  --gemm-out 2048
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 32  --hidden 1024 --gemm-out 4096
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 64  --hidden 2048 --gemm-out 8192
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 128 --hidden 2048 --gemm-out 8192
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 256 --hidden 4096 --gemm-out 16384
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 512 --hidden 4096 --gemm-out 16384
```

### Tail sweep

```bash
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 63  --hidden 63   --gemm-out 63
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 64  --hidden 64   --gemm-out 64
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 65  --hidden 65   --gemm-out 65
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 127 --hidden 4096 --gemm-out 4095
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 128 --hidden 4096 --gemm-out 4096
./build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 129 --hidden 4096 --gemm-out 4097
```

## 6. 记录模板

建议每次 sweep 以同样格式落表，便于横向比较：

| Scenario | Precision | avg ms | p50 ms | p95 ms | GFLOP/s | speedup vs fp32 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| decode_m1_h4096_ffn | FP32 |  |  |  |  | 1.00x |  |
| decode_m1_h4096_ffn | FP16 |  |  |  |  |  |  |
| decode_m1_h4096_ffn | INT8 |  |  |  |  |  |  |

如果有整模型验证，再追加：

| Scenario | Precision | Prefill tok/s | Decode ms/tok | Decode p50 | Decode p95 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| llama_65m_prompt128_decode128 | FP32 |  |  |  |  |  |
| llama_65m_prompt128_decode128 | FP16 |  |  |  |  |  |
| llama_65m_prompt128_decode128 | INT8 |  |  |  |  |  |

## 7. 验收口径

### 7.1 必须满足

- `Decode` 小 `M` 场景中，`FP16` 和 `INT8` 不能系统性慢于 `FP32`
- `Prefill` 大 `M` 场景中，blocked kernel 要稳定优于 simple path
- tail 邻域中，相邻尺寸吞吐波动应可控，不应出现明显断崖
- micro benchmark 的收益，要能在 `--gemm-only` 入口中保留下来

### 7.2 建议门槛

- `decode` 邻域：相邻形状吞吐或时延波动控制在 `10% ~ 15%`
- `prefill` 邻域：相邻形状吞吐波动控制在 `15%` 以内
- 同一形状下：
  - `FP16` 至少不低于 `FP32`
  - `INT8` 至少在大部分 `N >= K` 或 `N >> M` 场景中明显优于 `FP32`

## 8. 里程碑清单

## M0. 基线冻结

目标：得到可复现的 GEMM 基线结果。

交付物：

- 固定的 benchmark 命令集合
- 一版 `FP32 / FP16 / INT8` 基线数据
- 一份硬件 / 编译 / 线程配置说明

完成标准：

- `Decode`、`Prefill`、`Tail` 三类场景至少各跑完一轮
- 能稳定复现同一命令结果

## M1. Decode 优先优化

目标：先解决 `M=1/2/4/8` 的低延迟问题。

建议工作项：

- 强化 `RuntimeWeight::gemmNT()` 的 packed 路线
- 为小 `M` 场景建立单独 dispatch
- 分离 pack 成本与 compute 成本

完成标准：

- `decode_m1_*`、`decode_m2_*`、`decode_m4_*`、`decode_m8_*` 全部有结果
- 小 `M` 场景没有“换低精度反而更慢”的系统性问题

## M2. FP16 真正 packed 化

目标：让 `FP16` 路线使用真正面向 decode / prefill 的 packed 表示，而不是只做格式转换。

建议工作项：

- 检查 decode pack 是否仍有不必要的 `FP32` 化
- 为 `FP16` 单独维护 pack 与 row-packed kernel 路径
- 对比 `micro` 与 `entry` 两层收益保真度

完成标准：

- `FP16` 在 decode 与 prefill 主场景中都至少不弱于 `FP32`
- `FP16` 收益在 `--gemm-only` 中能够保留下来

## M3. Prefill 吞吐优化

目标：优化大 `M` 场景吞吐。

建议工作项：

- 系统扫描 `MC / NC / KC`
- 优化 OpenMP 切分与并行粒度
- 按 `M / N / K` 分桶，替代单阈值 heuristics

完成标准：

- `prefill_m64+` 场景吞吐稳定优于当前基线
- 大 `M` 场景无明显线程扩展异常

## M4. Tail / 非对齐健壮性

目标：消除维度轻微变化带来的性能断崖。

建议工作项：

- 专门审查 `N` tail、`K` tail、`M` tail
- 比较 simple path 与 blocked path 在边界点的切换质量
- 评估是否需要补若干专门的 remainder kernel

完成标准：

- `63/64/65`、`127/128/129`、`4095/4096/4097` 等邻域不出现大幅跳变

## M5. 性能门禁接入

目标：把 GEMM 从“人工观察”变成“可持续回归检测”。

建议工作项：

- 固化一组代表性 benchmark case
- 保存最近一次基线结果
- 后续每次 GEMM 相关修改都跑最小回归集

完成标准：

- 至少有一组 `decode`、一组 `prefill`、一组 `tail` 被纳入回归
- 文档中有明确的 pass / fail 标准

## 9. 推荐先做的最小闭环

如果只做一轮最有价值的工作，建议按下面顺序：

1. 跑完 `Decode 小 M + Prefill 吞吐` 两组基线
2. 修 `FP16` decode packed 路线
3. 建 `M/N/K` 分桶 dispatch
4. 再补 tail 邻域回归

这是最小、最稳、最容易反映到整模型体验上的闭环。
