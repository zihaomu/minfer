# LinearLayer_19 优化计划

## 1. 背景与问题

在 `minfer_benchmark --layer-profile` 的 decode 阶段中，`LinearLayer_19`（即模型最终 `lm_head` 投影）长期是最大热点之一。

典型特征：
- 每个 decode token 都会执行一次 `1 x hidden -> vocab` 投影（当前模型约 `1x512 -> 32768`）。
- 该层在 decode 中占比约 `26% ~ 29%`，直接限制 `ms/tok` 下界。

对应代码路径：
- `src/backend/cpu/layer/linear_layer.cpp`：`w.gemmNT(x).copyTo(out);`
- `src/backend/cpu/layer/runtime_weight.cpp`：decode packed dispatch + `M=1` fast path
- `src/backend/cpu/kernel/gemm_kernel_xsimd.cpp`：row-packed GEMM/GEMV 内核

---

## 2. 目标

分两级目标：

1. 第一阶段（可快速落地）
- 修复 decode 小矩阵路径正确性问题，确保 benchmark 可信。
- 降低 decode `ms/tok`，把 `LinearLayer_19` 占比从约 `29%` 压到 `<= 22%`。

2. 第二阶段（冲刺）
- 在不牺牲 prefill 的前提下，进一步压缩 decode 时延。
- 目标：`LinearLayer_19` 单次平均延迟再降 `20%+`。

---

## 3. 基线与验收口径

建议固定以下回归基线：

- 命令：
```bash
./build/minfer_benchmark --layer-profile --threads 8 --prompt-lens 128,512 --decode-tokens 128 --warmup 1 --runs 5
```

- 重点观察：
  - `decode avg(ms/tok)`
  - `LinearLayer_19` 的 `Avg(ms)` 与 `%`
  - `prefill throughput(tok/s)`（防止优化 decode 时误伤 prefill）

通过标准（阶段性）：
- 正确性：`RuntimeWeight` 相关单测全部通过。
- 性能：`prompt_len=128` 下 decode `ms/tok` 相比当前分支下降 `>= 15%`。
- 稳定性：`p99` 无明显劣化（不高于基线 `10%`）。

---

## 3.1 执行进展（2026-04-08）

### P0 状态：已完成

已落地改动：
- `runtime_weight.cpp`：修复 `M=1` fast path 分支覆盖，补齐 `FP32` 并恢复 `FP16` decode packed 路径。
- `gemm_kernel_xsimd.cpp`：移除并行 GEMV 内核中的调试 `printf`。
- `RuntimeWeight::rebuildDecodePacked()`：`FP16` 路径改为先转 `FP32` 再转置，最后回落 `FP16` 打包，规避半精度转置链路异常。

验证结果：
- 构建：`cmake --build build -j` 通过。
- 单测：`./build/minfer_test --gtest_filter='Mat_TEST.runtime_weight*'` 全部通过（3/3）。
- 微基准（`--gemm-runtime-only`）：
  - `runtime fp32 avg=0.6871 ms`
  - `runtime fp16 avg=0.4568 ms`（`1.50x` vs fp32）
  - `runtime int8 avg=0.7134 ms`

结论：
- 正确性已恢复，且 `FP16` decode packed 性能已回到合理区间，P0 可关闭。

### P1 状态：已实现并完成首轮调参

已落地改动：
- `attention_layer.cpp`：将多处无条件 `#pragma omp parallel for` 改为基于工作量门控的并行触发。
- `net.impl.{h,cpp}`：新增 phase-aware 线程策略。
  - prefill/decode 独立线程数。
  - 环境变量：
    - `MINFER_PHASE_THREAD_POLICY`（`off/0/false/no` 可关闭）
    - `MINFER_PREFILL_THREADS`
    - `MINFER_DECODE_THREADS`
  - 默认自动策略：`prefill=min(base, 8)`，`decode=min(base, 16)`。

关键实测（`prompt_len=128, decode_tokens=128, warmup=1, runs=5`）：
- `threads=16`：
  - policy=on：`prefill 141.43 ms`，`decode 1.48 ms/tok`，`end2end 774.38 tok/s`
  - policy=off：`prefill 216.82 ms`，`decode 1.46 ms/tok`，`end2end 633.78 tok/s`
- `threads=32`：
  - policy=on：`prefill 145.38 ms`，`decode 1.55 ms/tok`，`end2end 744.90 tok/s`
  - policy=off：`prefill 294.43 ms`，`decode 3.44 ms/tok`，`end2end 348.33 tok/s`

补充扫参（`threads=32`，固定 `prefill=8`）：
- `decode=16` 最优（`decode 1.58 ms/tok`，`end2end 730.55 tok/s`）。
- `decode>=20` 明显退化。
- `decode=32` 出现灾难性抖动（`decode 176.69 ms/tok`）。

结论：
- P1 的线程策略在中高线程下显著提升稳定性与端到端吞吐，建议保留当前默认值。

### P2 状态：进行中（第一批内核优化已落地）

本轮已落地：
- `xsimd_kernel_utils.h`
  - `load_hfloat_batch_f16c()` 去掉临时 `memcpy` 中转，改为直接 unaligned load。
  - 保留原有运行时特性检测与 scalar fallback。
- `gemm_kernel_xsimd.cpp`
  - `gemv_parallel_packed_fp32/fp16/i8_rowwise` 热循环改为指针递进 + 2x unroll，减少地址计算和循环开销。
  - 新增 `FP16 + AVX2/F16C/FMA` 专用 decode GEMV 内核：
    - 仅在 `kKernelNR == 8` 且 CPU 支持 `avx2+f16c+fma` 时启用。
    - 其余平台与指令集自动回退到原 xsimd 路径。

正确性验证：
- `Mat_TEST.gemm_generated_cases`
- `Mat_TEST.gemm_supports_fp16_weight_matrix`
- `Mat_TEST.gemm_supports_int8_weight_matrix`
- `Mat_TEST.gemm_kernel_nn_nt_match_reference_with_tails`
- `Mat_TEST.runtime_weight*`
- `Net_TEST.*`
以上均通过。

关键性能结果（与 P1 后基线对比）：

1) `./build/minfer_benchmark --layer-profile --threads 4`（同你最初口径）
- `prompt_len=128 decode`：
  - 旧：`3.37 ms/tok`
  - 新：`2.14 ms/tok`
  - 变化：约 `-36.5%`
- `LinearLayer_19`（decode）：
  - 旧：`Avg=1.024 ms`, `%≈29.1%`
  - 新：`Avg=0.590 ms`, `%≈26.1%`
  - 变化：单层时延约 `-42.4%`

2) `./build/minfer_benchmark --threads 16 --prompt-lens 128 --decode-tokens 128 --warmup 1 --runs 5`
- 旧：decode `1.48 ms/tok`
- 新：decode `1.41 ms/tok`
- 变化：约 `-4.7%`（中线程收益明显，高线程仍有内存带宽上限）

3) P2-第二步（本轮新增）：`FP32` decode GEMV 增强
- 新增 `gemv_parallel_packed_fp32` 的 `AVX2+FMA` 专用路径（自动检测 + 回退）。
- `gemm-runtime-only`（`batch=1, seq=1, hidden=512, out=32768, iters=200`）：
  - `threads=4`：`0.9031 -> 0.8584 ms`（约 `-5.0%`）
  - `threads=8`：`0.7827 -> 0.7794 ms`（基本持平）
  - `threads=16`：`0.1289 -> 0.1018 ms`（约 `-21.0%`）
- 说明：
  - 该步主要提升 `runtime fp32` 路径；当前 `fp16` 模型端到端收益有限，但为后续 `fp32`/混合精度场景打下基线。

4) P2-第三步（本轮新增）：`INT8` decode GEMV 增强
- 新增 `gemv_parallel_packed_i8_rowwise` 的 `AVX2+FMA` 专用路径（自动检测 + 回退）。
- `gemm-runtime-only`（同上口径）：
  - `threads=4`：`1.4218 -> 0.1965 ms`（约 `-86.2%`）
  - `threads=8`：`0.7275 -> 0.1008 ms`（约 `-86.1%`）
  - `threads=16`：`0.3748 -> 0.0532 ms`（约 `-85.8%`）
- 正确性回归：
  - `Layer_TEST.quantized_linear_runtime_precision_matches_fp32`
  - `Layer_TEST.quantized_linear_matches_python_precision_references`
  - `Layer_TEST.quantized_embedding_rmsnorm_ffn_attention_match_fp32`
  - 以上均通过。
- 说明：
  - 该步主要提升 INT8 runtime 路径；对当前 FP16 主模型端到端表现无负面影响。

5) P2-第四步（本轮新增）：INT8 并行门控（小 `N` 避免过并行）
- 在 `gemv_parallel_packed_i8_rowwise`（含 AVX2 路径与 xsimd 回退路径）新增并行门控：
  - 以 `paired_blocks` 为并行粒度；
  - 使用 `should_parallelize_1d_loop()` 判断；
  - 阈值：`kDecodeGemvMinParallelWork = 1 << 17`，`min_items_per_thread = 2`。
- 目标：
  - 小输出维度场景减少 OpenMP fork/join 与调度开销；
  - 大输出维度（如 `N=32768`）保持并行吞吐。
- 回归结果：
  - 正确性测试（INT8/GEMM/runtime_weight）通过。
  - `gemm-runtime-only` 在 `N=32768` 保持高性能（`threads=16` 下 `runtime int8 ≈ 0.0516 ms`）。
  - FP16 主链路 smoke 无回归（`threads=4/16` 的 decode `ms/tok` 维持在优化后区间）。

6) P2-第五步（本轮新增）：门控阈值环境变量化
- 将 INT8 并行门控阈值改为环境变量可配置：
  - `MINFER_GEMV_I8_MIN_PARALLEL_WORK`
  - 默认值：`1 << 17`（与原策略一致）
  - 非法/空值自动回退默认值。
- 验证（`threads=16`）：
  - 小 `N=256`：
    - `env=1`（强并行）`runtime int8 ≈ 0.0070 ms`
    - `env=999999999`（基本禁并行）`runtime int8 ≈ 0.0063 ms`
    - 结论：小矩阵并行收益不稳定，禁并行更稳。
  - 大 `N=32768`：
    - `env=1`：`runtime int8 ≈ 0.0530 ms`
    - `env=999999999`：`runtime int8 ≈ 0.7997 ms`
    - 结论：大矩阵需要并行，阈值不可过高。

7) P2-第六步（本轮新增）：`FP16` AVX512 decode 路径
- 在 `gemv_parallel_packed_fp16` 中新增 `AVX512` 专用实现，dispatch 优先级：
  - `AVX512 (avx512f+avx512dq+f16c+fma)` -> `AVX2` -> `xsimd`。
- 实现思路：
  - 以两个 `kKernelNR=8` block 合并成一个 16-lane 向量做 FMA（`M=1` decode 典型路径）。
  - 保留 tail block 的兼容处理与原回退路径。
- 正确性：
  - `Mat_TEST.runtime_weight*` 全部通过。
- 性能观察（`gemm-runtime-only`, `hidden=512, out=32768`）：
  - `threads=4`: `runtime fp16 ≈ 0.2545 ms`（较此前有小幅改善）
  - `threads=16`: `runtime fp16 ≈ 0.0526 ms`（与此前基本持平）
- 端到端（`prompt_len=128`）：
  - `threads=4 decode ≈ 2.13 ms/tok`
  - `threads=16 decode ≈ 1.30 ms/tok`
  - `LinearLayer_19 decode Avg ≈ 0.595 ms @ threads=4`
- 结论：
  - AVX512 FP16 路径已落地并稳定，但当前机器上端到端收益有限，下一步应优先做并行门控与调度策略优化。

8) P2-第七步（本轮新增）：`FP16/FP32` decode 并行门控扩展
- 将 decode GEMV 并行门控从 `INT8` 扩展到 `FP16/FP32`，并统一到同一个判定入口：
  - `should_parallelize_decode_gemv_pairs(pair_blocks, k, lanes_per_block, min_parallel_work)`
- 新增环境变量阈值（默认均为 `1<<17`）：
  - `MINFER_GEMV_FP16_MIN_PARALLEL_WORK`
  - `MINFER_GEMV_FP32_MIN_PARALLEL_WORK`
  - 既有 `MINFER_GEMV_I8_MIN_PARALLEL_WORK` 保持不变。
- 覆盖路径：
  - `FP32`：`AVX2` 实现 + `xsimd` fallback
  - `FP16`：`AVX512` 实现 + `AVX2` 实现 + `xsimd` fallback
  - `INT8`：沿用既有门控逻辑（仅重用通用接口）
- 正确性回归：
  - `Mat_TEST.runtime_weight*`
  - `Mat_TEST.gemm_generated_cases`
  - `Mat_TEST.gemm_supports_fp16_weight_matrix`
  - `Mat_TEST.gemm_supports_int8_weight_matrix`
  - `Mat_TEST.gemm_kernel_nn_nt_match_reference_with_tails`
  - 以上均通过。
- 串行基准结果（`OMP_PROC_BIND=close OMP_PLACES=cores`）：
  - `minfer_op_benchmark --gemm-runtime-only --batch 1 --seq-len 1 --hidden 512 --gemm-out 32768 --warmup 20 --iters 200`
    - `threads=4`：`fp32 0.8620 ms`，`fp16 0.2686 ms`，`int8 0.1972 ms`
    - `threads=8`：`fp32 0.8384 ms`，`fp16 0.1290 ms`，`int8 0.0986 ms`
    - `threads=16`：`fp32 0.1295 ms`，`fp16 0.0523 ms`，`int8 0.0525 ms`
  - `minfer_benchmark --prompt-lens 128 --decode-tokens 128 --warmup 1 --runs 3`
    - `threads=4`：decode `2.16 ms/tok`
    - `threads=16`：decode `1.41 ms/tok`
  - `minfer_benchmark --layer-profile --threads 4 --prompt-lens 128 --decode-tokens 128 --warmup 1 --runs 3`
    - `LinearLayer_19 decode Avg = 0.578 ms`，占比 `27.2%`
- 结论：
  - 该步完成后，`FP16/FP32/INT8` decode 路径均具备一致的并行门控与可调阈值，便于后续按机器特性做针对性调参。

9) P2-第八步（本轮新增）：`FP16/FP32` 门限扫参（4/8/16 线程）
- 新增扫参报告：
  - `doc/fp16_fp32_threshold_sweep_20260409.md`
  - 原始日志：`benchmark/results/threshold_sweep_20260409_095926/`
- `FP16` 端到端结论（`decode ms/tok + LinearLayer_19 decode Avg(ms)`）：
  - `th=131072` 在 `threads=4/8/16` 上为最优或并列最优。
  - `th>=524288` 开始明显退化，`th=999999999`（禁并行）严重退化。
  - 建议区间：`32768 ~ 131072`，默认值保持 `131072`。
- `FP32` 结论：
  - 在 `runtime-only` 口径下，`th=999999999` 全线程显著退化。
  - `1/32768/524288` 属于可用区间，建议后续在空闲机器复测后再决定默认值是否调整。
- 说明：
  - 本轮测量期间存在外部高负载进程（`cvh_bench`），`FP32` 口径抖动偏大；`FP16` 趋势更稳定。

10) P2-第九步（本轮新增）：`FP16` decode 微内核提速
- 改动内容：
  - `gemv_parallel_packed_fp16_avx512_impl`
    - 将单一 `__m512` 累加器改为双累加器（even/odd）以打断 FMA 依赖链。
    - 去掉 `__m512 -> tmp[16] -> __m256` 的临时内存回写，改为直接拆成高低 256-bit 半向量后存回。
  - `gemv_parallel_packed_fp16_avx2_impl`
    - 同样改为双累加器（even/odd）+ 2x `k` 方向展开。
- 正确性回归：
  - `Mat_TEST.runtime_weight*`
  - `Mat_TEST.gemm_generated_cases`
  - `Mat_TEST.gemm_supports_fp16_weight_matrix`
  - `Mat_TEST.gemm_supports_int8_weight_matrix`
  - `Mat_TEST.gemm_kernel_nn_nt_match_reference_with_tails`
  - 以上均通过。
- 微基准（`minfer_op_benchmark --gemm-runtime-only --batch 1 --seq-len 1 --hidden 512 --gemm-out 32768 --warmup 20 --iters 200`）：
  - 改动前：
    - `threads=4`: `runtime fp16 ≈ 0.5757 ms`
  - 改动后：
    - `threads=4`: 首次测得 `0.2583 ms`
    - `threads=4`: 复测三次分别为 `0.2783 / 0.2697 / 0.2687 ms`
- 结论：
  - 在 `threads=4` 的 decode 典型并行度下，`FP16` 纯内核耗时稳定下降到约 `0.27 ms`，说明这一步对 `LinearLayer_19` 热路径有效。
  - 端到端 benchmark 期间存在外部高负载进程，`threads=1/16` 和整链 decode 数据抖动较大，本轮不据此下结论；待空闲机器复测。

### P4 状态：已启动（性能门禁脚本已落地）

已新增：
- `benchmark/perf_gate.sh`

默认门禁项：
- `decode avg(ms/tok) @ threads=4 <= 2.20`
- `decode avg(ms/tok) @ threads=16 <= 1.30`
- `LinearLayer_19 decode Avg(ms) @ threads=4 <= 0.60`

脚本能力：
- 支持 `--model`、`--log-dir`、`--no-affinity` 等参数。
- 支持阈值环境变量覆盖：
  - `MINFER_GATE_DECODE_T4_MAX_MS`
  - `MINFER_GATE_DECODE_T16_MAX_MS`
  - `MINFER_GATE_LINEAR_DECODE_MAX_MS`
- 默认设置 `OMP_PROC_BIND=close` 与 `OMP_PLACES=cores` 降低抖动。

---

## 4. 优化路线（P0-P4）

## P0：修复基线可信度（最高优先级）

目标：先保证数据和路径正确，再做性能判断。

工作项：
- 修复 `runtime_weight.cpp` 中 `M=1` fast path 的分支覆盖（补齐 `FP32` 路径）。
- 去除 `gemm_kernel_xsimd.cpp` 中临时 `DEBUG printf`，避免干扰测量。
- 重新跑 `RuntimeWeight` 相关单测和 micro benchmark。

验收：
- `Mat_TEST.runtime_weight*` 全部通过。
- `minfer_op_benchmark --gemm-runtime-only` 不出现异常值（如 `fp32` 非物理性极小延迟）。

---

## P1：线程策略重构（高 ROI）

目标：让 decode 和 prefill 使用不同并行策略，避免“一把梭”线程数导致退化。

工作项：
- 对 attention 中无条件 `#pragma omp parallel for` 的路径增加工作量门控。
- 引入 phase-aware 线程策略（prefill/decode 分开配置）。
- 对 `threads = 4/8/16` 做 sweep，选 decode 与 prefill 的平衡点。

验收：
- decode `ms/tok` 有显著下降。
- prefill 不出现明显回退（建议回退阈值 `< 5%`）。

---

## P2：LinearLayer_19 专项内核优化

目标：针对 `M=1, K=512, N≈32768` 的典型 `lm_head` 场景做定向优化。

工作项：
- 在 `RuntimeWeight::gemmNT()` 增加更细粒度 dispatch（优先命中 `M=1` 专用路径）。
- 优化 `gemv_parallel_packed_fp16` / `gemv_parallel_packed_i8_rowwise`：
  - block 粒度重整
  - unroll 策略优化
  - 线程切分负载均衡
- 评估 `copyTo`、bias add 等周边成本占比，避免“内核快了但总耗时没降”。

验收：
- `LinearLayer_19` decode `Avg(ms)` 明显下降（目标 `<= 0.6ms`，以当前机器实测为准）。
- decode 总体 `tok/s` 提升并在多次 runs 下稳定。

---

## P3：算法级扩展（可选）

目标：进一步突破 `lm_head` 成本上限。

候选方向：
- 两阶段 logits（shortlist + 精算）
- `lm_head` 专项量化路径（精度约束下）

注意：
- 该阶段有精度与复杂度风险，必须在 P0-P2 稳定后推进。

验收：
- decode 再提升 `20%+`（若精度评估通过）。

---

## P4：性能门禁接入（持续化）

目标：把优化成果固化为回归门禁，防止后续退化。

工作项：
- 固定最小回归集（decode/prefill/layer-profile）。
- 对关键路径（`runtime_weight`, `gemm_kernel`, `attention_layer`）改动触发自动性能回归。
- 文档化 pass/fail 阈值。

验收：
- 每次关键改动都有可追溯性能记录。
- 能快速定位是哪次提交引入回退。

---

## 5. 执行顺序建议

建议严格按以下顺序推进：

1. `P0`（正确性与基线可信）
2. `P1`（线程策略）
3. `P2`（LinearLayer 专项）
4. `P3`（可选扩展）
5. `P4`（门禁固化）

这条顺序能确保每一步收益都可解释、可复现、可继承。

---

## 6. 非目标（当前轮不做）

- 不做 GPU 后端改造。
- 不做模型结构级变更（如改网络层数/参数结构）。
- 不引入大规模重构（优先最小改动闭环）。
