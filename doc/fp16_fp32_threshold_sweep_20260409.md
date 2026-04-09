# FP16/FP32 Decode 门限扫参（2026-04-09）

## 1. 目标

验证并收敛以下两个门限的推荐区间（`threads=4/8/16`）：

- `MINFER_GEMV_FP16_MIN_PARALLEL_WORK`
- `MINFER_GEMV_FP32_MIN_PARALLEL_WORK`

重点指标：

- `decode avg(ms/tok)`（`minfer_benchmark --layer-profile`）
- `LinearLayer_19` decode `Avg(ms)`

补充指标（FP32 路径）：

- `runtime fp32 avg(ms)`（`minfer_op_benchmark --gemm-runtime-only`）

---

## 2. 扫参口径

阈值点：

- `1`
- `32768` (`1<<15`)
- `131072` (`1<<17`)
- `524288` (`1<<19`)
- `999999999`（近似禁并行）

固定参数：

```bash
OMP_PROC_BIND=close OMP_PLACES=cores
./build/minfer_benchmark --layer-profile --threads {4|8|16} --prompt-lens 128 --decode-tokens 128 --warmup 1 --runs 3
./build/minfer_op_benchmark --gemm-runtime-only --batch 1 --seq-len 1 --hidden 512 --gemm-out 32768 --warmup 20 --iters 200 --threads {4|8|16}
```

原始日志目录：

- `benchmark/results/threshold_sweep_20260409_095926/`

---

## 3. 结果摘要

### 3.1 FP16 门限对端到端（高置信）

来自 `summary_fp16.csv`：

- `th=131072` 在 `threads=4/8/16` 上均为最优或并列最优。
- `th=524288` 开始出现明显退化（decode 约 `+18% ~ +26%`）。
- `th=999999999`（禁并行）严重退化（decode 约 `+100% ~ +211%`，`LinearLayer_19` 约 `+101% ~ +307%`）。

结论：

- 推荐区间：`32768 ~ 131072`
- 默认值建议：保持 `131072`（当前默认）

### 3.2 FP32 门限（runtime-only 参考）

来自 `summary_fp32_runtime.csv`：

- `th=999999999` 在全部线程显著退化（约 `+102% ~ +1491%`）。
- `th=1/32768/524288` 存在线程相关优劣差异，整体属于可用区间。
- `th=131072` 在本轮数据下不是最优点（但仍远好于禁并行）。

结论（暂定）：

- 可用区间：`1 ~ 524288`
- 保守推荐区间：`32768 ~ 524288`
- `999999999` 不可作为默认值

---

## 4. 干扰与置信度说明

本轮测量期间系统存在持续高负载进程：

- `./build-full-test/cvh_bench`（`%CPU` 约 `400%+`）

影响：

- FP32 相关口径抖动增大，尤其端到端 decode 指标。

建议：

- 在空闲机器复测一次 FP32 门限（同口径）后再考虑是否调整默认值。
- FP16 默认值 `131072` 当前可先保持，不阻塞后续优化。

