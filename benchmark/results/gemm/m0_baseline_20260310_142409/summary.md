# GEMM M0 Baseline Summary

## Context

- Timestamp: 2026-03-10
- Commit: `df860ae223adf12fbe4f0e3631ff3641a1030f89`
- CPU: `AMD Ryzen 9 7950X 16-Core Processor`
- Build: `Release`
- Threads: `32`
- Warmup: `10`
- Iters: `30`

Reports:

- `decode/report.md`
- `prefill/report.md`
- `tail/report.md`

## Key Findings

### Decode

- The new `runtime` path is materially better than plain `gemm` entry for small `M`.
- `decode_m1_h512_ffn`: `runtime fp32 = 24.86 GFLOP/s` vs `gemm fp32 = 14.56 GFLOP/s`
- `decode_m1_h1024_ffn`: `runtime fp32 = 28.28 GFLOP/s` vs `gemm fp32 = 14.50 GFLOP/s`
- `decode_m1_h2048_ffn`: `runtime fp32 = 22.85 GFLOP/s` vs `gemm fp32 = 14.50 GFLOP/s`
- `decode_m4_h4096_ffn`: `runtime fp32 = 21.37 GFLOP/s` vs `gemm fp32 = 9.95 GFLOP/s`
- `decode_m8_h4096_ffn`: `runtime fp32 = 21.40 GFLOP/s` vs `gemm fp32 = 9.87 GFLOP/s`
- `runtime fp16` is roughly flat with `runtime fp32`, not clearly better yet.
- `runtime int8` remains bandwidth-limited and does not beat `fp32` on these decode cases.

### Prefill

- For large `M`, blocked `nt entry` is the only competitive path.
- `prefill_m64_h2048_ffn`: `nt entry fp32 = 470.65 GFLOP/s`, `rowpacked fp32 = 22.57 GFLOP/s`
- `prefill_m128_h2048_ffn`: `nt entry fp32 = 585.56 GFLOP/s`, `rowpacked fp32 = 22.82 GFLOP/s`
- `prefill_m256_h4096_ffn`: `nt entry fp32 = 686.64 GFLOP/s`, `rowpacked fp32 = 21.17 GFLOP/s`
- `prefill_m512_h4096_ffn`: `nt entry fp32 = 702.54 GFLOP/s`, `rowpacked fp32 = 21.98 GFLOP/s`
- `fp16` and `int8` are close to `fp32` on large prefill shapes, but not decisively better.

### Tail

- Small odd sizes are stable: `63 / 64 / 65` `gemm fp32` stays around `32.8 ~ 34.3 GFLOP/s`.
- Large prefill tail is also stable:
  - `127 x 4096 x 4095`: `49.20 GFLOP/s`
  - `128 x 4096 x 4096`: `50.68 GFLOP/s`
  - `129 x 4096 x 4097`: `49.21 GFLOP/s`
- Decode tail benefits from packed runtime-style micro kernels:
  - `1 x 4096 x 4095`: `rowpacked fp32 = 19.64 GFLOP/s`
  - `1 x 4096 x 4096`: `rowpacked fp32 = 22.61 GFLOP/s`
  - `1 x 4096 x 4097`: `rowpacked fp32 = 22.39 GFLOP/s`

## Actionable Conclusion

- `M0` is now frozen with reproducible command sets, per-group reports, and environment metadata.
- `M1` moved the runtime decode path into the right regime: small `M` should use packed runtime GEMM, not plain `gemm()` entry.
- The next highest-value work item is `M2`: keep the same small-`M` dispatch, but stop routing `FP16` decode pack through `FP32` packed storage.
