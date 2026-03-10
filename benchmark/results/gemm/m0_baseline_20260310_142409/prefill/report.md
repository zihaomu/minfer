# GEMM Benchmark Report

## Context

- Timestamp: 2026-03-10T14:26:38+08:00
- Commit: df860ae223adf12fbe4f0e3631ff3641a1030f89
- CPU: AMD Ryzen 9 7950X 16-Core Processor
- Build: Release
- Binary: /home/moo/work/my_lab/minfer/build/minfer_op_benchmark
- Group: prefill
- Mode: both
- Threads: 32
- Warmup: 10
- Iters: 30
- Filter: 

## Results

### prefill_m16_h512_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 16 --hidden 512 --gemm-out 2048 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/entry_prefill_prefill_m16_h512_ffn.log

```text
gemm nt fp32         avg=1.0386 ms p50=1.0362 ms p95=1.0579 ms throughput=32.31 GFLOP/s speedup=1.00x
gemm nt fp16         avg=1.7232 ms p50=1.7173 ms p95=1.7487 ms throughput=19.47 GFLOP/s speedup=0.60x
gemm nt int8         avg=1.2377 ms p50=1.2373 ms p95=1.2453 ms throughput=27.11 GFLOP/s speedup=0.84x
```

### prefill_m16_h512_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 16 --hidden 512 --gemm-out 2048 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/micro_prefill_prefill_m16_h512_ffn.log

```text
nt entry fp32        avg=1.0382 ms p50=1.0372 ms p95=1.0482 ms throughput=32.32 GFLOP/s speedup=1.00x
nt entry fp16        avg=1.6439 ms p50=1.6480 ms p95=1.7178 ms throughput=20.41 GFLOP/s speedup=0.63x
nt entry int8        avg=1.1189 ms p50=1.1138 ms p95=1.1664 ms throughput=29.99 GFLOP/s speedup=0.93x
rowpacked fp32       avg=1.1618 ms p50=1.1568 ms p95=1.1995 ms throughput=28.88 GFLOP/s speedup=1.00x
rowpacked fp16       avg=4.5782 ms p50=4.5630 ms p95=4.6114 ms throughput=7.33 GFLOP/s speedup=0.25x
rowpacked int8       avg=5.8016 ms p50=5.7485 ms p95=6.0607 ms throughput=5.78 GFLOP/s speedup=0.20x
```

### prefill_m32_h1024_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 32 --hidden 1024 --gemm-out 4096 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/entry_prefill_prefill_m32_h1024_ffn.log

```text
gemm nt fp32         avg=7.0929 ms p50=7.0035 ms p95=7.7155 ms throughput=37.85 GFLOP/s speedup=1.00x
gemm nt fp16         avg=8.9186 ms p50=8.9100 ms p95=9.2435 ms throughput=30.10 GFLOP/s speedup=0.80x
gemm nt int8         avg=7.0186 ms p50=7.0142 ms p95=7.0592 ms throughput=38.25 GFLOP/s speedup=1.01x
```

### prefill_m32_h1024_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 32 --hidden 1024 --gemm-out 4096 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/micro_prefill_prefill_m32_h1024_ffn.log

```text
nt entry fp32        avg=6.6290 ms p50=6.6349 ms p95=6.7229 ms throughput=40.49 GFLOP/s speedup=1.00x
nt entry fp16        avg=8.7159 ms p50=8.6993 ms p95=8.9537 ms throughput=30.80 GFLOP/s speedup=0.76x
nt entry int8        avg=7.4282 ms p50=7.2930 ms p95=8.0660 ms throughput=36.14 GFLOP/s speedup=0.89x
rowpacked fp32       avg=9.9921 ms p50=9.8680 ms p95=10.5290 ms throughput=26.86 GFLOP/s speedup=1.00x
rowpacked fp16       avg=36.8205 ms p50=36.8086 ms p95=36.9089 ms throughput=7.29 GFLOP/s speedup=0.27x
rowpacked int8       avg=45.7736 ms p50=45.5172 ms p95=47.3018 ms throughput=5.86 GFLOP/s speedup=0.22x
```

### prefill_m64_h2048_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 64 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/entry_prefill_prefill_m64_h2048_ffn.log

```text
gemm nt fp32         avg=5.5402 ms p50=5.4576 ms p95=6.4715 ms throughput=387.62 GFLOP/s speedup=1.00x
gemm nt fp16         avg=6.0574 ms p50=6.1042 ms p95=6.8369 ms throughput=354.52 GFLOP/s speedup=0.91x
gemm nt int8         avg=5.4741 ms p50=5.4090 ms p95=6.1130 ms throughput=392.30 GFLOP/s speedup=1.01x
```

### prefill_m64_h2048_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 64 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/micro_prefill_prefill_m64_h2048_ffn.log

```text
nt entry fp32        avg=4.5628 ms p50=4.2800 ms p95=7.7331 ms throughput=470.65 GFLOP/s speedup=1.00x
nt entry fp16        avg=5.3605 ms p50=4.2760 ms p95=9.0892 ms throughput=400.61 GFLOP/s speedup=0.85x
nt entry int8        avg=3.8246 ms p50=3.5983 ms p95=4.8574 ms throughput=561.49 GFLOP/s speedup=1.19x
rowpacked fp32       avg=95.1510 ms p50=95.3819 ms p95=98.4144 ms throughput=22.57 GFLOP/s speedup=1.00x
rowpacked fp16       avg=296.7185 ms p50=296.5270 ms p95=298.1664 ms throughput=7.24 GFLOP/s speedup=0.32x
rowpacked int8       avg=363.1230 ms p50=361.4832 ms p95=373.5620 ms throughput=5.91 GFLOP/s speedup=0.26x
```

### prefill_m128_h2048_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 128 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/entry_prefill_prefill_m128_h2048_ffn.log

```text
gemm nt fp32         avg=8.6881 ms p50=8.5994 ms p95=9.7465 ms throughput=494.35 GFLOP/s speedup=1.00x
gemm nt fp16         avg=9.1929 ms p50=9.0741 ms p95=10.3274 ms throughput=467.20 GFLOP/s speedup=0.95x
gemm nt int8         avg=8.6733 ms p50=8.4678 ms p95=10.3638 ms throughput=495.19 GFLOP/s speedup=1.00x
```

### prefill_m128_h2048_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 128 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/micro_prefill_prefill_m128_h2048_ffn.log

```text
nt entry fp32        avg=7.3348 ms p50=6.8451 ms p95=11.8130 ms throughput=585.56 GFLOP/s speedup=1.00x
nt entry fp16        avg=7.0093 ms p50=6.7721 ms p95=8.4802 ms throughput=612.75 GFLOP/s speedup=1.05x
nt entry int8        avg=9.8980 ms p50=7.7986 ms p95=18.4107 ms throughput=433.92 GFLOP/s speedup=0.74x
rowpacked fp32       avg=188.1909 ms p50=188.6871 ms p95=191.4672 ms throughput=22.82 GFLOP/s speedup=1.00x
rowpacked fp16       avg=593.0431 ms p50=592.6035 ms p95=596.0685 ms throughput=7.24 GFLOP/s speedup=0.32x
rowpacked int8       avg=724.3879 ms p50=724.1233 ms p95=728.1272 ms throughput=5.93 GFLOP/s speedup=0.26x
```

### prefill_m256_h4096_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 256 --hidden 4096 --gemm-out 16384 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/entry_prefill_prefill_m256_h4096_ffn.log

```text
gemm nt fp32         avg=51.6970 ms p50=51.2639 ms p95=55.6290 ms throughput=664.64 GFLOP/s speedup=1.00x
gemm nt fp16         avg=53.9428 ms p50=54.1027 ms p95=58.7637 ms throughput=636.97 GFLOP/s speedup=0.96x
gemm nt int8         avg=54.9117 ms p50=55.8495 ms p95=58.7066 ms throughput=625.73 GFLOP/s speedup=0.94x
```

### prefill_m256_h4096_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 256 --hidden 4096 --gemm-out 16384 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/micro_prefill_prefill_m256_h4096_ffn.log

```text
nt entry fp32        avg=50.0402 ms p50=48.5224 ms p95=56.2191 ms throughput=686.64 GFLOP/s speedup=1.00x
nt entry fp16        avg=51.6915 ms p50=50.3987 ms p95=57.6718 ms throughput=664.71 GFLOP/s speedup=0.97x
nt entry int8        avg=50.7751 ms p50=49.9904 ms p95=55.8710 ms throughput=676.71 GFLOP/s speedup=0.99x
rowpacked fp32       avg=1623.1364 ms p50=1611.3933 ms p95=1703.8810 ms throughput=21.17 GFLOP/s speedup=1.00x
rowpacked fp16       avg=4752.3618 ms p50=4749.2361 ms p95=4768.9757 ms throughput=7.23 GFLOP/s speedup=0.34x
rowpacked int8       avg=5864.6035 ms p50=5857.5913 ms p95=5922.9923 ms throughput=5.86 GFLOP/s speedup=0.28x
```

### prefill_m512_h4096_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 512 --hidden 4096 --gemm-out 16384 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/entry_prefill_prefill_m512_h4096_ffn.log

```text
gemm nt fp32         avg=106.9672 ms p50=105.5967 ms p95=114.2071 ms throughput=642.43 GFLOP/s speedup=1.00x
gemm nt fp16         avg=108.4386 ms p50=107.4234 ms p95=117.3094 ms throughput=633.72 GFLOP/s speedup=0.99x
gemm nt int8         avg=112.8883 ms p50=111.1436 ms p95=122.6263 ms throughput=608.74 GFLOP/s speedup=0.95x
```

### prefill_m512_h4096_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 512 --hidden 4096 --gemm-out 16384
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/prefill/micro_prefill_prefill_m512_h4096_ffn.log

```text
nt entry fp32        avg=97.8159 ms p50=97.4257 ms p95=104.7654 ms throughput=702.54 GFLOP/s speedup=1.00x
nt entry fp16        avg=98.8872 ms p50=98.5422 ms p95=103.0943 ms throughput=694.93 GFLOP/s speedup=0.99x
nt entry int8        avg=98.1047 ms p50=96.6166 ms p95=104.8448 ms throughput=700.47 GFLOP/s speedup=1.00x
rowpacked fp32       avg=3126.4929 ms p50=3104.3214 ms p95=3252.4037 ms throughput=21.98 GFLOP/s speedup=1.00x
rowpacked fp16       avg=9504.0813 ms p50=9497.5296 ms p95=9532.4896 ms throughput=7.23 GFLOP/s speedup=0.33x
rowpacked int8       avg=11738.8486 ms p50=11724.4662 ms p95=11808.3656 ms throughput=5.85 GFLOP/s speedup=0.27x
```
