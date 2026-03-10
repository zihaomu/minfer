# GEMM Benchmark Report

## Context

- Timestamp: 2026-03-10T15:04:02+08:00
- Commit: df860ae223adf12fbe4f0e3631ff3641a1030f89
- CPU: AMD Ryzen 9 7950X 16-Core Processor
- Build: Release
- Binary: /home/moo/work/my_lab/minfer/build/minfer_op_benchmark
- Group: tail
- Mode: both
- Threads: 32
- Warmup: 10
- Iters: 30
- Filter: 

## Results

### tail_small_63_63_63 (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 63 --hidden 63 --gemm-out 63 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/entry_tail_tail_small_63_63_63.log

```text
gemm nt fp32         avg=0.0146 ms p50=0.0140 ms p95=0.0192 ms throughput=34.28 GFLOP/s speedup=1.00x
gemm nt fp16         avg=0.0164 ms p50=0.0164 ms p95=0.0164 ms throughput=30.49 GFLOP/s speedup=0.89x
gemm nt int8         avg=0.0152 ms p50=0.0149 ms p95=0.0170 ms throughput=32.91 GFLOP/s speedup=0.96x
```

### tail_small_63_63_63 (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 63 --hidden 63 --gemm-out 63 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/micro_tail_tail_small_63_63_63.log

```text
nt entry fp32        avg=0.0130 ms p50=0.0130 ms p95=0.0130 ms throughput=38.52 GFLOP/s speedup=1.00x
nt entry fp16        avg=0.0168 ms p50=0.0157 ms p95=0.0211 ms throughput=29.73 GFLOP/s speedup=0.77x
nt entry int8        avg=0.0151 ms p50=0.0142 ms p95=0.0208 ms throughput=33.19 GFLOP/s speedup=0.86x
rowpacked fp32       avg=0.0170 ms p50=0.0170 ms p95=0.0170 ms throughput=29.42 GFLOP/s speedup=1.00x
rowpacked fp16       avg=0.0730 ms p50=0.0725 ms p95=0.0767 ms throughput=6.85 GFLOP/s speedup=0.23x
rowpacked int8       avg=0.0968 ms p50=0.0957 ms p95=0.1014 ms throughput=5.17 GFLOP/s speedup=0.18x
```

### tail_small_64_64_64 (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 64 --hidden 64 --gemm-out 64 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/entry_tail_tail_small_64_64_64.log

```text
gemm nt fp32         avg=0.0156 ms p50=0.0136 ms p95=0.0320 ms throughput=33.60 GFLOP/s speedup=1.00x
gemm nt fp16         avg=0.0163 ms p50=0.0163 ms p95=0.0163 ms throughput=32.23 GFLOP/s speedup=0.96x
gemm nt int8         avg=0.0154 ms p50=0.0147 ms p95=0.0148 ms throughput=34.00 GFLOP/s speedup=1.01x
```

### tail_small_64_64_64 (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 64 --hidden 64 --gemm-out 64 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/micro_tail_tail_small_64_64_64.log

```text
nt entry fp32        avg=0.0130 ms p50=0.0130 ms p95=0.0131 ms throughput=40.19 GFLOP/s speedup=1.00x
nt entry fp16        avg=0.0157 ms p50=0.0156 ms p95=0.0157 ms throughput=33.50 GFLOP/s speedup=0.83x
nt entry int8        avg=0.0142 ms p50=0.0141 ms p95=0.0141 ms throughput=36.96 GFLOP/s speedup=0.92x
rowpacked fp32       avg=0.0178 ms p50=0.0174 ms p95=0.0222 ms throughput=29.49 GFLOP/s speedup=1.00x
rowpacked fp16       avg=0.0753 ms p50=0.0745 ms p95=0.0818 ms throughput=6.96 GFLOP/s speedup=0.24x
rowpacked int8       avg=0.0996 ms p50=0.0983 ms p95=0.1076 ms throughput=5.27 GFLOP/s speedup=0.18x
```

### tail_small_65_65_65 (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 65 --hidden 65 --gemm-out 65 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/entry_tail_tail_small_65_65_65.log

```text
gemm nt fp32         avg=0.0168 ms p50=0.0163 ms p95=0.0195 ms throughput=32.79 GFLOP/s speedup=1.00x
gemm nt fp16         avg=0.0194 ms p50=0.0190 ms p95=0.0210 ms throughput=28.27 GFLOP/s speedup=0.86x
gemm nt int8         avg=0.0177 ms p50=0.0174 ms p95=0.0177 ms throughput=31.00 GFLOP/s speedup=0.95x
```

### tail_small_65_65_65 (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 65 --hidden 65 --gemm-out 65 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/micro_tail_tail_small_65_65_65.log

```text
nt entry fp32        avg=0.0157 ms p50=0.0157 ms p95=0.0157 ms throughput=35.00 GFLOP/s speedup=1.00x
nt entry fp16        avg=0.0184 ms p50=0.0184 ms p95=0.0186 ms throughput=29.90 GFLOP/s speedup=0.85x
nt entry int8        avg=0.0169 ms p50=0.0167 ms p95=0.0182 ms throughput=32.53 GFLOP/s speedup=0.93x
rowpacked fp32       avg=0.0208 ms p50=0.0204 ms p95=0.0211 ms throughput=26.41 GFLOP/s speedup=1.00x
rowpacked fp16       avg=0.0864 ms p50=0.0859 ms p95=0.0892 ms throughput=6.36 GFLOP/s speedup=0.24x
rowpacked int8       avg=0.1145 ms p50=0.1141 ms p95=0.1168 ms throughput=4.80 GFLOP/s speedup=0.18x
```

### tail_prefill_127_4096_4095 (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 127 --hidden 4096 --gemm-out 4095 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/entry_tail_tail_prefill_127_4096_4095.log

```text
gemm nt fp32         avg=86.6002 ms p50=86.3704 ms p95=87.9312 ms throughput=49.20 GFLOP/s speedup=1.00x
gemm nt fp16         avg=94.4160 ms p50=94.2743 ms p95=95.3369 ms throughput=45.12 GFLOP/s speedup=0.92x
gemm nt int8         avg=88.7806 ms p50=88.6528 ms p95=89.7566 ms throughput=47.99 GFLOP/s speedup=0.98x
```

### tail_prefill_127_4096_4095 (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 127 --hidden 4096 --gemm-out 4095 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/micro_tail_tail_prefill_127_4096_4095.log

```text
nt entry fp32        avg=85.8743 ms p50=85.7538 ms p95=87.1648 ms throughput=49.61 GFLOP/s speedup=1.00x
nt entry fp16        avg=93.9133 ms p50=93.8329 ms p95=94.4929 ms throughput=45.36 GFLOP/s speedup=0.91x
nt entry int8        avg=88.8050 ms p50=88.6819 ms p95=89.8861 ms throughput=47.97 GFLOP/s speedup=0.97x
rowpacked fp32       avg=193.4263 ms p50=190.2445 ms p95=205.0662 ms throughput=22.03 GFLOP/s speedup=1.00x
rowpacked fp16       avg=589.5755 ms p50=589.3614 ms p95=591.9257 ms throughput=7.23 GFLOP/s speedup=0.33x
rowpacked int8       avg=724.9140 ms p50=724.8083 ms p95=728.0494 ms throughput=5.88 GFLOP/s speedup=0.27x
```

### tail_prefill_128_4096_4096 (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 128 --hidden 4096 --gemm-out 4096 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/entry_tail_tail_prefill_128_4096_4096.log

```text
gemm nt fp32         avg=84.7465 ms p50=84.7045 ms p95=85.0808 ms throughput=50.68 GFLOP/s speedup=1.00x
gemm nt fp16         avg=93.7895 ms p50=93.8092 ms p95=94.5515 ms throughput=45.79 GFLOP/s speedup=0.90x
gemm nt int8         avg=88.3328 ms p50=88.0940 ms p95=89.7053 ms throughput=48.62 GFLOP/s speedup=0.96x
```

### tail_prefill_128_4096_4096 (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 128 --hidden 4096 --gemm-out 4096 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/micro_tail_tail_prefill_128_4096_4096.log

```text
nt entry fp32        avg=84.5999 ms p50=84.4648 ms p95=85.2269 ms throughput=50.77 GFLOP/s speedup=1.00x
nt entry fp16        avg=93.4952 ms p50=93.2998 ms p95=94.7065 ms throughput=45.94 GFLOP/s speedup=0.90x
nt entry int8        avg=87.9697 ms p50=87.8263 ms p95=88.3856 ms throughput=48.82 GFLOP/s speedup=0.96x
rowpacked fp32       avg=193.4221 ms p50=192.4077 ms p95=198.1701 ms throughput=22.21 GFLOP/s speedup=1.00x
rowpacked fp16       avg=597.3463 ms p50=597.1094 ms p95=598.8265 ms throughput=7.19 GFLOP/s speedup=0.32x
rowpacked int8       avg=731.0697 ms p50=730.5521 ms p95=735.7477 ms throughput=5.87 GFLOP/s speedup=0.26x
```

### tail_prefill_129_4096_4097 (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 129 --hidden 4096 --gemm-out 4097 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/entry_tail_tail_prefill_129_4096_4097.log

```text
gemm nt fp32         avg=87.9795 ms p50=87.8406 ms p95=88.5887 ms throughput=49.21 GFLOP/s speedup=1.00x
gemm nt fp16         avg=96.4338 ms p50=96.4246 ms p95=96.7013 ms throughput=44.90 GFLOP/s speedup=0.91x
gemm nt int8         avg=91.2014 ms p50=91.0798 ms p95=91.8830 ms throughput=47.47 GFLOP/s speedup=0.96x
```

### tail_prefill_129_4096_4097 (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 129 --hidden 4096 --gemm-out 4097 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/micro_tail_tail_prefill_129_4096_4097.log

```text
nt entry fp32        avg=88.0082 ms p50=87.9476 ms p95=88.6327 ms throughput=49.20 GFLOP/s speedup=1.00x
nt entry fp16        avg=98.2497 ms p50=97.8840 ms p95=102.3379 ms throughput=44.07 GFLOP/s speedup=0.90x
nt entry int8        avg=92.0827 ms p50=91.3437 ms p95=95.1649 ms throughput=47.02 GFLOP/s speedup=0.96x
rowpacked fp32       avg=197.7951 ms p50=193.4825 ms p95=225.1616 ms throughput=21.89 GFLOP/s speedup=1.00x
rowpacked fp16       avg=601.1790 ms p50=600.7405 ms p95=604.2801 ms throughput=7.20 GFLOP/s speedup=0.33x
rowpacked int8       avg=735.4270 ms p50=735.4171 ms p95=739.3730 ms throughput=5.89 GFLOP/s speedup=0.27x
```

### tail_decode_m1_k4096_n4095 (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 4096 --gemm-out 4095 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/entry_tail_tail_decode_m1_k4096_n4095.log

```text
gemm nt fp32         avg=3.3282 ms p50=3.3281 ms p95=3.4150 ms throughput=10.08 GFLOP/s speedup=1.00x
gemm nt fp16         avg=9.2320 ms p50=9.2308 ms p95=9.2755 ms throughput=3.63 GFLOP/s speedup=0.36x
gemm nt int8         avg=5.8535 ms p50=5.8474 ms p95=5.9264 ms throughput=5.73 GFLOP/s speedup=0.57x
```

### tail_decode_m1_k4096_n4095 (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 4096 --gemm-out 4095 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/micro_tail_tail_decode_m1_k4096_n4095.log

```text
nt entry fp32        avg=3.6148 ms p50=3.5438 ms p95=3.9782 ms throughput=9.28 GFLOP/s speedup=1.00x
nt entry fp16        avg=9.2900 ms p50=9.2793 ms p95=9.4348 ms throughput=3.61 GFLOP/s speedup=0.39x
nt entry int8        avg=5.9130 ms p50=5.8887 ms p95=6.1731 ms throughput=5.67 GFLOP/s speedup=0.61x
rowpacked fp32       avg=1.7080 ms p50=1.6975 ms p95=1.9249 ms throughput=19.64 GFLOP/s speedup=1.00x
rowpacked fp16       avg=4.6823 ms p50=4.6679 ms p95=4.7843 ms throughput=7.16 GFLOP/s speedup=0.36x
rowpacked int8       avg=5.7475 ms p50=5.7458 ms p95=5.7755 ms throughput=5.84 GFLOP/s speedup=0.30x
```

### tail_decode_m1_k4096_n4096 (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 4096 --gemm-out 4096 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/entry_tail_tail_decode_m1_k4096_n4096.log

```text
gemm nt fp32         avg=3.5595 ms p50=3.4993 ms p95=3.7825 ms throughput=9.43 GFLOP/s speedup=1.00x
gemm nt fp16         avg=9.2814 ms p50=9.2800 ms p95=9.3415 ms throughput=3.62 GFLOP/s speedup=0.38x
gemm nt int8         avg=5.9053 ms p50=5.9067 ms p95=5.9266 ms throughput=5.68 GFLOP/s speedup=0.60x
```

### tail_decode_m1_k4096_n4096 (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 4096 --gemm-out 4096 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/micro_tail_tail_decode_m1_k4096_n4096.log

```text
nt entry fp32        avg=3.3546 ms p50=3.3527 ms p95=3.4240 ms throughput=10.00 GFLOP/s speedup=1.00x
nt entry fp16        avg=9.2019 ms p50=9.2160 ms p95=9.2549 ms throughput=3.65 GFLOP/s speedup=0.36x
nt entry int8        avg=5.8581 ms p50=5.8692 ms p95=5.8905 ms throughput=5.73 GFLOP/s speedup=0.57x
rowpacked fp32       avg=1.4838 ms p50=1.4744 ms p95=1.5428 ms throughput=22.61 GFLOP/s speedup=1.00x
rowpacked fp16       avg=4.6325 ms p50=4.6376 ms p95=4.6526 ms throughput=7.24 GFLOP/s speedup=0.32x
rowpacked int8       avg=5.6882 ms p50=5.6859 ms p95=5.7365 ms throughput=5.90 GFLOP/s speedup=0.26x
```

### tail_decode_m1_k4096_n4097 (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 4096 --gemm-out 4097 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/entry_tail_tail_decode_m1_k4096_n4097.log

```text
gemm nt fp32         avg=3.4246 ms p50=3.4213 ms p95=3.5057 ms throughput=9.80 GFLOP/s speedup=1.00x
gemm nt fp16         avg=9.1537 ms p50=9.1828 ms p95=9.2599 ms throughput=3.67 GFLOP/s speedup=0.37x
gemm nt int8         avg=5.8943 ms p50=5.9003 ms p95=5.9669 ms throughput=5.69 GFLOP/s speedup=0.58x
```

### tail_decode_m1_k4096_n4097 (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 4096 --gemm-out 4097 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/tail/micro_tail_tail_decode_m1_k4096_n4097.log

```text
nt entry fp32        avg=3.3773 ms p50=3.3611 ms p95=3.4908 ms throughput=9.94 GFLOP/s speedup=1.00x
nt entry fp16        avg=9.2538 ms p50=9.2528 ms p95=9.2720 ms throughput=3.63 GFLOP/s speedup=0.36x
nt entry int8        avg=5.8994 ms p50=5.8996 ms p95=5.9187 ms throughput=5.69 GFLOP/s speedup=0.57x
rowpacked fp32       avg=1.4989 ms p50=1.4961 ms p95=1.5585 ms throughput=22.39 GFLOP/s speedup=1.00x
rowpacked fp16       avg=4.6605 ms p50=4.6603 ms p95=4.6834 ms throughput=7.20 GFLOP/s speedup=0.32x
rowpacked int8       avg=5.7466 ms p50=5.7395 ms p95=5.7644 ms throughput=5.84 GFLOP/s speedup=0.26x
```

