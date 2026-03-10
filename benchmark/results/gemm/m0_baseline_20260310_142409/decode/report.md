# GEMM Benchmark Report

## Context

- Timestamp: 2026-03-10T14:24:15+08:00
- Commit: df860ae223adf12fbe4f0e3631ff3641a1030f89
- CPU: AMD Ryzen 9 7950X 16-Core Processor
- Build: Release
- Binary: /home/moo/work/my_lab/minfer/build/minfer_op_benchmark
- Group: decode
- Mode: full
- Threads: 32
- Warmup: 10
- Iters: 30
- Filter: 

## Results

### decode_m1_h512_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 512 --gemm-out 2048 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/entry_decode_decode_m1_h512_ffn.log

```text
gemm nt fp32         avg=0.1441 ms p50=0.1276 ms p95=0.1667 ms throughput=14.56 GFLOP/s speedup=1.00x
gemm nt fp16         avg=0.5875 ms p50=0.5855 ms p95=0.6172 ms throughput=3.57 GFLOP/s speedup=0.25x
gemm nt int8         avg=0.4105 ms p50=0.4091 ms p95=0.4148 ms throughput=5.11 GFLOP/s speedup=0.35x
```

### decode_m1_h512_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 512 --gemm-out 2048 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/micro_decode_decode_m1_h512_ffn.log

```text
nt entry fp32        avg=0.1275 ms p50=0.1265 ms p95=0.1304 ms throughput=16.45 GFLOP/s speedup=1.00x
nt entry fp16        avg=0.5850 ms p50=0.5840 ms p95=0.6112 ms throughput=3.58 GFLOP/s speedup=0.22x
nt entry int8        avg=0.4102 ms p50=0.4080 ms p95=0.4137 ms throughput=5.11 GFLOP/s speedup=0.31x
rowpacked fp32       avg=0.0840 ms p50=0.0797 ms p95=0.1102 ms throughput=24.98 GFLOP/s speedup=1.00x
rowpacked fp16       avg=0.3177 ms p50=0.3155 ms p95=0.3228 ms throughput=6.60 GFLOP/s speedup=0.26x
rowpacked int8       avg=0.3965 ms p50=0.3934 ms p95=0.4023 ms throughput=5.29 GFLOP/s speedup=0.21x
```

### decode_m1_h512_ffn (runtime)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-runtime-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 512 --gemm-out 2048 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/runtime_decode_decode_m1_h512_ffn.log

```text
runtime fp32         avg=0.0843 ms p50=0.0804 ms p95=0.0984 ms throughput=24.86 GFLOP/s speedup=1.00x
runtime fp16         avg=0.0811 ms p50=0.0804 ms p95=0.0858 ms throughput=25.87 GFLOP/s speedup=1.04x
runtime int8         avg=0.3991 ms p50=0.3938 ms p95=0.4199 ms throughput=5.25 GFLOP/s speedup=0.21x
```

### decode_m1_h1024_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 1024 --gemm-out 4096 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/entry_decode_decode_m1_h1024_ffn.log

```text
gemm nt fp32         avg=0.5784 ms p50=0.5774 ms p95=0.5956 ms throughput=14.50 GFLOP/s speedup=1.00x
gemm nt fp16         avg=2.4214 ms p50=2.4447 ms p95=2.4511 ms throughput=3.46 GFLOP/s speedup=0.24x
gemm nt int8         avg=1.4616 ms p50=1.4702 ms p95=1.4751 ms throughput=5.74 GFLOP/s speedup=0.40x
```

### decode_m1_h1024_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 1024 --gemm-out 4096 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/micro_decode_decode_m1_h1024_ffn.log

```text
nt entry fp32        avg=0.5187 ms p50=0.5194 ms p95=0.5319 ms throughput=16.17 GFLOP/s speedup=1.00x
nt entry fp16        avg=2.1678 ms p50=2.1632 ms p95=2.2082 ms throughput=3.87 GFLOP/s speedup=0.24x
nt entry int8        avg=1.4502 ms p50=1.4450 ms p95=1.4775 ms throughput=5.78 GFLOP/s speedup=0.36x
rowpacked fp32       avg=0.2901 ms p50=0.2886 ms p95=0.3015 ms throughput=28.92 GFLOP/s speedup=1.00x
rowpacked fp16       avg=1.1406 ms p50=1.1390 ms p95=1.1524 ms throughput=7.35 GFLOP/s speedup=0.25x
rowpacked int8       avg=1.4052 ms p50=1.4165 ms p95=1.4234 ms throughput=5.97 GFLOP/s speedup=0.21x
```

### decode_m1_h1024_ffn (runtime)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-runtime-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 1024 --gemm-out 4096 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/runtime_decode_decode_m1_h1024_ffn.log

```text
runtime fp32         avg=0.2966 ms p50=0.2945 ms p95=0.3159 ms throughput=28.28 GFLOP/s speedup=1.00x
runtime fp16         avg=0.2955 ms p50=0.2924 ms p95=0.3131 ms throughput=28.39 GFLOP/s speedup=1.00x
runtime int8         avg=1.4112 ms p50=1.4017 ms p95=1.4414 ms throughput=5.94 GFLOP/s speedup=0.21x
```

### decode_m1_h2048_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/entry_decode_decode_m1_h2048_ffn.log

```text
gemm nt fp32         avg=2.3146 ms p50=2.3163 ms p95=2.3687 ms throughput=14.50 GFLOP/s speedup=1.00x
gemm nt fp16         avg=9.0056 ms p50=9.0061 ms p95=9.2401 ms throughput=3.73 GFLOP/s speedup=0.26x
gemm nt int8         avg=5.8441 ms p50=5.8591 ms p95=5.8720 ms throughput=5.74 GFLOP/s speedup=0.40x
```

### decode_m1_h2048_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/micro_decode_decode_m1_h2048_ffn.log

```text
nt entry fp32        avg=2.3180 ms p50=2.3276 ms p95=2.3630 ms throughput=14.48 GFLOP/s speedup=1.00x
nt entry fp16        avg=8.9237 ms p50=8.9229 ms p95=9.0328 ms throughput=3.76 GFLOP/s speedup=0.26x
nt entry int8        avg=5.8342 ms p50=5.8564 ms p95=5.9049 ms throughput=5.75 GFLOP/s speedup=0.40x
rowpacked fp32       avg=1.4482 ms p50=1.4401 ms p95=1.4934 ms throughput=23.17 GFLOP/s speedup=1.00x
rowpacked fp16       avg=4.6410 ms p50=4.6385 ms p95=4.6648 ms throughput=7.23 GFLOP/s speedup=0.31x
rowpacked int8       avg=5.7204 ms p50=5.7172 ms p95=5.7501 ms throughput=5.87 GFLOP/s speedup=0.25x
```

### decode_m1_h2048_ffn (runtime)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-runtime-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/runtime_decode_decode_m1_h2048_ffn.log

```text
runtime fp32         avg=1.4685 ms p50=1.4581 ms p95=1.5448 ms throughput=22.85 GFLOP/s speedup=1.00x
runtime fp16         avg=1.4537 ms p50=1.4375 ms p95=1.5075 ms throughput=23.08 GFLOP/s speedup=1.01x
runtime int8         avg=5.6321 ms p50=5.6331 ms p95=5.7286 ms throughput=5.96 GFLOP/s speedup=0.26x
```

### decode_m2_h2048_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 2 --seq-len 1 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/entry_decode_decode_m2_h2048_ffn.log

```text
gemm nt fp32         avg=4.6522 ms p50=4.6643 ms p95=4.7154 ms throughput=14.43 GFLOP/s speedup=1.00x
gemm nt fp16         avg=17.8619 ms p50=17.9270 ms p95=18.0704 ms throughput=3.76 GFLOP/s speedup=0.26x
gemm nt int8         avg=11.5798 ms p50=11.5686 ms p95=11.7181 ms throughput=5.80 GFLOP/s speedup=0.40x
```

### decode_m2_h2048_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 2 --seq-len 1 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/micro_decode_decode_m2_h2048_ffn.log

```text
nt entry fp32        avg=0.8067 ms p50=0.6497 ms p95=1.3124 ms throughput=83.19 GFLOP/s speedup=1.00x
nt entry fp16        avg=1.5966 ms p50=1.4625 ms p95=2.4174 ms throughput=42.03 GFLOP/s speedup=0.51x
nt entry int8        avg=0.7693 ms p50=0.7102 ms p95=1.1319 ms throughput=87.23 GFLOP/s speedup=1.05x
rowpacked fp32       avg=2.9324 ms p50=2.9071 ms p95=3.0429 ms throughput=22.88 GFLOP/s speedup=1.00x
rowpacked fp16       avg=9.2688 ms p50=9.2745 ms p95=9.3142 ms throughput=7.24 GFLOP/s speedup=0.32x
rowpacked int8       avg=11.4120 ms p50=11.4146 ms p95=11.4820 ms throughput=5.88 GFLOP/s speedup=0.26x
```

### decode_m2_h2048_ffn (runtime)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-runtime-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 2 --seq-len 1 --hidden 2048 --gemm-out 8192 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/runtime_decode_decode_m2_h2048_ffn.log

```text
runtime fp32         avg=2.9174 ms p50=2.8999 ms p95=2.9948 ms throughput=23.00 GFLOP/s speedup=1.00x
runtime fp16         avg=2.8945 ms p50=2.8888 ms p95=2.9811 ms throughput=23.18 GFLOP/s speedup=1.01x
runtime int8         avg=11.3453 ms p50=11.3574 ms p95=11.5762 ms throughput=5.92 GFLOP/s speedup=0.26x
```

### decode_m4_h4096_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 4 --seq-len 1 --hidden 4096 --gemm-out 16384 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/entry_decode_decode_m4_h4096_ffn.log

```text
gemm nt fp32         avg=53.9468 ms p50=53.6562 ms p95=58.0427 ms throughput=9.95 GFLOP/s speedup=1.00x
gemm nt fp16         avg=147.4666 ms p50=147.5324 ms p95=148.6643 ms throughput=3.64 GFLOP/s speedup=0.37x
gemm nt int8         avg=94.1843 ms p50=93.9384 ms p95=95.2760 ms throughput=5.70 GFLOP/s speedup=0.57x
```

### decode_m4_h4096_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 4 --seq-len 1 --hidden 4096 --gemm-out 16384 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/micro_decode_decode_m4_h4096_ffn.log

```text
nt entry fp32        avg=7.9733 ms p50=7.6009 ms p95=11.3797 ms throughput=67.33 GFLOP/s speedup=1.00x
nt entry fp16        avg=6.0855 ms p50=5.5657 ms p95=8.1336 ms throughput=88.22 GFLOP/s speedup=1.31x
nt entry int8        avg=3.1373 ms p50=2.9819 ms p95=4.9240 ms throughput=171.12 GFLOP/s speedup=2.54x
rowpacked fp32       avg=24.6608 ms p50=24.6100 ms p95=25.4406 ms throughput=21.77 GFLOP/s speedup=1.00x
rowpacked fp16       avg=74.2846 ms p50=74.2877 ms p95=74.5639 ms throughput=7.23 GFLOP/s speedup=0.33x
rowpacked int8       avg=91.2979 ms p50=91.2721 ms p95=92.3034 ms throughput=5.88 GFLOP/s speedup=0.27x
```

### decode_m4_h4096_ffn (runtime)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-runtime-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 4 --seq-len 1 --hidden 4096 --gemm-out 16384 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/runtime_decode_decode_m4_h4096_ffn.log

```text
runtime fp32         avg=25.1240 ms p50=25.0642 ms p95=25.6583 ms throughput=21.37 GFLOP/s speedup=1.00x
runtime fp16         avg=25.1374 ms p50=25.0147 ms p95=25.7273 ms throughput=21.36 GFLOP/s speedup=1.00x
runtime int8         avg=91.0030 ms p50=90.9012 ms p95=91.7266 ms throughput=5.90 GFLOP/s speedup=0.28x
```

### decode_m8_h4096_ffn (entry)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 8 --seq-len 1 --hidden 4096 --gemm-out 16384 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/entry_decode_decode_m8_h4096_ffn.log

```text
gemm nt fp32         avg=108.7651 ms p50=108.6125 ms p95=109.8128 ms throughput=9.87 GFLOP/s speedup=1.00x
gemm nt fp16         avg=293.1402 ms p50=293.0992 ms p95=294.8412 ms throughput=3.66 GFLOP/s speedup=0.37x
gemm nt int8         avg=187.4688 ms p50=187.6529 ms p95=188.5012 ms throughput=5.73 GFLOP/s speedup=0.58x
```

### decode_m8_h4096_ffn (micro)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-micro-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 8 --seq-len 1 --hidden 4096 --gemm-out 16384 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/micro_decode_decode_m8_h4096_ffn.log

```text
nt entry fp32        avg=7.7039 ms p50=7.7187 ms p95=8.7631 ms throughput=139.38 GFLOP/s speedup=1.00x
nt entry fp16        avg=7.6942 ms p50=6.9864 ms p95=11.8561 ms throughput=139.55 GFLOP/s speedup=1.00x
nt entry int8        avg=4.0623 ms p50=3.7905 ms p95=5.7269 ms throughput=264.32 GFLOP/s speedup=1.90x
rowpacked fp32       avg=50.1278 ms p50=49.8684 ms p95=51.3929 ms throughput=21.42 GFLOP/s speedup=1.00x
rowpacked fp16       avg=148.3356 ms p50=148.3139 ms p95=148.6626 ms throughput=7.24 GFLOP/s speedup=0.34x
rowpacked int8       avg=182.0494 ms p50=182.0679 ms p95=183.0247 ms throughput=5.90 GFLOP/s speedup=0.28x
```

### decode_m8_h4096_ffn (runtime)

- Command:

```bash
/home/moo/work/my_lab/minfer/build/minfer_op_benchmark --gemm-runtime-only --warmup 10 --iters 30 --threads 32 --head-count 1 --head-count-kv 1 --batch 8 --seq-len 1 --hidden 4096 --gemm-out 16384 
```

- Log: benchmark/results/gemm/m0_baseline_20260310_142409/decode/runtime_decode_decode_m8_h4096_ffn.log

```text
runtime fp32         avg=50.1853 ms p50=49.8334 ms p95=52.4046 ms throughput=21.40 GFLOP/s speedup=1.00x
runtime fp16         avg=53.4435 ms p50=52.2773 ms p95=61.8074 ms throughput=20.09 GFLOP/s speedup=0.94x
runtime int8         avg=184.5338 ms p50=184.5631 ms p95=185.4419 ms throughput=5.82 GFLOP/s speedup=0.27x
```

