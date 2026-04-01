# 记录当前项目加速紧张

版本commit：bb0a95a66c388571678063c31ebbc48609120df0
'''bash
─$ ./minfer_benchmark 
Model: /home/moo/work/my_lab/minfer/test/big_models/Lite-Oute-1-65M-FP16.gguf
Prompt lengths: 32,128,512
Decode tokens per run: 128
Warmup: 1
Runs: 5
DEBUG: Starting to get_key inside LLama_loader!
DEBUG: Checking ROPE_FREQ_BASE
DEBUG: Checking LAYERNORM_RMS_EPS
DEBUG: Finished parsing in loader! eps = 1e-06
Arch = llama
n_vocab = 32768
n_ctx_length = 2048
n_embd = 512
n_ff = 2048
n_head = 16
n_head_kv = 8
n_layer = 8
n_rope_dim_count = 32
rope_freq_base_train = 10000
f_norm_rms_eps = 1e-06
Model load time: 192.63 ms
Running case prompt_len=32 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 823.37 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=3175.50 ms
  [run 1/5] start
  [run 1/5] prefill done: 782.76 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=3180.54 ms
  [run 2/5] start
  [run 2/5] prefill done: 783.44 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=3172.44 ms
  [run 3/5] start
  [run 3/5] prefill done: 780.65 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=3174.09 ms
  [run 4/5] start
  [run 4/5] prefill done: 780.56 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=3170.45 ms
  [run 5/5] start
  [run 5/5] prefill done: 782.56 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=3177.95 ms
Running case prompt_len=128 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 3182.62 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=3231.42 ms
  [run 1/5] start
  [run 1/5] prefill done: 3177.06 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=3242.20 ms
  [run 2/5] start
  [run 2/5] prefill done: 3183.38 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=3250.50 ms
  [run 3/5] start
  [run 3/5] prefill done: 3175.74 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=3246.21 ms
  [run 4/5] start
  [run 4/5] prefill done: 3184.31 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=3248.03 ms
  [run 5/5] start
  [run 5/5] prefill done: 3178.71 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=3244.38 ms
Running case prompt_len=512 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 13604.58 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=3456.66 ms
  [run 1/5] start
  [run 1/5] prefill done: 13652.66 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=3500.27 ms
  [run 2/5] start
  [run 2/5] prefill done: 13630.61 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=3495.95 ms
  [run 3/5] start
  [run 3/5] prefill done: 13620.67 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=3444.88 ms
  [run 4/5] start
  [run 4/5] prefill done: 13603.84 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=3449.05 ms
  [run 5/5] start


  [run 5/5] prefill done: 13588.62 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=3457.86 ms

=== Benchmark Results ===
[prompt_len=32, decode_tokens=128]
prefill : avg=781.99 ms, p50=782.56 ms, p90=783.16 ms, throughput=40.92 tok/s
decode  : avg=24.81 ms/tok, p50=24.78, p90=25.07, p99=26.00, throughput=40.31 tok/s
ttft    : avg=806.57 ms
end2end : throughput=40.43 tok/s (prompt+decode)

[prompt_len=128, decode_tokens=128]
prefill : avg=3179.84 ms, p50=3178.71 ms, p90=3183.93 ms, throughput=40.25 tok/s
decode  : avg=25.36 ms/tok, p50=25.36, p90=25.65, p99=26.34, throughput=39.43 tok/s
ttft    : avg=3204.97 ms
end2end : throughput=39.84 tok/s (prompt+decode)

[prompt_len=512, decode_tokens=128]
prefill : avg=13619.28 ms, p50=13620.67 ms, p90=13643.84 ms, throughput=37.59 tok/s
decode  : avg=27.11 ms/tok, p50=27.12, p90=27.59, p99=28.17, throughput=36.89 tok/s
ttft    : avg=13646.48 ms
end2end : throughput=37.45 tok/s (prompt+decode)
'''


将GEMM接入 google highway 之后的效果:
commit: 62eda5b8049635f1183361cbecfdc94ab4bcc9cd
'''
Model: /home/moo/work/my_lab/minfer/test/big_models/Lite-Oute-1-65M-FP16.gguf
Prompt lengths: 32,128,512
Decode tokens per run: 128
Warmup: 1
Runs: 5
Arch = llama
n_vocab = 32768
n_ctx_length = 2048
n_embd = 512
n_ff = 2048
n_head = 16
n_head_kv = 8
n_layer = 8
n_rope_dim_count = 32
rope_freq_base_train = 10000
f_norm_rms_eps = 1e-06
Model load time: 179.34 ms
Running case prompt_len=32 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 208.69 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=800.67 ms
  [run 1/5] start
  [run 1/5] prefill done: 186.78 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=791.18 ms
  [run 2/5] start
  [run 2/5] prefill done: 185.51 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=789.14 ms
  [run 3/5] start
  [run 3/5] prefill done: 185.25 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=790.16 ms
  [run 4/5] start
  [run 4/5] prefill done: 185.58 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=793.92 ms
  [run 5/5] start
  [run 5/5] prefill done: 185.57 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=794.08 ms
Running case prompt_len=128 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 765.14 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=828.28 ms
  [run 1/5] start
  [run 1/5] prefill done: 762.84 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=818.47 ms
  [run 2/5] start
  [run 2/5] prefill done: 760.37 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=834.84 ms
  [run 3/5] start
  [run 3/5] prefill done: 759.67 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=818.90 ms
  [run 4/5] start
  [run 4/5] prefill done: 755.92 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=822.47 ms
  [run 5/5] start
  [run 5/5] prefill done: 756.16 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=820.44 ms
Running case prompt_len=512 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 3484.41 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=952.74 ms
  [run 1/5] start
  [run 1/5] prefill done: 3455.77 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=952.83 ms
  [run 2/5] start
  [run 2/5] prefill done: 3459.47 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=945.37 ms
  [run 3/5] start
  [run 3/5] prefill done: 3481.69 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=948.12 ms
  [run 4/5] start
  [run 4/5] prefill done: 3451.29 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=930.49 ms
  [run 5/5] start
  [run 5/5] prefill done: 3442.08 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=924.69 ms

=== Benchmark Results ===
[prompt_len=32, decode_tokens=128]
prefill : avg=185.74 ms, p50=185.57 ms, p90=186.30 ms, throughput=172.29 tok/s
decode  : avg=6.19 ms/tok, p50=6.19, p90=6.29, p99=6.62, throughput=161.68 tok/s
ttft    : avg=191.91 ms
end2end : throughput=163.69 tok/s (prompt+decode)

[prompt_len=128, decode_tokens=128]
prefill : avg=758.99 ms, p50=759.67 ms, p90=761.85 ms, throughput=168.64 tok/s
decode  : avg=6.43 ms/tok, p50=6.40, p90=6.60, p99=6.96, throughput=155.52 tok/s
ttft    : avg=765.58 ms
end2end : throughput=161.82 tok/s (prompt+decode)

[prompt_len=512, decode_tokens=128]
prefill : avg=3458.06 ms, p50=3455.77 ms, p90=3472.80 ms, throughput=148.06 tok/s
decode  : avg=7.35 ms/tok, p50=7.30, p90=7.66, p99=8.14, throughput=136.13 tok/s
ttft    : avg=3465.69 ms
end2end : throughput=145.51 tok/s (prompt+decode)
'''


主要修改：
从google highway换成xsimd，切换原因，highway是对原有的intrinsic更深度的封装，而xsimd是更接近原始intrinsic的实现。我们场景更需要自己基于intrinsic来实现gemm等算子的微内核。目前还没有加入微内核的划分，只使用了xsimd和openmp并行。
tag:f49dfabd6c5acd8cc6350fdb834c02cf59289194
'''
./minfer_benchmark 
Model: /home/moo/work/my_lab/minfer/test/big_models/Lite-Oute-1-65M-FP16.gguf
Prompt lengths: 32,128,512
Decode tokens per run: 128
Warmup: 1
Runs: 5
Threads: 32
Arch = llama
n_vocab = 32768
n_ctx_length = 2048
n_embd = 512
n_ff = 2048
n_head = 16
n_head_kv = 8
n_layer = 8
n_rope_dim_count = 32
rope_freq_base_train = 10000
f_norm_rms_eps = 1e-06
Model load time: 543.65 ms
Running case prompt_len=32 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 133.86 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=801.73 ms
  [run 1/5] start
  [run 1/5] prefill done: 126.74 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=804.22 ms
  [run 2/5] start
  [run 2/5] prefill done: 128.74 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=807.12 ms
  [run 3/5] start
  [run 3/5] prefill done: 129.56 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=807.69 ms
  [run 4/5] start
  [run 4/5] prefill done: 128.56 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=814.38 ms
  [run 5/5] start
  [run 5/5] prefill done: 131.55 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=807.94 ms
Running case prompt_len=128 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 676.21 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=832.86 ms
  [run 1/5] start
  [run 1/5] prefill done: 678.43 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=833.70 ms
  [run 2/5] start
  [run 2/5] prefill done: 659.44 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=829.88 ms
  [run 3/5] start
  [run 3/5] prefill done: 677.94 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=830.99 ms
  [run 4/5] start
  [run 4/5] prefill done: 636.89 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=831.27 ms
  [run 5/5] start
  [run 5/5] prefill done: 662.81 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=829.27 ms
Running case prompt_len=512 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 1087.31 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=933.31 ms
  [run 1/5] start
  [run 1/5] prefill done: 1021.08 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=940.90 ms
  [run 2/5] start
  [run 2/5] prefill done: 981.53 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=932.48 ms
  [run 3/5] start
  [run 3/5] prefill done: 1006.06 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=936.65 ms
  [run 4/5] start
  [run 4/5] prefill done: 984.77 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=929.97 ms
  [run 5/5] start
  [run 5/5] prefill done: 994.03 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=937.45 ms

=== Benchmark Results ===
[prompt_len=32, decode_tokens=128]
prefill : avg=129.03 ms, p50=128.74 ms, p90=130.75 ms, throughput=248.00 tok/s
decode  : avg=6.31 ms/tok, p50=6.29, p90=6.44, p99=6.87, throughput=158.36 tok/s
ttft    : avg=135.43 ms
end2end : throughput=170.70 tok/s (prompt+decode)

[prompt_len=128, decode_tokens=128]
prefill : avg=663.10 ms, p50=662.81 ms, p90=678.23 ms, throughput=193.03 tok/s
decode  : avg=6.49 ms/tok, p50=6.48, p90=6.62, p99=6.90, throughput=154.03 tok/s
ttft    : avg=670.04 ms
end2end : throughput=171.34 tok/s (prompt+decode)

[prompt_len=512, decode_tokens=128]
prefill : avg=997.50 ms, p50=994.03 ms, p90=1015.07 ms, throughput=513.29 tok/s
decode  : avg=7.31 ms/tok, p50=7.28, p90=7.47, p99=7.91, throughput=136.83 tok/s
ttft    : avg=1005.15 ms
end2end : throughput=331.09 tok/s (prompt+decode)
'''


兼容MobileKV后的Attention实现：
'''
Model: /home/moo/work/my_lab/minfer/test/big_models/Lite-Oute-1-65M-FP16.gguf
Precision: fp16
Prompt lengths: 32,128,512
Decode tokens per run: 128
Warmup: 1
Runs: 5
Threads: 32
Arch = llama
n_vocab = 32768
n_ctx_length = 2048
n_embd = 512
n_ff = 2048
n_head = 16
n_head_kv = 8
n_layer = 8
n_rope_dim_count = 32
rope_freq_base_train = 10000
f_norm_rms_eps = 1e-06
Model load time: 542.41 ms
Running case prompt_len=32 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 106.96 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=597.23 ms
  [run 1/5] start
  [run 1/5] prefill done: 92.56 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=588.14 ms
  [run 2/5] start
  [run 2/5] prefill done: 111.09 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=583.76 ms
  [run 3/5] start
  [run 3/5] prefill done: 101.94 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=581.72 ms
  [run 4/5] start
  [run 4/5] prefill done: 104.76 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=583.32 ms
  [run 5/5] start
  [run 5/5] prefill done: 95.51 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=595.11 ms
Running case prompt_len=128 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 335.06 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=658.95 ms
  [run 1/5] start
  [run 1/5] prefill done: 374.05 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=663.10 ms
  [run 2/5] start
  [run 2/5] prefill done: 339.79 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=658.08 ms
  [run 3/5] start
  [run 3/5] prefill done: 321.97 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=653.02 ms
  [run 4/5] start
  [run 4/5] prefill done: 324.92 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=647.38 ms
  [run 5/5] start
  [run 5/5] prefill done: 397.10 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=671.68 ms
Running case prompt_len=512 ...
  [warmup 1/1] start
  [warmup 1/1] prefill done: 1187.06 ms
  [warmup 1/1] decode 32/128
  [warmup 1/1] decode 64/128
  [warmup 1/1] decode 96/128
  [warmup 1/1] decode 128/128
  [warmup 1/1] done: decode_total=918.07 ms
  [run 1/5] start
  [run 1/5] prefill done: 1178.29 ms
  [run 1/5] decode 32/128
  [run 1/5] decode 64/128
  [run 1/5] decode 96/128
  [run 1/5] decode 128/128
  [run 1/5] done: decode_total=924.65 ms
  [run 2/5] start
  [run 2/5] prefill done: 1165.72 ms
  [run 2/5] decode 32/128
  [run 2/5] decode 64/128
  [run 2/5] decode 96/128
  [run 2/5] decode 128/128
  [run 2/5] done: decode_total=926.90 ms
  [run 3/5] start
  [run 3/5] prefill done: 1143.01 ms
  [run 3/5] decode 32/128
  [run 3/5] decode 64/128
  [run 3/5] decode 96/128
  [run 3/5] decode 128/128
  [run 3/5] done: decode_total=936.38 ms
  [run 4/5] start
  [run 4/5] prefill done: 1116.02 ms
  [run 4/5] decode 32/128
  [run 4/5] decode 64/128
  [run 4/5] decode 96/128
  [run 4/5] decode 128/128
  [run 4/5] done: decode_total=963.85 ms
  [run 5/5] start
  [run 5/5] prefill done: 1166.86 ms
  [run 5/5] decode 32/128
  [run 5/5] decode 64/128
  [run 5/5] decode 96/128
  [run 5/5] decode 128/128
  [run 5/5] done: decode_total=923.87 ms

=== Benchmark Results ===
[prompt_len=32, decode_tokens=128]
prefill : avg=101.17 ms, p50=101.94 ms, p90=108.56 ms, throughput=316.29 tok/s
decode  : avg=4.58 ms/tok, p50=4.56, p90=4.89, p99=5.25, throughput=218.28 tok/s
ttft    : avg=106.08 ms
end2end : throughput=232.70 tok/s (prompt+decode)

[prompt_len=128, decode_tokens=128]
prefill : avg=351.56 ms, p50=339.79 ms, p90=387.88 ms, throughput=364.09 tok/s
decode  : avg=5.15 ms/tok, p50=5.10, p90=5.55, p99=6.59, throughput=194.34 tok/s
ttft    : avg=357.24 ms
end2end : throughput=253.41 tok/s (prompt+decode)

[prompt_len=512, decode_tokens=128]
prefill : avg=1153.98 ms, p50=1165.72 ms, p90=1173.72 ms, throughput=443.68 tok/s
decode  : avg=7.31 ms/tok, p50=7.28, p90=7.60, p99=9.33, throughput=136.88 tok/s
ttft    : avg=1161.58 ms
end2end : throughput=306.35 tok/s (prompt+decode)
'''