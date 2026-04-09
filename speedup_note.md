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

# 引入 OpenMP 并行后的最新架构基准测试
我们按照 `openmp_plan.md` 增加了 `#pragma omp parallel for`，特别对 `forwardDecode` 中的 `head_count` 迭代以及 `repeat_kv_if_needed` 内存搬运过程启用了多核。再次运行了 `./minfer_benchmark`，结果如下反思：

'''
=== Benchmark Results ===
[prompt_len=32, decode_tokens=128]
prefill : avg=148.09 ms, p50=150.67 ms, p90=163.06 ms, throughput=216.09 tok/s
decode  : avg=6.21 ms/tok, p50=4.72, p90=9.10, p99=18.63, throughput=161.13 tok/s
ttft    : avg=156.68 ms
end2end : throughput=169.76 tok/s (prompt+decode)

[prompt_len=128, decode_tokens=128]
prefill : avg=357.22 ms, p50=352.20 ms, p90=372.19 ms, throughput=358.32 tok/s
decode  : avg=6.08 ms/tok, p50=4.66, p90=8.60, p99=19.41, throughput=164.41 tok/s
ttft    : avg=374.61 ms
end2end : throughput=225.40 tok/s (prompt+decode)

[prompt_len=512, decode_tokens=128]
prefill : avg=1123.86 ms, p50=1099.56 ms, p90=1206.11 ms, throughput=455.57 tok/s
decode  : avg=7.72 ms/tok, p50=5.16, p90=11.09, p99=42.26, throughput=129.51 tok/s
ttft    : avg=1152.61 ms
end2end : throughput=303.00 tok/s (prompt+decode)
'''

**结论异常总结（Performance Degraded）**：
- OpenMP 未并发前：32 提示词时 Decode = 4.58 ms/tok。
- OpenMP 满负载 (32核) 并发后：32 提示词时 Decode = 6.21 ms/tok (大幅度负向衰减)。

**分析：多线程编排的“假共享 (False Sharing)” 和 “开销掩蔽 (Overhead Domination)”**
因为 B=1 (每帧 1 个 query token) 时，即便序列长度 `S` == 512，每个 Head 计算任务仅约 $512 \times 64 = 32000$ 个浮点乘加（MACs）。这种极其微观且在 L1 Cache 不可思议般热的计算，交由 `omp` 开辟 16 个线程或者使用屏障同步，它的多线程**唤醒/调度消耗**完全掩盖并且数百倍碾压了并行执行省下的时间。

---

# 测试 `Threads=4` 时的最佳甜点 (Sweet Spot)
为了验证是否是线程挂载过度导致的调度开销（Overhead），我们在命令行施加了 `--threads 4` 测试仅仅使用 4 个工作线程来服务 16 个 Head，使得每个线程分摊 4 个完整的 Head 计算：

'''
=== Benchmark Results (Threads=4) ===
[prompt_len=32, decode_tokens=128]
prefill : avg=60.20 ms, p50=60.36 ms, p90=60.42 ms, throughput=531.59 tok/s
decode  : avg=4.51 ms/tok, p50=4.44, p90=4.83, p99=5.47, throughput=221.92 tok/s
ttft    : avg=64.90 ms
end2end : throughput=251.19 tok/s (prompt+decode)

[prompt_len=128, decode_tokens=128]
prefill : avg=166.76 ms, p50=166.88 ms, p90=167.46 ms, throughput=767.59 tok/s
decode  : avg=4.50 ms/tok, p50=4.46, p90=4.75, p99=5.19, throughput=222.21 tok/s
ttft    : avg=171.98 ms
end2end : throughput=344.65 tok/s (prompt+decode)

[prompt_len=512, decode_tokens=128]
prefill : avg=792.77 ms, p50=791.91 ms, p90=795.69 ms, throughput=645.84 tok/s
decode  : avg=5.11 ms/tok, p50=4.96, p90=5.62, p99=6.85, throughput=195.61 tok/s
ttft    : avg=798.45 ms
end2end : throughput=442.25 tok/s (prompt+decode)
'''

**最终结论：**
1. **Prefill 实现大跃进**：长 Prompt (`512`) 的 Prefill 原为 `1153 ms`，4 线程下直接干到了 **`792.77 ms`**！
2. **Decode 重回巅峰并反超**：
   - 满载 32 线程：`6.21 ms/tok`
   - 单线程无 OMP：`4.58 ms/tok`
   - 适量的 4 线程：**`4.50 ms/tok`**（达到目前最优解，尤其 Prompt `512` 时从 `7.31` 降到了 `5.11 ms/tok`）。

由此可见当算力负载极其微缩时，**少量线程 (4 线程)** 平摊掉调度开销后能实现真正的吞吐量净增长！这说明了为 OpenMP 设定正确的 Thread 上限在边缘侧极度关键。

---

# XSIMD Transpose 向量化极限提速

我们在 `transpose_kernel.cpp` 中完全移除了原始嵌套 `for` 循环的缓慢缓存惩罚写法，并将 16-bit 和 32-bit 的矩阵切断重构为了能够适配任意硬件架构（NEON / AVX2 / SSE）的 `xsimd::transpose` 寄存器内极速转置。

**Benchmark 结果 (`shape=2048x2048`, 4 Threads)**：
- **传统的 Tiled C++ 循环转置**： `3.0157 ms`
- **使用 XSIMD Blocked 批量指令转置**： `1.1009 ms` \

**结论**：在最吃内存带宽和 Cache Missing 的 Transpose 算子上，通过底层寄存器 SIMD Shuffle 优化，达成了接近 **`300% (3倍)` 的史诗级提速**！这对于大型 Prompt 的 GQA/KV 重排等预处理过程极为关键。


## Speed up decode


=== Benchmark Results ===
[prompt_len=32, decode_tokens=128]
prefill : avg=59.97 ms, p50=59.56 ms, p90=61.24 ms, throughput=533.63 tok/s
decode  : avg=2.13 ms/tok, p50=2.10, p90=2.35, p99=2.59, throughput=468.99 tok/s
ttft    : avg=62.57 ms
end2end : throughput=480.63 tok/s (prompt+decode)

[prompt_len=128, decode_tokens=128]
prefill : avg=165.37 ms, p50=165.37 ms, p90=166.84 ms, throughput=774.04 tok/s
decode  : avg=2.29 ms/tok, p50=2.24, p90=2.51, p99=3.10, throughput=436.84 tok/s
ttft    : avg=168.50 ms
end2end : throughput=558.49 tok/s (prompt+decode)

[prompt_len=512, decode_tokens=128]
prefill : avg=763.34 ms, p50=765.19 ms, p90=767.37 ms, throughput=670.74 tok/s
decode  : avg=2.93 ms/tok, p50=2.87, p90=3.25, p99=3.62, throughput=341.03 tok/s
ttft    : avg=766.88 ms
end2end : throughput=562.06 tok/s (prompt+decode)


╔══════════════════════════════════════════════════════════════════════════╗
║                     Per-Layer Benchmark Report                          ║
╚══════════════════════════════════════════════════════════════════════════╝

  ── Prefill ──
  Layer               │  Calls │   Total(ms) │    Avg(ms) │      %
  ────────────────────┼────────┼─────────────┼────────────┼────────
  InputLayer_0        │     18 │     0.007   │    0.000   │   0.0%
  EmbeddingLayer_1    │     18 │     1.300   │    0.072   │   0.0%
  AttentionLayer_2    │     18 │   301.546   │   16.753   │   5.1%
  FeedForwardLayer_3  │     18 │   315.110   │   17.506   │   5.3%
  AttentionLayer_4    │     18 │   272.766   │   15.154   │   4.6%
  FeedForwardLayer_5  │     18 │   309.975   │   17.221   │   5.2%
  AttentionLayer_6    │     18 │   268.029   │   14.890   │   4.5%
  FeedForwardLayer_7  │     18 │   309.946   │   17.219   │   5.2%
  AttentionLayer_8    │     18 │   268.451   │   14.914   │   4.5%
  FeedForwardLayer_9  │     18 │   311.153   │   17.286   │   5.2%
  AttentionLayer_10   │     18 │   270.192   │   15.011   │   4.5%
  FeedForwardLayer_11 │     18 │   311.764   │   17.320   │   5.2%
  AttentionLayer_12   │     18 │   272.101   │   15.117   │   4.6%
  FeedForwardLayer_13 │     18 │   310.735   │   17.263   │   5.2%
  AttentionLayer_14   │     18 │   272.078   │   15.115   │   4.6%
  FeedForwardLayer_15 │     18 │   311.283   │   17.293   │   5.2%
  AttentionLayer_16   │     18 │   275.711   │   15.317   │   4.6%
  FeedForwardLayer_17 │     18 │   313.281   │   17.404   │   5.3%
  RMSNormLayer_18     │     18 │     3.813   │    0.212   │   0.1%
  LinearLayer_19      │     18 │  1062.278   │   59.015   │  17.9%
  OutputLayer_20      │     18 │   180.902   │   10.050   │   3.0%
  ────────────────────┼────────┼─────────────┼────────────┼────────
  TOTAL               │        │  5942.420   │            │ 100.0%


  ── Decode ──
  Layer               │  Calls │   Total(ms) │    Avg(ms) │      %
  ────────────────────┼────────┼─────────────┼────────────┼────────
  InputLayer_0        │   2304 │     0.277   │    0.000   │   0.0%
  EmbeddingLayer_1    │   2304 │     2.399   │    0.001   │   0.0%
  AttentionLayer_2    │   2304 │   211.303   │    0.092   │   3.7%
  FeedForwardLayer_3  │   2304 │   315.120   │    0.137   │   5.5%
  AttentionLayer_4    │   2304 │   209.911   │    0.091   │   3.7%
  FeedForwardLayer_5  │   2304 │   306.706   │    0.133   │   5.4%
  AttentionLayer_6    │   2304 │   208.687   │    0.091   │   3.7%
  FeedForwardLayer_7  │   2304 │   306.453   │    0.133   │   5.4%
  AttentionLayer_8    │   2304 │   209.867   │    0.091   │   3.7%
  FeedForwardLayer_9  │   2304 │   308.319   │    0.134   │   5.4%
  AttentionLayer_10   │   2304 │   208.522   │    0.091   │   3.7%
  FeedForwardLayer_11 │   2304 │   308.576   │    0.134   │   5.4%
  AttentionLayer_12   │   2304 │   208.038   │    0.090   │   3.7%
  FeedForwardLayer_13 │   2304 │   308.568   │    0.134   │   5.4%
  AttentionLayer_14   │   2304 │   208.803   │    0.091   │   3.7%
  FeedForwardLayer_15 │   2304 │   304.331   │    0.132   │   5.3%
  AttentionLayer_16   │   2304 │   206.206   │    0.089   │   3.6%
  FeedForwardLayer_17 │   2304 │   305.117   │    0.132   │   5.4%
  RMSNormLayer_18     │   2304 │     4.143   │    0.002   │   0.1%
  LinearLayer_19      │   2304 │  1537.110   │    0.667   │  27.0%
  OutputLayer_20      │   2304 │    12.173   │    0.005   │   0.2%
  ────────────────────┼────────┼─────────────┼────────────┼────────
  TOTAL               │        │  5690.628   │            │ 100.0%

  