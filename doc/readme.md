# Doc and Experiments

这个文件夹包含一些实现过程中所做的实验和算子分析。对于最重要的算子，可以一个算子一个文件夹，多组实验进行分析。

- `gemm_benchmark_plan.md`：GEMM 专项 benchmark matrix、命令模板、验收口径与里程碑清单。
- `../benchmark/gemm_bench_sweep.sh`：按分组批量执行 GEMM benchmark matrix 的 sweep 脚本。
- `../benchmark/perf_gate.sh`：关键 decode 指标性能门禁脚本（P4），用于本地/CI 回归检查。
- `perf_gate_baseline.md`：当前性能门禁基线快照（机器、阈值、实测值、复现命令）。
