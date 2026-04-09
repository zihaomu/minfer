# Perf Gate Baseline

## 基线快照

- Date: `2026-04-08T23:22:58+08:00`
- Commit: `93df9ffe0cafe32d85c1fe1a2df8f8327ef9bf98`
- Machine: `AMD Ryzen 9 7950X 16-Core Processor`
- Script: `./benchmark/perf_gate.sh`
- Raw logs: `benchmark/results/perf_gate/baseline_20260408_2323/`

## 门禁阈值（默认）

- `decode avg(ms/tok) @ threads=4 <= 2.20`
- `decode avg(ms/tok) @ threads=16 <= 1.30`
- `LinearLayer_19 decode Avg(ms) @ threads=4 <= 0.60`

## 本次实测结果

- `decode avg(ms/tok) @ threads=4 = 2.14` (`PASS`)
- `decode avg(ms/tok) @ threads=16 = 1.30` (`PASS`)
- `LinearLayer_19 decode Avg(ms) @ threads=4 = 0.596` (`PASS`)

补充指标（同一批日志）：
- `threads=4`: `prefill=163.58 ms`, `decode throughput=467.23 tok/s`, `end2end=585.11 tok/s`
- `threads=16`: `prefill=142.78 ms`, `decode throughput=770.89 tok/s`, `end2end=828.96 tok/s`

## 复现实验命令

```bash
./benchmark/perf_gate.sh --log-dir benchmark/results/perf_gate/baseline_20260408_2323
```

如需临时调整阈值：

```bash
MINFER_GATE_DECODE_T4_MAX_MS=2.25 \
MINFER_GATE_DECODE_T16_MAX_MS=1.35 \
MINFER_GATE_LINEAR_DECODE_MAX_MS=0.62 \
./benchmark/perf_gate.sh
```

## 更新策略

- 仅在以下场景更新本基线：
  - 硬件/编译器发生变化。
  - 内核策略发生结构性变化且连续多轮测得新稳态。
- 建议至少保留最近 3 轮基线记录（目录+文档）用于回退定位。
