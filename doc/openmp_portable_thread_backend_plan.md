# OpenMP 到可移植多线程后端迁移方案

## 1. 背景

`minfer` 目前的 CPU 多线程几乎完全依赖 OpenMP：

- `include/minfer/system.h` / `src/core/system.cpp` 里的线程数接口直接包着 `omp_*`
- `src/backend/cpu/kernel/openmp_utils.h` 里的并行门控直接依赖 `omp_in_parallel()` 和 `omp_get_max_threads()`
- `benchmark/benchmark_threads.h`、`benchmark/op_benchmark.cpp`、`benchmark/llm_benchmark.cpp` 也都在读写 OpenMP 线程状态
- 大量 kernel 和 layer 代码直接写了 `#pragma omp parallel for`
- decode 路径里还有线程本地 scratch，依赖 `omp_get_thread_num()` 做索引

这在 Linux 上没问题，但在 macOS 上不是一个稳妥的产品方案。短期可以装 `libomp` 止血，长期必须把线程能力从 OpenMP 里拆出来。

## 2. 目标

这次迁移的目标不是“把 OpenMP 换个名字”，而是把线程能力变成一个独立后端。

### 目标状态

```
        kernels / layers / benchmark
                     |
              minfer::parallel
               /              \
     portable worker pool   OpenMP adapter
        (mac default)       (optional)
```

要求：

- macOS 默认不依赖 OpenMP，也能开多线程
- 线程数语义保持不变，`set_num_threads()` / `get_num_threads()` 仍然是全局线程预算
- 现有并行策略和阈值可以保留，但实现不再绑定 OpenMP API
- 不改数学逻辑，不改输出 contract，只改并行后端

## 3. 迁移原则

### 3.1 不引入新的重依赖

优先用 `std::thread` + 固定 worker pool 实现 portable backend，避免把 mac 端问题从“装不了 OpenMP”变成“又要装一个新线程库”。

### 3.2 先抽象，再迁移

不要在每个文件里补 `#ifdef __APPLE__`。那会把问题扩散成一地 ifdef，后面没人敢碰。

### 3.3 线程 ID 必须可用

decode GEMM、top-k、argmax 这些路径需要稳定的 worker slot。只知道“当前在并行区里”不够，必须能拿到 worker index。

### 3.4 保留回退通道

OpenMP 后端不一定要马上删除。它可以作为 Linux / CI 的对照实现，但不能再是 mac 的默认依赖。

## 4. 拟新增的并行接口

建议新增一个统一的并行 facade，例如：

- `include/minfer/parallel.h`
- `src/core/parallel.cpp`

建议接口：

- `void set_num_threads(int threads)`
- `int get_num_threads()`
- `bool in_parallel_region()`
- `int worker_index()`
- `int worker_count()`
- `bool should_parallelize_1d_loop(size_t trip_count, size_t work_per_item, long long min_total_work, int min_items_per_thread = 1)`
- `parallel_for(begin, end, grain, fn)` 或者等价的 1D 分发接口

约束：

- 并行 facade 负责“要不要并行”和“并行时谁来执行”
- kernel 只关心自己的 block 切分，不关心后端是不是 OpenMP
- nested parallel 要有统一抑制逻辑，避免线程池里再开线程池

## 5. 当前代码分层

### 5.1 需要先迁走的基础层

- `src/core/system.cpp`
- `include/minfer/system.h`
- `benchmark/benchmark_threads.h`
- `benchmark/op_benchmark.cpp`
- `benchmark/llm_benchmark.cpp`

### 5.2 需要迁移的 kernel / layer

先看简单路径：

- `src/backend/cpu/kernel/activation_kernel_xsimd.cpp`
- `src/backend/cpu/kernel/transpose_kernel.cpp`
- `src/backend/cpu/kernel/binary_kernel_xsimd.cpp`
- `src/backend/cpu/kernel/normalization_kernel_xsimd.cpp`
- `src/backend/cpu/kernel/rope_kernel.cpp`
- `src/backend/cpu/layer/runtime_weight.cpp`
- `src/core/mat_gemm.cpp`

后看重路径：

- `src/backend/cpu/kernel/gemm_kernel_xsimd.cpp`
- `src/backend/cpu/layer/attention_layer.cpp`

## 6. PR 切分

### PR1：可移植线程后端骨架 + 基础路径迁移

PR1 的目标很明确：

- macOS 可以在不装 OpenMP 的情况下构建和运行
- 线程数接口切到统一 facade
- 一批简单 kernel 已经通过新后端跑起来
- benchmark 和系统线程策略不再直接依赖 `omp_*`

PR1 建议范围：

- 新增并行 facade
- 新增 portable worker pool backend
- 保留 OpenMP adapter 作为可选实现
- 迁移 `system.cpp`、benchmark 线程配置、基础 kernel 的并行门控
- 补齐基础测试，确认单线程和多线程都正确

PR1 不做的事：

- 不强求 GEMM / decode 全部迁完
- 不强求性能已经和 OpenMP 完全一致
- 不清理所有旧的 OpenMP 代码

PR1 的文件级与函数级实施清单见：

- `doc/openmp_pr1_implementation_checklist.md`

### PR2：重路径迁移 + 旧 OpenMP 依赖收口

PR2 的目标是把真正吃性能的路径收干净：

- `gemm_kernel_xsimd.cpp` 的 decode / packed scratch 路径改成 portable backend
- `attention_layer.cpp` 的多处并行循环改成统一 facade
- `omp_get_thread_num()` 这类线程 ID 依赖全部换成 worker slot
- 旧的 OpenMP 直连代码只保留在 adapter 层，或者全部移除

PR2 完成后，macOS 不应再需要 OpenMP 才能跑出多线程。

## 7. PR1 验收标准

PR1 通过的标准，优先看“能不能用”，其次才看“快不快”。

### 7.1 构建与运行

- macOS 上可以不安装 OpenMP 直接构建通过
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- `cmake --build build -j`
- `./build/minfer_test` 能跑通基础测试

### 7.2 线程语义

- `set_num_threads()` / `get_num_threads()` 在 mac 和 Linux 上行为一致
- `benchmark` 和 `Net::NetImpl::applyPhaseThreads()` 还能正确设置 prefill / decode 线程预算
- `should_parallelize_1d_loop()` 在没有 OpenMP 的情况下仍然返回正确门控结果

### 7.3 正确性

- `threads=1` 和 `threads>1` 的输出一致，至少覆盖：
  - `activation`
  - `transpose`
  - `binary`
  - `normalization`
  - `rope`
  - `runtime_weight`
  - `mat_gemm`
- 核心单测通过，不允许因为线程后端迁移引入数据竞争或输出漂移

### 7.4 兼容性

- 现有 benchmark 命令还能执行
- `--threads` 参数仍然生效
- 没有把“mac 支持”绑定成“必须安装 libomp”

### 7.5 PR1 结束时的最低性能要求

- 单线程性能不能明显回退
- 多线程路径必须真的启用，不是静默退回单线程
- 对基础 kernel，不要求完全追平 OpenMP，但不能出现肉眼可见的灾难性退化

## 8. PR2 验收标准

PR2 通过的标准，优先看“重路径是否彻底接管”，再看“性能是否合理”。

### 8.1 覆盖范围

- `src/backend/cpu/kernel/gemm_kernel_xsimd.cpp` 的并行 scratch 路径改完
- `src/backend/cpu/layer/attention_layer.cpp` 的并行路径改完
- 代码里不再出现直接依赖 OpenMP 的核心执行路径
- `worker_index()` 能支撑 decode / top-k / argmax 这些线程本地状态

### 8.2 正确性

- `threads=1 / 2 / 4 / 8 / 16` 的结果一致
- repeated decode / repeated prefill 不出现 scratch 污染
- nested parallel 场景下不会死锁，不会无限开线程
- 线程切换不会污染下一次请求的线程预算

### 8.3 性能门槛

在同一台机器上，固定基线后验证：

- 大 decode 场景下，`threads=4` 必须比 `threads=1` 更快
- `threads=8` 或 `threads=16` 不能比 `threads=4` 明显崩掉
- 单线程基线回退不超过 `3%`
- `prefill` 和 decode 的端到端 benchmark 不允许出现系统性退化

### 8.4 收口标准

- 核心代码路径不再直接 `#include <omp.h>`
- `#pragma omp` 要么消失，要么只存在于可选 adapter 文件里
- macOS 默认构建不需要 OpenMP 运行时

## 9. 风险点

### 9.1 线程本地 scratch

风险最大的是 decode GEMM。现在它依赖 `omp_get_thread_num()` 取 scratch slot。换成 portable backend 后，必须确保 worker id 稳定，不然会有数据串扰。

### 9.2 嵌套并行

当前很多 kernel 会在更高层并行里再判断一次并行阈值。新的 backend 必须保留“已经在并行区内就不再嵌套”的语义。

### 9.3 性能抖动

mac 上线程调度和 Linux 很不一样。不要直接拿 OpenMP 的经验阈值照搬。阈值可以先保留，但要在 PR2 之后按 mac 实测重调。

### 9.4 退路

如果 portable backend 的某条路径出问题，回退应该是：

- 先把该路径切回单线程
- 再保留其他路径继续跑
- 不应该回到“整个项目只要 OpenMP 才能多线程”的老状态

## 10. 验证命令

PR1 / PR2 的验收建议用下面几类命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

```bash
./build/minfer_benchmark --threads 4 --prompt-lens 128 --decode-tokens 32 --warmup 1 --runs 3
```

```bash
./build/minfer_op_benchmark \
  --threads 4 \
  --gemm-runtime-only \
  --warmup 10 \
  --iters 30 \
  --head-count 1 \
  --head-count-kv 1 \
  --batch 1 \
  --seq-len 1 \
  --hidden 512 \
  --gemm-out 32768
```

## 11. 结论

这条路最稳的做法不是“让 mac 也装 OpenMP”，而是把线程能力从 OpenMP 里抽出来。

PR1 先把多线程后端做成可移植的。  
PR2 再把 GEMM、decode、attention 这些真正吃性能的路径切过去。  

这样 mac 能用，Linux 还能保留 OpenMP 作为对照，代码也不会继续被 `#ifdef _OPENMP` 撕成两半。
