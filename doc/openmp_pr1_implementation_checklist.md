# PR1 实施清单：可移植多线程后端骨架

## 1. PR1 范围

PR1 只做两件事：

- 建立统一并行 facade，让 macOS 在不依赖 OpenMP 的前提下也能启用多线程。
- 迁移基础路径（system/benchmark/基础 kernel），不包含 `gemm_kernel_xsimd.cpp` 和 `attention_layer.cpp` 的重路径重写。

不在 PR1 的范围：

- decode scratch 的 worker-id 路由改造
- `gemm_kernel_xsimd.cpp` 全量去 OpenMP
- 追求与 OpenMP 完全同等性能

## 2. 交付结构（文件级）

### 2.1 新增文件

- `include/minfer/parallel.h`
- `src/core/parallel.cpp`
- `src/core/parallel_portable.cpp`
- `src/core/parallel_openmp.cpp`

备注：
- 具体拆成 2 个 `.cpp` 还是 1 个 `.cpp` 可按实现调整。
- 保持接口在 `include/minfer/parallel.h`，实现细节不暴露给 kernel。

### 2.2 修改文件

- `CMakeLists.txt`
- `include/minfer/system.h`
- `src/core/system.cpp`
- `src/backend/cpu/kernel/openmp_utils.h`（PR1 先保留文件名，内部改为 facade 调用）
- `benchmark/benchmark_threads.h`
- `benchmark/op_benchmark.cpp`
- `benchmark/llm_benchmark.cpp`
- `test/benchmark_threads_test.cpp`

基础 kernel 迁移：

- `src/backend/cpu/kernel/activation_kernel_xsimd.cpp`
- `src/backend/cpu/kernel/transpose_kernel.cpp`
- `src/backend/cpu/kernel/binary_kernel_xsimd.cpp`
- `src/backend/cpu/kernel/normalization_kernel_xsimd.cpp`
- `src/backend/cpu/kernel/rope_kernel.cpp`
- `src/backend/cpu/layer/runtime_weight.cpp`
- `src/core/mat_gemm.cpp`

## 3. 统一接口（函数级约束）

`include/minfer/parallel.h` 提供以下最小接口：

- `void parallel_set_num_threads(int threads);`
- `int parallel_get_num_threads();`
- `int parallel_get_num_procs();`
- `bool parallel_in_parallel();`
- `int parallel_get_worker_index();`
- `int parallel_get_worker_count();`
- `void parallel_for_1d(long long begin, long long end, long long grain, const std::function<void(long long, long long)>& fn);`

实现要求：

- OpenMP backend：调用 `omp_*` 语义与现有逻辑一致。
- Portable backend：固定 worker pool + thread-local worker index。
- nested parallel 抑制：若已在并行区内，`parallel_for_1d` 退化为当前线程串行执行，避免嵌套线程爆炸。

## 4. 执行顺序（按依赖）

## Step 0：CMake 与后端选择

### 文件
- `CMakeLists.txt`

### 任务
- 新增线程后端选项，例如 `MINFER_THREAD_BACKEND`：`auto|openmp|portable`
- `auto` 行为：
  - OpenMP 可用且非 mac：默认 `openmp`
  - 其他情况：默认 `portable`
- 把 `M_HAS_OPENMP` 从“全局前提”改为“仅 OpenMP backend 编译宏”

### 验收
- mac 本地禁 OpenMP 也可构建通过
- Linux 有 OpenMP 时仍可走 OpenMP backend

## Step 1：并行 facade 落地

### 文件
- `include/minfer/parallel.h`
- `src/core/parallel*.cpp`

### 任务
- 实现统一并行 API
- Portable backend 提供稳定 worker index
- `parallel_get_num_threads()`、`parallel_set_num_threads()` 线程安全

### 验收
- 新增最小自测或在现有测试里验证：
  - `set/get_num_threads` 有效
  - `parallel_in_parallel` 在并行区内外行为正确
  - `parallel_get_worker_index` 范围合法 `[0, worker_count)`

## Step 2：system 层改为 facade

### 文件
- `include/minfer/system.h`
- `src/core/system.cpp`

### 任务
- `set_num_threads(int)` 调用 `parallel_set_num_threads`
- `get_num_threads()` 调用 `parallel_get_num_threads`
- 注释从 OpenMP 术语改为“并行后端”

### 受影响调用点
- `src/core/net.impl.cpp` 的 `initPhaseThreadPolicy()` / `applyPhaseThreads()`

### 验收
- `Net` 的 prefill/decode 线程策略行为不变
- 旧接口签名不变，不引入 API break

## Step 3：并行门控函数去 OpenMP 直连

### 文件
- `src/backend/cpu/kernel/openmp_utils.h`

### 任务
- `should_parallelize_1d_loop(...)` 内部不再调用 `omp_in_parallel`/`omp_get_max_threads`
- 改为调用 facade：
  - `parallel_in_parallel()`
  - `parallel_get_num_threads()`

### 验收
- 无 OpenMP 构建时，门控逻辑仍然基于当前线程预算工作
- 现有调用点无需改函数签名

## Step 4：benchmark 线程入口迁移

### 文件与函数
- `benchmark/benchmark_threads.h`
  - `configure_benchmark_threads(...)`
  - `configure_benchmark_max_threads()`
- `benchmark/op_benchmark.cpp`
  - `configured_threads(...)`
- `benchmark/llm_benchmark.cpp`
  - 使用 `configure_benchmark_threads` 的路径保持不变，仅底层实现切换

### 任务
- 去掉 `omp_set_dynamic/omp_set_num_threads/omp_get_max_threads/omp_get_num_procs` 直连
- 统一改为 facade API

### 验收
- `--threads` 参数仍生效
- benchmark 打印线程数和实际执行线程数一致

## Step 5：基础 kernel 并行循环迁移

PR1 目标是“基础路径可用”，不是把所有 kernel 都改成复杂通用任务图。

### 文件与函数

- `src/backend/cpu/kernel/activation_kernel_xsimd.cpp`
  - `silu_kernel_xsimd`
- `src/backend/cpu/kernel/transpose_kernel.cpp`
  - `transpose2d_tiled`
  - `transpose2d_xsimd`
  - default case transpose loop
- `src/backend/cpu/kernel/binary_kernel_xsimd.cpp`
  - `binary_broadcast_xsimd`
  - `binary_add_weighted_xsimd`（float/int32）
  - `unary_negate_xsimd`（float/int32）
- `src/backend/cpu/kernel/normalization_kernel_xsimd.cpp`
  - `softmax_lastdim_xsimd`
  - `causal_masked_softmax_square_xsimd`
  - `causal_softmax_weighted_sum_square_xsimd`
  - 其余 `#pragma omp parallel for` 循环
- `src/backend/cpu/kernel/rope_kernel.cpp`
  - `rope_kernel_inplace`（并行分支）
- `src/backend/cpu/layer/runtime_weight.cpp`
  - `RuntimeWeight::gemmNT` 的 `outer` 并行循环（`M>1` path）
- `src/core/mat_gemm.cpp`
  - `gemm_impl_naive`
  - `gemm_impl_row`

### 任务
- 将 `#pragma omp parallel for ...` 改为 `parallel_for_1d(...)`
- 保持原有门控条件（`should_parallelize_1d_loop` 阈值不变）
- 确保 loop index 类型一致（`long long`）

### 验收
- 单线程输出与迁移前一致
- 多线程输出与单线程一致
- 无死锁、无崩溃、无明显竞争

## Step 6：测试迁移

### 文件
- `test/benchmark_threads_test.cpp`

### 任务
- 去掉 `#ifdef _OPENMP` 专属断言
- 断言改为 backend-neutral：
  - `configure_benchmark_threads()` 返回值 >= 1
  - 显式线程设置后，`get_num_threads()` 与期望一致
  - `should_parallelize_1d_loop(...)` 在小工作量下返回 false
- 可选新增：
  - `test/core/parallel_backend_test.cpp`

### 验收
- 测试在 OpenMP backend 和 portable backend 都能通过

## 5. PR1 DoD（最终打勾）

- [ ] macOS 在无 OpenMP 依赖环境下构建通过
- [ ] `set_num_threads/get_num_threads` 语义稳定
- [ ] benchmark 线程参数行为不变
- [ ] 基础 kernel 并行路径已迁移到 facade
- [ ] `test/benchmark_threads_test.cpp` 已改为 backend-neutral
- [ ] `ctest` 通过
- [ ] `threads=1` 与 `threads>1` 输出一致性验证通过
- [ ] 无新增数据竞争或崩溃

## 6. 推荐验证命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

```bash
./build/minfer_test --gtest_filter='BenchmarkThreads.*:Mat_TEST.runtime_weight*:Mat_TEST.gemm_*'
```

```bash
./build/minfer_benchmark --threads 1 --prompt-lens 128 --decode-tokens 32 --warmup 1 --runs 3
./build/minfer_benchmark --threads 4 --prompt-lens 128 --decode-tokens 32 --warmup 1 --runs 3
```

```bash
./build/minfer_op_benchmark --gemm-runtime-only --threads 1 --warmup 10 --iters 30 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 512 --gemm-out 32768
./build/minfer_op_benchmark --gemm-runtime-only --threads 4 --warmup 10 --iters 30 --head-count 1 --head-count-kv 1 --batch 1 --seq-len 1 --hidden 512 --gemm-out 32768
```

## 7. 与 PR2 的边界

PR1 完成后，下面这些留给 PR2：

- `src/backend/cpu/kernel/gemm_kernel_xsimd.cpp` 的 decode scratch / worker-id 路由
- `src/backend/cpu/layer/attention_layer.cpp` 全量并行迁移
- 核心执行路径里剩余 `omp_get_thread_num()` 清零
- 阈值与调度的性能二次调优（尤其是 mac）
