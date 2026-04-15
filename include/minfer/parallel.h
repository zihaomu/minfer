#ifndef MINFER_PARALLEL_H
#define MINFER_PARALLEL_H

#include <functional>

namespace minfer
{

void parallel_set_num_threads(int threads);
int parallel_get_num_threads();
int parallel_get_num_procs();

bool parallel_in_parallel();
int parallel_get_worker_index();
int parallel_get_worker_count();

void parallel_for_1d(long long begin,
                     long long end,
                     long long grain,
                     const std::function<void(long long, long long)>& fn);

} // namespace minfer

#endif // MINFER_PARALLEL_H
