#include "minfer/parallel.h"

namespace minfer
{

namespace parallel_backend_openmp
{
void set_num_threads(int threads);
int get_num_threads();
int get_num_procs();
bool in_parallel();
int get_worker_index();
int get_worker_count();
void parallel_for_1d(long long begin,
                     long long end,
                     long long grain,
                     const std::function<void(long long, long long)>& fn);
}

namespace parallel_backend_portable
{
void set_num_threads(int threads);
int get_num_threads();
int get_num_procs();
bool in_parallel();
int get_worker_index();
int get_worker_count();
void parallel_for_1d(long long begin,
                     long long end,
                     long long grain,
                     const std::function<void(long long, long long)>& fn);
}

#if defined(MINFER_USE_OPENMP_BACKEND) && MINFER_USE_OPENMP_BACKEND
namespace backend = parallel_backend_openmp;
#else
namespace backend = parallel_backend_portable;
#endif

void parallel_set_num_threads(int threads)
{
    backend::set_num_threads(threads);
}

int parallel_get_num_threads()
{
    return backend::get_num_threads();
}

int parallel_get_num_procs()
{
    return backend::get_num_procs();
}

bool parallel_in_parallel()
{
    return backend::in_parallel();
}

int parallel_get_worker_index()
{
    return backend::get_worker_index();
}

int parallel_get_worker_count()
{
    return backend::get_worker_count();
}

void parallel_for_1d(long long begin,
                     long long end,
                     long long grain,
                     const std::function<void(long long, long long)>& fn)
{
    backend::parallel_for_1d(begin, end, grain, fn);
}

} // namespace minfer
