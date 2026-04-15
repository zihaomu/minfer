#include "minfer/parallel.h"

#include <algorithm>

#if defined(MINFER_USE_OPENMP_BACKEND) && MINFER_USE_OPENMP_BACKEND
#include <omp.h>
#endif

namespace minfer
{
namespace parallel_backend_openmp
{

void set_num_threads(int threads)
{
#if defined(MINFER_USE_OPENMP_BACKEND) && MINFER_USE_OPENMP_BACKEND
    if (threads > 0)
    {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
#else
    (void)threads;
#endif
}

int get_num_threads()
{
#if defined(MINFER_USE_OPENMP_BACKEND) && MINFER_USE_OPENMP_BACKEND
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

int get_num_procs()
{
#if defined(MINFER_USE_OPENMP_BACKEND) && MINFER_USE_OPENMP_BACKEND
    return std::max(1, omp_get_num_procs());
#else
    return 1;
#endif
}

bool in_parallel()
{
#if defined(MINFER_USE_OPENMP_BACKEND) && MINFER_USE_OPENMP_BACKEND
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

int get_worker_index()
{
#if defined(MINFER_USE_OPENMP_BACKEND) && MINFER_USE_OPENMP_BACKEND
    return in_parallel() ? omp_get_thread_num() : 0;
#else
    return 0;
#endif
}

int get_worker_count()
{
#if defined(MINFER_USE_OPENMP_BACKEND) && MINFER_USE_OPENMP_BACKEND
    return in_parallel() ? std::max(1, omp_get_num_threads()) : 1;
#else
    return 1;
#endif
}

void parallel_for_1d(long long begin,
                     long long end,
                     long long grain,
                     const std::function<void(long long, long long)>& fn)
{
    if (!fn || end <= begin)
    {
        return;
    }

    const long long block = std::max(1LL, grain);

#if defined(MINFER_USE_OPENMP_BACKEND) && MINFER_USE_OPENMP_BACKEND
    if (in_parallel() || get_num_threads() <= 1)
    {
        for (long long s = begin; s < end; s += block)
        {
            fn(s, std::min(end, s + block));
        }
        return;
    }

    const long long total = end - begin;
    const long long chunks = (total + block - 1) / block;

#pragma omp parallel for schedule(static)
    for (long long chunk = 0; chunk < chunks; ++chunk)
    {
        const long long s = begin + chunk * block;
        const long long e = std::min(end, s + block);
        fn(s, e);
    }
#else
    for (long long s = begin; s < end; s += block)
    {
        fn(s, std::min(end, s + block));
    }
#endif
}

} // namespace parallel_backend_openmp
} // namespace minfer
