#include "minfer/parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace minfer
{
namespace parallel_backend_portable
{

namespace
{

int detect_num_procs()
{
    const unsigned int hw_threads = std::thread::hardware_concurrency();
    return hw_threads == 0 ? 1 : static_cast<int>(hw_threads);
}

thread_local int t_parallel_depth = 0;
thread_local int t_worker_index = 0;
thread_local int t_worker_count = 1;

class ParallelRegionScope
{
public:
    ParallelRegionScope(int worker_index, int worker_count)
        : prev_depth_(t_parallel_depth),
          prev_worker_index_(t_worker_index),
          prev_worker_count_(t_worker_count)
    {
        t_parallel_depth = prev_depth_ + 1;
        t_worker_index = std::max(0, worker_index);
        t_worker_count = std::max(1, worker_count);
    }

    ~ParallelRegionScope()
    {
        t_parallel_depth = prev_depth_;
        t_worker_index = prev_worker_index_;
        t_worker_count = prev_worker_count_;
    }

private:
    int prev_depth_;
    int prev_worker_index_;
    int prev_worker_count_;
};

class PortableThreadPool
{
public:
    PortableThreadPool()
        : max_workers_(std::max(0, detect_num_procs() - 1))
    {
        workers_.reserve(static_cast<size_t>(max_workers_));
        for (int worker_index = 1; worker_index <= max_workers_; ++worker_index)
        {
            workers_.emplace_back([this, worker_index]() { worker_loop(worker_index); });
        }
    }

    ~PortableThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_task_.notify_all();
        for (auto& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    int max_worker_count() const
    {
        return max_workers_ + 1;
    }

    template<class TaskFn>
    void run(int worker_count, const TaskFn& task_fn)
    {
        const int active_worker_count = std::max(1, std::min(worker_count, max_worker_count()));
        if (active_worker_count <= 1)
        {
            ParallelRegionScope scope(0, 1);
            task_fn(0, 1);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_worker_count_ = active_worker_count;
            pending_workers_ = active_worker_count - 1;
            worker_exception_ = nullptr;
            task_ = [&task_fn](int worker_index, int worker_total) { task_fn(worker_index, worker_total); };
            ++job_epoch_;
        }
        cv_task_.notify_all();

        std::exception_ptr main_exception;
        try
        {
            ParallelRegionScope scope(0, active_worker_count);
            task_fn(0, active_worker_count);
        }
        catch (...)
        {
            main_exception = std::current_exception();
        }

        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [this]() { return pending_workers_ == 0; });
        if (!worker_exception_ && main_exception)
        {
            worker_exception_ = main_exception;
        }
        std::exception_ptr final_exception = worker_exception_;
        task_ = nullptr;
        lock.unlock();

        if (final_exception)
        {
            std::rethrow_exception(final_exception);
        }
    }

private:
    void worker_loop(int worker_index)
    {
        uint64_t observed_epoch = 0;
        for (;;)
        {
            std::function<void(int, int)> task;
            int worker_total = 1;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_task_.wait(lock, [&]() { return stopping_ || job_epoch_ != observed_epoch; });
                if (stopping_)
                {
                    return;
                }
                observed_epoch = job_epoch_;
                worker_total = active_worker_count_;
                task = task_;
                if (!task || worker_index >= worker_total)
                {
                    continue;
                }
            }

            std::exception_ptr local_exception;
            try
            {
                ParallelRegionScope scope(worker_index, worker_total);
                task(worker_index, worker_total);
            }
            catch (...)
            {
                local_exception = std::current_exception();
            }

            std::lock_guard<std::mutex> lock(mutex_);
            if (local_exception && !worker_exception_)
            {
                worker_exception_ = local_exception;
            }
            if (pending_workers_ > 0)
            {
                --pending_workers_;
                if (pending_workers_ == 0)
                {
                    cv_done_.notify_one();
                }
            }
        }
    }

    const int max_workers_;
    std::vector<std::thread> workers_;

    mutable std::mutex mutex_;
    std::condition_variable cv_task_;
    std::condition_variable cv_done_;
    bool stopping_ = false;
    uint64_t job_epoch_ = 0;

    int active_worker_count_ = 1;
    int pending_workers_ = 0;
    std::function<void(int, int)> task_;
    std::exception_ptr worker_exception_;
};

const int g_num_procs = detect_num_procs();
std::atomic<int> g_num_threads{std::max(1, g_num_procs)};
PortableThreadPool g_pool;

inline void run_serial_chunks(long long begin,
                              long long end,
                              long long block,
                              const std::function<void(long long, long long)>& fn)
{
    for (long long start = begin; start < end; start += block)
    {
        fn(start, std::min(end, start + block));
    }
}

} // namespace

void set_num_threads(int threads)
{
    if (threads > 0)
    {
        g_num_threads.store(threads, std::memory_order_relaxed);
    }
}

int get_num_threads()
{
    return std::max(1, g_num_threads.load(std::memory_order_relaxed));
}

int get_num_procs()
{
    return std::max(1, g_num_procs);
}

bool in_parallel()
{
    return t_parallel_depth > 0;
}

int get_worker_index()
{
    return in_parallel() ? t_worker_index : 0;
}

int get_worker_count()
{
    return in_parallel() ? t_worker_count : 1;
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
    const long long chunks = (end - begin + block - 1) / block;
    if (chunks <= 1 || in_parallel() || get_num_threads() <= 1)
    {
        run_serial_chunks(begin, end, block, fn);
        return;
    }

    const int requested_threads = get_num_threads();
    const int active_worker_count = static_cast<int>(
        std::min<long long>(chunks, std::min<long long>(requested_threads, g_pool.max_worker_count())));
    if (active_worker_count <= 1)
    {
        run_serial_chunks(begin, end, block, fn);
        return;
    }

    g_pool.run(active_worker_count, [&](int worker_index, int worker_total) {
        const long long workers = static_cast<long long>(worker_total);
        const long long base = chunks / workers;
        const long long rem = chunks % workers;

        const long long worker_idx_ll = static_cast<long long>(worker_index);
        const long long chunk_begin = worker_idx_ll * base + std::min(worker_idx_ll, rem);
        const long long chunk_count = base + (worker_idx_ll < rem ? 1 : 0);
        const long long chunk_end = chunk_begin + chunk_count;

        for (long long chunk = chunk_begin; chunk < chunk_end; ++chunk)
        {
            const long long chunk_start = begin + chunk * block;
            const long long chunk_stop = std::min(end, chunk_start + block);
            fn(chunk_start, chunk_stop);
        }
    });
}

} // namespace parallel_backend_portable
} // namespace minfer
