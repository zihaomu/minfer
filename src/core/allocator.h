//
// Created by mzh on 2023/11/2.
//
// 此部分是抄 MNN
#ifndef MINFER_ALLOC_H
#define MINFER_ALLOC_H

#include "non_copyable.h"
#include "muinfer.h"
#include "memory_utils.h"
#include <vector>
#include <map>
#include <memory>

namespace mu
{

// 内存分配器，
// 所有的Tensor创建和析构都需要经过内存分配器
// 一个Net，拥有一个独立的Allocator，去分配一个net运行所需要的内存。

// 存在的问题，不能去释放不同后端的内存，这个要怎么解决？
// 复用Allocator？还是复用Tensor？
class MU_PUBLIC Allocator : NonCopyable
{
public:
    class AllocatorImpl { // Virtual class
    public:
        AllocatorImpl() = default;
        virtual ~ AllocatorImpl() = default;
        virtual std::pair<void*, size_t> alloc(size_t size, size_t align) = 0;
        virtual void release(std::pair<void*, size_t> ptr) = 0;
        static std::shared_ptr<AllocatorImpl> createDefault();
    };

    Allocator(std::shared_ptr<AllocatorImpl> impl, size_t align = MU_MEMORY_ALIGN_DEFAULT) : allocImpl(impl), mAlign(align)
    {
        // nothing
    }

    ~Allocator()
    {
        release();
    }

    // TODO 增加阻塞锁，让分配和释放都是一个线程去完成。
    /**
     * @brief free all allocated memories.
     * @sa allocSeparate
     * @sa alloc
     * if allRelease, clear all memory , otherwise delete freelist
     */
    void release(bool allRelease = true);

    /**
     * @brief query total size allocated indeed.
     * @return total size allocated indeed.
     */
    size_t totalSize() const {
        return mTotalSize;
    }

    /// 内存分配接口
    /// \param size
    /// \param align
    /// \return
    std::pair<void*, size_t> alloc(size_t size, size_t align = 0);

    bool free(std::pair<void*, size_t> pointer); // 释放内存

    void returnMemory(std::pair<void*, size_t> pointer);

    // TODO 加入多线程考虑
private:
    std::shared_ptr<AllocatorImpl> allocImpl;
    size_t mAlign;
    size_t mTotalSize = 0;

    typedef std::multimap<size_t, void*> MemoryList;
    std::pair<void*, size_t> getFromFreeList(size_t size, size_t align);
    // 使用队列
    // Why two free list?
    MemoryList freeList; // 可以用指针
    MemoryList usedList; // 使用指针
};

}

#endif //MINFER_ALLOC_H
