//
// Created by mzh on 2023/11/2.
//

#include "allocator.h"
#include "define.impl.h"
#include "memory_utils.h"

namespace minfer
{

class DefaultAllocatorImpl : public Allocator::AllocatorImpl
{
public:
    DefaultAllocatorImpl()
    {
        // Do nothing
    }

    virtual ~ DefaultAllocatorImpl()
    {
        // Do nothing
    }

    virtual std::pair<void*, size_t> alloc(size_t size, size_t align)
    {
        size_t newSize = UP_DIV(size, align) * align; // 返回对齐之后的内存。
        return std::make_pair(MMemoryAllocAlign(size, align), newSize);
    }

    virtual void release(std::pair<void*, size_t> ptr)
    {
        M_ASSERT(ptr.second == 0);
        MMemoryFreeAlign(ptr.first);
    }
};

std::shared_ptr<Allocator::AllocatorImpl> Allocator::AllocatorImpl::createDefault() {
    std::shared_ptr<Allocator::AllocatorImpl> _res;
    _res.reset(new DefaultAllocatorImpl);
    return _res;
}

void Allocator::release(bool allRelease)
{
    if (allRelease)
    {
        usedList.clear();
        freeList.clear();

        mTotalSize = 0;

        return;
    }

    for (auto f : freeList)
    {
        if (f.second == nullptr)
        {
            M_ASSERT(mTotalSize >= f.first);
            mTotalSize -= f.first;
        }
    }
    freeList.clear();
}

std::pair<void *, size_t> Allocator::alloc(size_t size, size_t align)
{
    if (align == 0)
        align = mAlign;

    std::pair<void*, size_t> pointer;

    pointer = getFromFreeList(size, align);
    if (pointer.first != nullptr)
        return pointer;

    // alloc otherwise
    pointer = allocImpl->alloc(size, align);

    // alloc fail!
    if (nullptr == pointer.first)
    {
        return pointer;
    }

    mTotalSize += size;

    usedList.insert(std::make_pair(pointer.second, pointer.first));
    M_ASSERT(pointer.second % align == 0);
    return pointer;
}

std::pair<void *, size_t> Allocator::getFromFreeList(size_t size, size_t align)
{
    size_t realSize = size;
    bool needExtraSize = mAlign % align != 0;

    if (needExtraSize)
        realSize = (realSize + align - 1);

    auto x = freeList.lower_bound(realSize);

    if (x == freeList.end())
        return std::make_pair(nullptr, 0);
    else
    {
        auto p = *x;
        usedList.insert(p);
        freeList.erase(x);
        return std::make_pair(p.second, p.first);
    }
}

bool Allocator::free(std::pair<void *, size_t> pointer)
{
    bool foundUsed = false;
    auto x = usedList.begin();
    for (auto i = usedList.begin(); i != usedList.end(); ++i)
    {
        if (i->second == pointer.first)
        {
            foundUsed = true;
            x = i;
        }
    }

    if (foundUsed)
    {
        usedList.erase(x);
        returnMemory(pointer);
    }
    else
    {
        return false;
    }

    return true;
}

void Allocator::returnMemory(std::pair<void *, size_t> pointer)
{
    freeList.insert(std::make_pair(pointer.second, pointer.first));
}

}