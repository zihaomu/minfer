//
// Created by mzh on 2023/11/3.
//

#ifndef MINFER_AUTOBUFFER_H
#define MINFER_AUTOBUFFER_H

#include <stdint.h>
#include <string.h>
#include "memory_utils.h"

namespace minfer
{

// 不用手动释放的内存块，但是不会走内存复用
template <typename T>
class AutoBuffer
{
public:
    AutoBuffer() : mSize(0), mData(nullptr)
    {
    }

    AutoBuffer(int size)
    {
        mData = (T*) MMemoryAllocAlign(sizeof(T) * size, M_MEMORY_ALIGN_DEFAULT);
        mSize = size;
    }

    ~AutoBuffer()
    {
        if (mData != nullptr)
        {
            MMemoryFreeAlign(mData);
        }
    }

    inline int size() const
    {
        return mSize;
    }

    void set(T* data, int size)
    {
        if (mData != nullptr && mData != data)
        {
            MMemoryFreeAlign(mData);
        }

        mData = data;
        mSize = size;
    }

    void release()
    {
        if (mData != nullptr)
        {
            MUMemoryFreeAlign(mData);
            mData = nullptr;
            mSize = 0;
        }
    }

    void clear()
    {
        ::memset(mData, 0, mSize * sizeof(T));
    }

    T* data() const
    {
        return mData;
    }

private:
    size_t mSize;
    T* mData;
};

// Auto Release Class
template <typename T>
class AutoRelease
{
public:
    AutoRelease(T* d = nullptr)
    {
        mData = d;
    }

    ~AutoRelease()
    {
        if (mData != nullptr)
            delete mData;
    }

    AutoRelease(const AutoRelease&) = delete;
    T* operator->()
    {
        return mData;
    }

    void reset(T* d)
    {
        if (mData != nullptr)
        {
            delete mData;
        }
        mData = d;
    }

    T* data()
    {
        return mData;
    }

    const T* data() const
    {
        return mData;
    }
private:
    T* mData = nullptr;
};

class RefCount
{
public:
    void addRef() const
    {
        mNum++;
    }

    void decRef() const
    {
        --mNum;
        if (0 >= mNum)
            delete this;
    }

    inline int cout() const
    {
        return mNum;
    }

protected:
    RefCount() : mNum(1){}
    RefCount(const RefCount& f) : mNum(f.mNum) {}

    void operator=(const RefCount& f)
    {
        if (this != &f)
        {
            mNum = f.mNum;
        }
    }

    virtual ~RefCount() {}

private:
    mutable int mNum;
};

}

#endif //MINFER_AUTOBUFFER_H
