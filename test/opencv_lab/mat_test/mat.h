//
// Created by mzh on 2024/1/30.
//

#ifndef MINFER_MAT_H
#define MINFER_MAT_H

#include <iostream>
#include "assert.h"
// Task: implement a basic Mat class. It should include some basic function like, create, assignment, some basic computation like sub, add.
// And Mat should support at least two type: float and int.

namespace opencv_lab
{

typedef unsigned char uchar;
#define CV_MAX_DIM 8
class MatAllocator;
struct MatData;
class Mat;

enum MatType
{
    FloatType = 0,
    IntType = 1,
};

enum MatDataUsageFlags
{
    USAGE_DEFAULT = 0,
    USAGE_ALLOCATE_HOST_MEMORY = 1 << 0,
    USAGE_ALLOCATE_DEVICE_MEMORY = 1 << 1,
};

// API for mat memory allocator
class MatAllocator
{
public:
    MatAllocator() {}
    virtual ~MatAllocator() {}

    virtual MatData* allocate(int dims, const int* sizes, int type, void* data) const = 0;
    virtual bool allocate(MatData* data, MatDataUsageFlags usageFlags) const = 0;
    virtual void deallocate(MatData* data) const = 0;
    virtual void map(MatData* data) const;
    virtual void unmap(MatData* data) const;
};


// Some basic information
struct MatData
{
    enum MemoryFlag
    {
        HOST_MEMORY = 1,
        DEVICE_MEMORY = 2,
        HOST_DEVICE_MEMORY = 3,
        USER_MEMORY = 4,
    };

    MatData(const MatAllocator* allocator);
    ~MatData();

    const MatAllocator* allocator;
    int refcount;
    uchar* data;

    size_t size;
    MatData::MemoryFlag flags;
};

struct MatSize
{
    MatSize(int* _p);
    int dims() const;
    const int& operator[](int i) const;
    int& operator[](int i);
    bool operator == (const MatSize& sz) const;
    bool operator != (const MatSize& sz) const;

    int* p0; // p0[0] is dim
    int* p;  // p is p0+1
};

// Mat class
class Mat
{
public:
    Mat();

    Mat(int dims, const int* sizes, int type);

    Mat(const std::vector<int>& sizes, int type);

//    Mat(const Mat& m);

    Mat(int dims, const int* sizes, int type, void* data);

    Mat(const std::vector<int>& sizes, int type, void* data);

    ~Mat();

    // No data copy, just add reference counter.
//    Mat& operator=(const Mat& m);

    // Create a full copy of the array and the underlying data.
    Mat clone() const;

    void copyTo(Mat& m) const;

    void create(int ndims, const int* sizes, int type);

    void create(const std::vector<int>& sizes, int type);

    // Question: why have release and deallocate at the same time?

    void release();

    void deallocate();

    int type() const;

    size_t total() const;

    size_t total(int startDim, int endDim=INT_MAX) const;

    bool empty() const;

    void copySize(const Mat& m);

    uchar* ptr();

    template<typename _Tp> _Tp& at(int i0 = 0);

    enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0};
    // This a atomic operation. The method increments the reference counter associated with the matrix data.
    void addref();


    static MatAllocator* getStdAllocator();
    static MatAllocator* getDefaultAllocator();

    int flags;
    int dims;
    uchar* data;

    MatAllocator* allocator;

    MatData* u;
    MatSize size;
    MatType matType;
};

}

#endif //MINFER_MAT_H
