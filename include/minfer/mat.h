//
// Created by mzh on 2024/3/27.
//

#ifndef MINFER_MAT_H
#define MINFER_MAT_H

#include <iostream>
#include <assert.h>

#include "define.h"
#include "system.h"
#include "libnpy/npy.hpp"

namespace minfer
{

typedef std::vector<int> MatShape;
#define MAT_MAX_DIM 8 // Max mat dimension

class MatAllocator;
struct MatData;
class Mat;
class MatOp;
class MatExpr;

enum MatDataUsageFlags
{
    USAGE_DEFAULT = 0,
    USAGE_ALLOCATE_HOST_MEMORY = 1 << 0,
    USAGE_ALLOCATE_DEVICE_MEMORY = 1 << 1,
};

// API for mat memory allocator
// 支持不同后端，需要支持不同的Mat
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

// Mat class, 从OpenCV处抄过来
class Mat
{
public:
    Mat();

    Mat(int dims, const int* sizes, int type);

    Mat(const std::vector<int> sizes, int type);

    // create specific dimension Mat with given default value.
    // Note the value is int type, if you wanna set the default value as other type
    // please reinterpret_cast it to int value first.
    Mat(int dims, const int* sizes, int type, int v);

    Mat(const std::vector<int> sizes, int type, int v);

    // when use reference mat, we need to create new memory for the mat size.
    Mat(const Mat& m);

    Mat(int dims, const int* sizes, int type, void* data);

    Mat(const std::vector<int> sizes, int type, void* data);

    ~Mat();

    // No data copy, just add reference counter.
    Mat& operator=(const Mat& m);

    // Mat& operator+=(const Mat& m);
    // Mat& operator-=(const Mat& m);
    // Mat& operator/=(const Mat& m);
    // Mat& operator*=(const Mat& m);

    Mat& operator=(const MatExpr& e);

    // set all mat value to given value, it will not change the mat type, just assign value according the data type.
    Mat& operator=(const float v);

    // set all mat value to given value
    Mat& operator=(const int v);

    // Create a full copy of the array and the underlying data.
    Mat clone() const;

    // reshape the mat to new shape
    Mat reshape(int newDims, const int* newSizes) const;

    Mat reshape(const std::vector<int> newSizes) const;

    void copyTo(Mat& m) const;

    // convert mat to other mat with specific data type.
    void convertTo(Mat& m, int rtype) const;

    void create(int ndims, const int* sizes, int type);

    void create(const std::vector<int> sizes, int type);

    void release();

    void deallocate();

    int type() const;

    // 只设置 Mat的shape而不分配内存
    // ⚠️ 这个有可能打破Mat的安全性，double check
    void setSize(std::vector<int> size);

    void setSize(int dim, const int* size);

    // 设置和输入Mat 一样的shape
    void setSize(Mat& m);

    // 将Mat中的所有数值设置到给定的v中，注意，内部会将float v转换成Mat所对应的数据类型。
    void setTo(float v);

    size_t total() const;

    size_t total(int startDim, int endDim=INT_MAX) const;

    // print the mat value and shape.
    void print(int len = -1) const;

    MatShape shape() const;

    bool empty() const;

    void copySize(const Mat& m);

    uchar* ptr();

    template<typename _Tp> _Tp& at(int i0 = 0);

    // This a atomic operation. The method increments the reference counter associated with the matrix data.
    void addref();

    static MatAllocator* getStdAllocator();
    static MatAllocator* getDefaultAllocator();

    int dims;
    uchar* data;

    MatAllocator* allocator;

    MatData* u;
    MatSize size;
    int matType;

    struct MatExtrInfo;
    std::shared_ptr<MatExtrInfo> extrInfo; // 保存backend的信息, TODO 暂时没用到
};

///////////////////////////////// Matrix Expressions //////////////

// virtual class for computation operator
class MatOp
{
public:
    MatOp();
    virtual ~MatOp();

    virtual bool elementWise(const MatExpr& expr) const;
    virtual void assign(const MatExpr& expr, Mat& m, int type = -1) const = 0;

    // augment assign? for compound assignment, +=, -=, *= ...
    virtual void augAssginAdd(const MatExpr& expr, Mat& m) const;
    virtual void augAssginSubtract(const MatExpr& expr, Mat& m) const;
    virtual void augAssginMultiply(const MatExpr& expr, Mat& m) const;
    virtual void augAssginDivide(const MatExpr& expr, Mat& m) const;

    virtual void add(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void subtract(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void multiply(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void divide(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;

    virtual int type(const MatExpr& expr) const;
};

class MatExpr
{
public:
    MatExpr();
    explicit MatExpr(const Mat& m);

    MatExpr(const MatOp* _op, int _flags, const Mat& _a = Mat(), const Mat& _b = Mat(),
            const Mat& _c = Mat(), double _alpha = 1, double _beta = 1);

    operator Mat() const;

    int type() const;

    const MatOp* op;
    int flags;

    Mat a, b, c;
    double alpha, beta;
};

MatExpr operator + (const Mat& a, const Mat& b);
MatExpr operator + (const Mat& a, const MatExpr& e);
MatExpr operator + (const MatExpr& e, const Mat& b);
MatExpr operator + (const MatExpr& e1, const MatExpr& e2);
MatExpr operator + (const MatExpr& e1, const float e);
MatExpr operator + (const float e, const MatExpr& e2);
MatExpr operator + (const Mat& a, const float e);
MatExpr operator + (const float e, const Mat& b);

MatExpr operator - (const Mat& a);
MatExpr operator - (const MatExpr& e);
MatExpr operator - (const Mat& a, const Mat& b);
MatExpr operator - (const Mat& a, const MatExpr& e);
MatExpr operator - (const MatExpr& e, const Mat& b);
MatExpr operator - (const MatExpr& e1, const MatExpr& e2);
MatExpr operator - (const MatExpr& e1, const float e);
MatExpr operator - (const float e, const MatExpr& e2);
MatExpr operator - (const Mat& a, const float e);
MatExpr operator - (const float e, const Mat& b);

MatExpr operator * (const Mat& a, const Mat& b);
MatExpr operator * (const Mat& a, const MatExpr& e);
MatExpr operator * (const MatExpr& e, const Mat& b);
MatExpr operator * (const MatExpr& e1, const MatExpr& e2);
MatExpr operator * (const MatExpr& e1, const float e);
MatExpr operator * (const float e, const MatExpr& e2);
MatExpr operator * (const Mat& a, const float e);
MatExpr operator * (const float e, const Mat& b);

MatExpr operator / (const Mat& a, const Mat& b);
MatExpr operator / (const Mat& a, const MatExpr& e);
MatExpr operator / (const MatExpr& e, const Mat& b);
MatExpr operator / (const MatExpr& e1, const MatExpr& e2);
MatExpr operator / (const MatExpr& e1, const float e);
MatExpr operator / (const float e, const MatExpr& e2);
MatExpr operator / (const Mat& a, const float e);
MatExpr operator / (const float e, const Mat& b);

MatExpr operator == (const Mat& a, const Mat& b);

// compute shape.
size_t total(const Mat& m);
size_t total(const Mat& m, int startDim, int endDim = -1);
size_t total(const MatShape shape);
size_t total(const MatShape shape, int startDim, int endDim = -1);

// for fast gemm
Mat gemm(const Mat& a, const Mat& b, bool transA = false, bool transB = false);

// read data from given path and re-construct it to Mat.
Mat readMatFromNpy(const std::string& path);

// TODO Mat inv, and other type mat operator.
}

#include "./mat.inl.h"

#endif //MINFER_MAT_H
