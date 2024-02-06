//
// Created by mzh on 2024/1/31.
//

#include "mat.h"
#include "iostream"

#define CV_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)

namespace opencv_lab
{

template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    assert((n & (n - 1)) == 0); // n is a power of 2
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

#define  CV_MALLOC_ALIGN    64
void* fastMalloc(size_t size)
{
    // size + one more pointer + alignment_size
    uchar* udata = (uchar*) malloc(size + sizeof(void*) + CV_MALLOC_ALIGN);
    if (!udata)
    {
        std::cerr<<"Out of memory in fastMalloc"<<std::endl;
        return nullptr;
    }

    uchar** adata = alignPtr((uchar**)udata + 1, CV_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

void fastFree(void* ptr)
{
    if (ptr)
    {
        uchar* udata = ((uchar**)ptr)[-1];
        assert(udata < (uchar*) ptr && ((uchar*)ptr - udata) <= (sizeof(void *) + CV_MALLOC_ALIGN));
        free(udata);
    }
}

// <<<<<<<<<<<<<<<<<<<<<   MatData   >>>>>>>>>>>>
MatData::MatData(const MatAllocator *_allocator)
{
    allocator = _allocator;
    refcount = 0;
    data = 0;
    size = 0;
    flags = static_cast<MatData::MemoryFlag>(0);
}

MatData::~MatData()
{
    allocator = 0;
    refcount = 0;
    data = 0;
    flags = static_cast<MatData::MemoryFlag>(0);
}


// <<<<<<<<<<<<<<<<<<<<<   StdMatAllocator   >>>>>>>>>>>>
class StdMatAllocator : public MatAllocator
{
public:

    MatData* allocate(int dims, const int* sizes, int type, void* data0) const override
    {
        size_t total = (MatType(type) == MatType::FloatType) ? sizeof(float ) : sizeof (int);

        for (int i = dims -1; i >= 0; i--)
        {
            total *= sizes[i];
        }

        uchar* data = data0 ? (uchar*)data0 : (uchar*) fastMalloc(total);
        MatData* u = new MatData(this);
        u->data = data;
        u->size = total;
        if (data0)
            u->flags = MatData::USER_MEMORY;

        return u;
    }

    // check if the MatData is allocated?
    bool allocate(MatData* u, MatDataUsageFlags) const override
    {
        if (!u) return false;
        return true;
    }

    void deallocate(MatData* u) const
    {
        if(!u)
            return;

        assert(u->refcount == 0);
        if (u->flags != MatData::USER_MEMORY)
        {
            fastFree(u->data);
            u->data = 0;
        }

        delete u;
    }
};

void MatAllocator::map(MatData*) const
{
    // do nothing
}

void MatAllocator::unmap(MatData* u) const
{
    if(u->refcount == 0)
    {
        deallocate(u);
    }
}

// <<<<<<<<<<<<<<<<<<<<<   MatSize   >>>>>>>>>>>>
inline
MatSize::MatSize(int *_p):p0(_p)
{
    p = _p + 1;
}

inline
int MatSize::dims() const
{
    return p[0];
}

inline
const int& MatSize::operator[](int i) const
{
    return p[i];
}

inline
int& MatSize::operator[](int i)
{
    return p[i];
}

inline
bool MatSize::operator!=(const opencv_lab::MatSize &sz) const
{
    return !(*this == sz);
}

bool MatSize::operator==(const opencv_lab::MatSize &sz) const
{
    int d = dims();
    int dsz = sz.dims();

    if (d != dsz)
        return false;

    for (int i = 0; i < d; i++)
    {
        if (p[i] != sz[i])
            return false;
    }

    return true;
}

// <<<<<<<<<<<<<<<<<<<<<   Mat   >>>>>>>>>>>>
// Setting the dim for mat.
void setSize(Mat& m, int _dim, const int* _sz)
{
    assert(_dim <= CV_MAX_DIM && _dim > 0);

    if (_dim != m.dims)
    {
        fastFree(m.size.p0); // free first.
        m.size.p0 = (int *) fastMalloc((_dim + 1) * sizeof( int)); // one more for dim
        m.size.p = m.size.p0 + 1;
        m.size.p0[0] = _dim;
    }

    m.dims = _dim;

    if (!_sz)
        return;

    for (int i = _dim - 1; i >= 0; i--)
    {
        int s = _sz[i];
        assert(s >=0);
        m.size.p[i] = s;
    }
}

static
MatAllocator*& getDefaultAllocatorMatRef()
{
    static MatAllocator* g_matAllocator = Mat::getStdAllocator();
    return g_matAllocator;
}

MatAllocator *Mat::getDefaultAllocator()
{
    return getDefaultAllocatorMatRef();
}

MatAllocator *Mat::getStdAllocator()
{
    static MatAllocator* const allocator = new StdMatAllocator();
    return allocator;
}

void Mat::copySize(const Mat &m)
{
    // This code will free and alloc a new buffer for m.size.
    setSize(*this, m.dims, 0);

    for (int i = 0; i < dims; i++)
    {
        size[i] = m.size[i];
    }
}

// Create empty Mat
Mat::Mat()
:flags(MAGIC_VAL), dims(0), data(0), allocator(0), u(0), size(0), matType(FloatType)
{
}

Mat::Mat(int _dims, const int* _sizes, int _type)
:flags(MAGIC_VAL), dims(0), data(0), allocator(0), u(0), size(0), matType(FloatType)
{
    create(_dims, _sizes, _type);
}

Mat::Mat(const std::vector<int>& _sizes, int _type)
:flags(MAGIC_VAL), dims(0), data(0), allocator(0), u(0), size(0), matType(FloatType)
{
    create(_sizes, _type);
}

//Mat::Mat(const Mat& m)
//:flags(m.flags), dims(m.dims), data(m.data), allocator(m.allocator), u(m.u), size(0), matType(m.matType)
//{
//
//    if (u)
//    {
//        //    std::atomic_fetch_add((_Atomic(int)*)(&u->refcount), 1);
//        CV_XADD(&u->refcount, 1);
//    }
//
//    dims = 0; // reset dims
//    copySize(m);
//}

Mat::Mat(int _dims, const int* _sizes, int _type, void* _data)
:flags(MAGIC_VAL), dims(0), data(0), allocator(0), u(0), size(&dims), matType(MatType(_type))
{
    data = (uchar*)_data;
    setSize(*this, _dims, _sizes);
}

Mat::Mat(const std::vector<int>& _sizes, int _type, void* _data)
:flags(MAGIC_VAL), dims(0), data(0), allocator(0), u(0), size(&dims), matType(MatType(_type))
{
    data = (uchar*)_data;
    setSize(*this, _sizes.size(), _sizes.data());
}

Mat::~Mat()
{
    release();
    fastFree(size.p0);
}

// Not copy mat data, only add reference counter.
//Mat& Mat::operator=(const Mat& m)
//{
//    if (this != &m) // check if there are same Mat.
//    {
//        if (m.u)
//            CV_XADD(&m.u->refcount, 1);
//
//        release(); // release this resource first.
//        copySize(m);
//
//        data = m.data;
//        allocator = m.allocator;
//        u = m.u;
//    }
//
//    return *this;
//}

// copy a full Mat, it will re-allocate memory.
void Mat::copyTo(Mat &dst) const
{
    MatType t = (MatType)dst.type();

    if (t != matType)
    {
        assert(false && "Unsupported data transform right now!");
        return;
    }

    if (empty())
    {
        dst.release();
        return;
    }

    dst.create(dims, size.p, type());
    if (data == dst.data)
        return;

    // if the data ptr is not the same,
    if (total() != 0)
    {
        int esz = sizeof(float );
        size_t total_size = total() * esz;
        memcpy(dst.data, data, total_size);
    }
}

Mat Mat::clone() const
{
    Mat m;
    copyTo(m);
    return m;
}

void Mat::create(int _dims, const int* _sizes, int _type)
{
    int i;
    assert(0 < _dims && _dims <= CV_MAX_DIM && _sizes);

    // same Mat
    if (data && (_dims == dims) && type() == _type)
    {
        for (i = 0; i < _dims; i++)
        {
            if (size.p[i] != _sizes[i]) break;
        }

        if (i == _dims)
            return; // same Mat, do not free and re-allocate
    }

    int size_back[CV_MAX_DIM];
    if (_sizes == size.p)
    {
        for (int i = 0; i < _dims; i++)
        {
            size_back[i] = size.p[i];
        }
    }

    release();

    if (_dims == 0)
        return;

    matType = (MatType)_type;
    setSize(*this, _dims, _sizes);

    // check if need to allocate memory
    if (total() > 0)
    {
        MatAllocator *a = allocator;

        if (!a)
            a = getDefaultAllocator();

        try
        {
            u = a->allocate(dims, size.p, type(), 0);
            assert(u != 0);
        }
        catch (...)
        {
            throw;
        }
    }

    addref();
}

void Mat::create(const std::vector<int>& _sizes, int _type)
{
    create(_sizes.size(), _sizes.data(), _type);
}

void Mat::release()
{
    if (u && CV_XADD(&u->refcount, -1) == 1)
    {
        deallocate();
    }

    u = NULL;
    data = 0;

    for (int i = 0; i < dims; ++i)
    {
     size.p[i] = 0;
    }
}

void Mat::deallocate()
{
    if (u)
    {
        MatData* u_ = u;
        u = NULL;
        (u_->allocator ? u_->allocator : allocator ? allocator : getDefaultAllocator())->unmap(u_);
    }
}

int Mat::type() const
{
    return (int)matType;
}

size_t Mat::total() const
{
    size_t p = 1;

    for (int i = dims-1; i >= 0; i--)
    {
        p *= size.p[i];
    }
    return p;
}

size_t Mat::total(int startDim, int endDim) const
{
    assert(startDim >= 0 && startDim <= endDim);
    size_t p = 1;

    int endDim_ = endDim <= dims ? endDim : dims;
    for (int i = startDim; i < endDim; i++)
    {
        p *= size.p[i];
    }
    return p;
}

bool Mat::empty() const
{
    return data == 0 || total() == 0 || dims == 0;
}

uchar *Mat::ptr()
{
    return data;
}

template<typename _Tp>
_Tp &Mat::at(int i0)
{
    return (_Tp*)data[i0];
}

void Mat::addref()
{
    if (u)
        CV_XADD(&u->refcount, 1);
}

}