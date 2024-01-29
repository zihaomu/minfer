#ifndef MINFER_TENSOR_H
#define MINFER_TENSOR_H

#include <vector>
#include "define.h"

namespace minfer
{

// Tensor 是实际内存的抽象，主要参考Mat和MNN的Tensor
class Tensor
{
public:
    enum MemeoryType {
        HOST_MEMORY = 0,
        DEVICE_MEMORY = 1,
        HOST_DEVICE_MEMORY = 2,
    };

    enum DataType {
        DT_INVALID = 0,
        DT_FLOAT = 1,
        DT_DOUBLE = 2,
        DT_INT32 = 3,
        DT_UINT8 = 4,
        DT_INT16 = 5,
        DT_INT8 = 6,
        DT_STRING = 7,
        DT_INT64 = 8,
        DT_BOOL = 9,
        DT_UINT16 = 10,
        DT_QINT8 = 11,
        DT_QUINT8 = 12,
        DT_QINT32 = 13,
        DT_BFLOAT16 = 14,
        DT_QINT16 = 15,
        DT_QUINT16 = 16,
        DT_HALF = 17, // FP16 ?
    };

    Tensor();

    // 创建Tensor，如果data是nullptr，则重新分配内存。
    Tensor(const std::vector<int> &shape, DataType type, void* data = nullptr);

    Tensor(Tensor& tensor);

    ~Tensor();

    size_t total(int start = -1, int end = -1);

    // Debug info.
    void print();
    void printShape();

    // Only can be called when Tensor is Device Tensor. TODO gpu和cpu之间的交互在哪里完成？
    void copyHostToDevice();
    void copyDeviceToHost();

    // 要求，tensor shape必须一样大，且都在CPU
    Tensor add(const Tensor& tensor);
    Tensor subtract(const Tensor& tensor);
    Tensor multiply(const Tensor& tensor);
    Tensor divide(const Tensor& tensor);

    // Tensor 的基础计算
    struct TensorExtraInfo;
private:
    // remove all assignment operator， TODO check if this good?
//    Tensor(const Tensor& tensor) = delete;
//    Tensor(const Tensor&& tensor) = delete;
//    Tensor& operator=(const Tensor&) = delete;
//    Tensor& operator=(const Tensor&&) = delete;

    void* data; // 指向数据的指针。
    std::vector<int> shape; // 数据的shape。
    DataType dType;
    MemeoryType mType; // GPU Tensor必须要GPU相关的class才能创建，无法直接创建。

    std::shared_ptr<TensorExtraInfo> extraInfo; // 用于保存gpu，量化信息。
};

Tensor add(const Tensor& tensor0, const Tensor& tensor1);
Tensor subtract(const Tensor& tensor0, const Tensor& tensor1);
Tensor multiply(const Tensor& tensor0, const Tensor& tensor1);
Tensor divide(const Tensor& tensor0, const Tensor& tensor1);

}

#endif //MINFER_TENSOR_H
