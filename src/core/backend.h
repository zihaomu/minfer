//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_BACKEND_H
#define MINFER_BACKEND_H

#include "non_copyable.h"
#include "minfer/tensor.h"
#include "minfer/layer.h"
#include <map>
#include "string"
#include "define.impl.h"

namespace minfer
{

// Todo check if we need use Config
//struct Config
//{
//};


// Backend 包含LayerFactory，用来创建不同的layer。
class Backend : NonCopyable {
public:
    class LayerFactory {
    public:
        typedef std::shared_ptr<Layer>(*Constructor)(std::shared_ptr<LayerParams> param);
        typedef std::map<LayerType, LayerFactory::Constructor> LayerFactoryMap;

        LayerFactory();
        ~LayerFactory();

        // different backend need to register supported layer at here.
        virtual void registerAllLayer();

        // 真实创建instance
        std::shared_ptr<Layer> createLayerInstance(std::shared_ptr<LayerParams> param);
        void registerLayer(LayerType type, Constructor constructor);

        bool checkLayerSupported(LayerType type, std::shared_ptr<LayerParams> param);

    protected:
        LayerFactoryMap layerMap;
        Mutex mutex;
    };

    Backend(std::string name = ""); // 创建Backend的具体限制，比如线程数，内存限制等。
    ~Backend();

    size_t getAllMemory(); // return all alloc memory
    std::string getName();
    bool checkLayerSupported(LayerType type, std::shared_ptr<LayerParams> param);

    virtual std::shared_ptr<Layer> createLayer(std::shared_ptr<LayerParams> param) = 0;                // 创建层
    // alloc specific memory for tensor. It will use some reuse strategy.
    virtual int allocTensorMemory(Tensor* tensor, std::vector<int> shape, Tensor::DataType type) = 0; // 直接创建GPU Tensor

protected:
    std::string name;
    size_t allocMemSize;
    std::shared_ptr<LayerFactory> layerFactory;
//    Config config;
};

}


#endif //MINFER_BACKEND_H
