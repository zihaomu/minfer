#ifndef MINFER_NET_H
#define MINFER_NET_H

#include "layer.h"

namespace minfer
{

// Net抽象
class Net {
public:
    Net();
    ~Net();

    // create new layer, and return layerId
    int createLayer(std::shared_ptr<LayerParams> param);

private:
    class NetImpl;
    NetImpl* impl; // 里面保存多种Backend，subnet，
};

}

#endif //MINFER_NET_H
