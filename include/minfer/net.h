#ifndef MINFER_NET_H
#define MINFER_NET_H

#include "layer.h"

namespace minfer
{

// Op 层抽象
class Net {
public:
    Net();
    ~Net();

    // 将l添加到某一层的后面
    void addLayerTo(Layer& l, int layerId);

private:
    class NetImpl;
    NetImpl* impl; // 里面保存多种Backend，subnet，
};

}

#endif //MINFER_NET_H
