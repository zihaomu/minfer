#include "../src/core/gguf_model/gguf_loader.h"
#include <iostream>

using namespace minfer;

int ____main() {
    std::vector<std::shared_ptr<LayerParams>> netParams;
    std::shared_ptr<GGUF_Vocab> gguf_vocab;
    readGGUF("test/big_models/Lite-Oute-1-65M-FP16.gguf", netParams, gguf_vocab);
    
    for (auto& p : netParams) {
        if (p->type == LayerType::RMSNorm) {
            auto rp = std::static_pointer_cast<RMSNormLayerParams>(p);
            std::cout << "Parsed RMSNormLayer Eps: " << rp->rms_eps << std::endl;
            break;
        }
    }
    return 0;
}
