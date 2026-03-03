#include "../../src/core/gguf_model/gguf_utils.h"
#include "../../src/core/gguf_model/gguf_loader.h"
#include <iostream>
#include <memory>

using namespace minfer;

int main() {
    auto meta = std::shared_ptr<GGUF_context>(gguf_init_from_file("../../test/big_models/Lite-Oute-1-65M-FP16.gguf"), [](GGUF_context* c){ delete c; });
    if (!meta) {
        std::cerr << "Failed to init gguf context" << std::endl;
        return 1;
    }
    float out_eps = 0;
    std::string key = "llama.attention.layer_norm_rms_epsilon";
    int kid = gguf_find_key(meta.get(), key.c_str());
    if (kid == -1) {
        std::cerr << "Key not found!" << std::endl;
        return 1;
    }
    
    std::cout << "Key ID: " << kid << std::endl;
    enum GGUF_TYPE kt = gguf_get_kv_type(meta.get(), kid);
    std::cout << "GGUF_TYPE extracted: " << GGUF_TYPE_name(kt) << " (Enum value " << (int)kt << ")" << std::endl;

    try {
        bool result = GGUFMeta::GKV<float>::set(meta.get(), key, out_eps);
        std::cout << "GKV<float>::set returned " << result << ", out_eps = " << out_eps << std::endl;
    } catch(std::exception& e) {
        std::cerr << "Exception when GKV<float>: " << e.what() << std::endl;
    }
    
    return 0;
}
