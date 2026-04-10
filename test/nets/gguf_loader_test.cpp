#include "gtest/gtest.h"

#include "../src/core/gguf_model/gguf_loader.h"

using namespace minfer;

TEST(GGUFLoader_TEST, final_output_projection_uses_lm_head_params)
{
    std::vector<std::shared_ptr<LayerParams>> netParams;
    std::shared_ptr<GGUF_Vocab> gguf_vocab = std::make_shared<GGUF_Vocab>();

    readGGUF(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", netParams, gguf_vocab);

    ASSERT_GE(netParams.size(), 2u);
    EXPECT_EQ(netParams[netParams.size() - 2]->type, LayerType::LmHead);
    EXPECT_EQ(netParams.back()->type, LayerType::Output);
}
