// Finish a chat example based on Minfer

#include "minfer.h"

using namespace minfer;

int main()
{
    Net net;
    net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf");

    // prompt
    std::string text = "Hello world! <s>";
    std::vector<int> ids;

    // tokenizer
    net.encode(text, ids);

    net.setInput(ids);
    net.init();

    Mat output = net.forward();

    std::vector<int> token_ids = argmax_tokens(reinterpret_cast<const float*>(output.data), output.size[0], output.size[1], output.size[2]);

    std::string out_text;
    net.decode(token_ids, out_text);

    std::cout << "Output: ";
    output.print();

    std::cout << "Decoded Text: " << out_text << std::endl;

    return 0;
}