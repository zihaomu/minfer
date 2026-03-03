#include "../../include/minfer.h"
#include <iostream>

using namespace minfer;

int main() {
    float x_data[] = {0.0201976, -0.00509417, -0.00165723, -0.00426197, -0.0121121};
    float w_data[] = {
        0.08435059, 0.09307861, 0.02182007, 0.26416016, -0.08007812,
        -0.13830566, 0.55126953, 0.08557129, -0.15332031, -0.06750488,
        0.3605957, -0.27026367, 0.27514648, 0.04647827, 0.1229248,
        -0.11834717, 0.34692383, -0.19018555, -0.01879883, -0.25,
        0.01831055, -0.09057617, 0.00277328, 0.19750977, -0.21582031
    };

    Mat x({1, 5}, DT_32F, x_data);
    Mat w({5, 5}, DT_32F, w_data);

    Mat out_f = gemm(x, w, false, false);
    Mat out_t = gemm(x, w, false, true);

    float* pf = (float*)out_f.data;
    float* pt = (float*)out_t.data;

    std::cout << "gemm(false, false) first 5: ";
    for (int i=0; i<5; ++i) std::cout << pf[i] << " ";
    std::cout << "\ngemm(false, true) first 5: ";
    for (int i=0; i<5; ++i) std::cout << pt[i] << " ";
    std::cout << std::endl;

    return 0;
}
