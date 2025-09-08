//
// Created by mzh on 2024/1/24.
//

#include "linear_layer.h"

namespace minfer {

LinearLayer::LinearLayer(const std::shared_ptr<LinearLayerParams> param)
{
    M_Assert(param->type == LayerType::Linear);
    getBasicInfo(param);

    MatShape w_shape = param->w.shape();
    M_Assert(w_shape.size() == 2);
    in_features = param->in_features;
    out_features = param->out_features;

    param->w.convertTo(w, DT_32F);

    if (w_shape[0] == out_features && w_shape[1] == in_features)
    {
        // 这种情况是[out_features, in_features]
        transposeW = true;
    }

    if (!param->b.empty())
    {
        M_Assert(param->b.shape().size() == 1 && param->b.shape()[0] == out_features);
        param->b.convertTo(b, DT_32F);
    }
}

LinearLayer::~LinearLayer()
{

}

void LinearLayer::init(const std::vector<Mat*> &input, std::vector<Mat*> &output)
{
    M_Assert(input.size() == output.size() && input.size() == 1);

    MatShape in_shape = input[0]->shape();
    MatShape w_shape = w.shape();
    MatShape output_shape = get_gemm_shape(in_shape, w_shape);

    output[0]->setSize(output_shape);
}

void LinearLayer::forward(const std::vector<Mat*> &input, std::vector<Mat*> &output)
{
    M_Assert(input.size() == 1 && input[0]);
    M_Assert(output.size() == 1 && output[0]);

    M_Assert(input[0]->type() == output[0]->type());

    Mat x = *input[0];
    Mat out = *output[0];

    // check input shape
    MatShape in_shape = x.shape();
    M_Assert(in_shape.size() == 3);
    M_Assert(in_shape[0] == 1 && "Currently, only support single batch!");
    M_Assert(in_shape[2] == in_features);

    // gemm: y = alpha * A * B + beta * C
    Mat out_tmp;
    gemm(x, w, false, transposeW).copyTo(out);

    // std::cout<<"out"<<std::endl;
    // out.print(10);
    if (!b.empty())
        out = out + b;
    // out.print(10);

    // out_tmp.copyTo(out);
}

std::shared_ptr<LinearLayer> LinearLayer::create(const std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<LinearLayerParams> l_param = std::dynamic_pointer_cast<LinearLayerParams>(param);

    M_Assert(l_param && "LinearLayerParams is empty!");
    M_Assert(l_param->type == LayerType::Linear);

    return std::shared_ptr<LinearLayer>(new LinearLayer(l_param));

}

}