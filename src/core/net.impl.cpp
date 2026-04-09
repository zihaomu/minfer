//
// Created by mzh on 2024/1/22.
//

#include "net.impl.h"
#include "gguf_model/gguf_loader.h"
#include "mobilekv/kv_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <climits>
#include <string>

namespace minfer
{

namespace {

int parse_positive_int_env(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0')
    {
        return 0;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > INT_MAX)
    {
        return 0;
    }
    return static_cast<int>(parsed);
}

bool env_policy_enabled()
{
    const char* policy = std::getenv("MINFER_PHASE_THREAD_POLICY");
    if (!policy || policy[0] == '\0')
    {
        return true;  // default auto policy
    }

    std::string policy_str(policy);
    std::transform(policy_str.begin(), policy_str.end(), policy_str.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return !(policy_str == "0" || policy_str == "off" || policy_str == "false" || policy_str == "no");
}

int clamp_threads(int threads, int base_threads)
{
    const int base = std::max(base_threads, 1);
    if (threads <= 0)
    {
        return base;
    }
    return std::max(1, std::min(threads, base));
}

}  // namespace

Net::NetImpl::NetImpl()
{
    gguf_vocab = std::shared_ptr<GGUF_Vocab>(new GGUF_Vocab());
    if (runtime == nullptr)
    {
        runtime = Runtime::getRuntime();
    }
}

Net::NetImpl::~NetImpl()
{

}

void Net::NetImpl::readNet(const std::string path,
                           RuntimePrecision precision,
                           const std::string modelType,
                           const std::string kv_cache_cfg_path)
{
    setRuntimePrecision(precision);
    if (!kv_cache_cfg_path.empty())
    {
        kv_cache_cfg_path_ = kv_cache_cfg_path;
        kv_cache_cfg_text_.clear();
    }
    // TODO Add model model type supported!
    std::vector<std::shared_ptr<LayerParams> > netParams;
    M_Assert(modelType == "gguf" && "Only GGUF model has been supported!");

    readGGUF(path, netParams, gguf_vocab);
    buildMobileKVStorage(netParams);

    createNet(netParams);
}

std::string Net::NetImpl::maybeCreateAutoMobileKVCfgText(int num_attention_layers,
                                                          int num_heads_kv,
                                                          int head_dim,
                                                          int max_seq_len) const
{
    M_Assert(num_attention_layers > 0);
    M_Assert(num_heads_kv > 0);
    M_Assert(head_dim > 0);
    M_Assert(max_seq_len > 0);

    std::string cfg_text;
    cfg_text += "model num_heads=" + std::to_string(num_heads_kv) + " head_dim=" + std::to_string(head_dim) + "\n";
    cfg_text += "storage default_alignment=64 thread_safe=false default_max_seq_capacity=" +
                std::to_string(max_seq_len) + "\n";
    cfg_text += "defaults k_type=fp32 v_type=fp32 initial=" + std::to_string(max_seq_len) +
                " max=" + std::to_string(max_seq_len) + "\n";
    cfg_text += "group 0-" + std::to_string(num_attention_layers - 1) + "\n";
    return cfg_text;
}

void Net::NetImpl::buildMobileKVStorage(std::vector<std::shared_ptr<LayerParams> >& netParams)
{
    kv_storage_.reset();

    const char* kv_backend = std::getenv("MINFER_KV_BACKEND");
    if (kv_backend && std::string(kv_backend) == "legacy")
    {
        kv_cache_cfg_text_.clear();
        return;
    }

    std::vector<std::shared_ptr<AttentionLayerParams> > attn_params;
    attn_params.reserve(netParams.size());
    for (auto& param : netParams)
    {
        if (param->type != LayerType::Attention)
        {
            continue;
        }
        auto attn = std::dynamic_pointer_cast<AttentionLayerParams>(param);
        M_Assert(attn);
        attn_params.push_back(attn);
    }
    if (attn_params.empty())
    {
        return;
    }

    const auto& first = attn_params[0];
    M_Assert(first->head_count > 0);
    M_Assert(first->head_count_kv > 0);
    M_Assert(first->embd_dim % first->head_count == 0);

    const int num_attention_layers = static_cast<int>(attn_params.size());
    const int num_heads_kv = first->head_count_kv;
    const int head_dim = first->embd_dim / first->head_count;
    const int max_seq_len = first->max_seq_len;

    for (const auto& attn : attn_params)
    {
        M_Assert(attn->head_count > 0);
        M_Assert(attn->head_count_kv == num_heads_kv);
        M_Assert(attn->embd_dim % attn->head_count == 0);
        M_Assert(attn->embd_dim / attn->head_count == head_dim);
        M_Assert(attn->max_seq_len == max_seq_len);
    }

    std::string cfg_error;
    std::unique_ptr<mobilekv::KVCacheStorage> kv_storage_unique;
    const std::string cfg_path = kv_cache_cfg_path_;
    if (!cfg_path.empty())
    {
        kv_storage_unique = mobilekv::create_storage_from_config_file(cfg_path, &cfg_error);
        if (!kv_storage_unique)
        {
            M_Error_(Error::StsError, ("create_storage_from_config_file failed: path=%s, error=%s",
                                       cfg_path.c_str(), cfg_error.c_str()));
        }
    }
    else
    {
        kv_cache_cfg_text_ = maybeCreateAutoMobileKVCfgText(
            num_attention_layers, num_heads_kv, head_dim, max_seq_len);
        kv_storage_unique = mobilekv::create_storage_from_config_string(kv_cache_cfg_text_, &cfg_error);
        if (!kv_storage_unique)
        {
            M_Error_(Error::StsError, ("create_storage_from_config_string failed: error=%s",
                                       cfg_error.c_str()));
        }
    }

    if (!kv_storage_unique)
    {
        M_Error_(Error::StsError, ("Fail to build mobilekv storage."));
    }
    M_Assert(kv_storage_unique);

    kv_storage_ = std::shared_ptr<mobilekv::KVCacheStorage>(std::move(kv_storage_unique));
    M_Assert(kv_storage_);

    for (int i = 0; i < num_attention_layers; ++i)
    {
        M_Assert(kv_storage_->has_layer(static_cast<uint32_t>(i)));
        attn_params[i]->kv_storage = kv_storage_;
        attn_params[i]->kv_cache_layer_id = i;
    }
}

void Net::NetImpl::setInput(const Mat input, const int _mIndx)
{
    int mIndx = _mIndx;
    if (mIndx == -1)
    {
        M_Assert(inputMatId.size() == 1);
        mIndx = inputMatId[0];
    }

    auto it = std::find(inputMatId.begin(), inputMatId.end(), mIndx);
    const int index = it - inputMatId.begin();

    if (hasInit)
    {
        M_Assert(!inputMatClone[index].empty() && "Input Mat has not been set!");

        if (inputMatClone[index].size != input.size || inputMatClone[index].type() != input.type())
        {
            // 输入的Mat和之前的Mat不一样，需要重新初始化，重新分配内存
            hasInit = false;
        }
    }
    inputMatClone[index] = input.clone();

    // Update input Mat pointer.
    auto itLayerId = matId2layer.find(mIndx);
    M_Assert(itLayerId != matId2layer.end());

    auto& ld = lds[itLayerId->second];
    if (ld.layer->getType() == LayerType::Input)
    {
        M_Assert(ld.inputsIdx.size() == 1);

        ld.inputs[0] = &inputMatClone[index];

        auto itM = mats.find(index);
        M_Assert(itM != mats.end());

        itM->second = &inputMatClone[index];
    }
}

void Net::NetImpl::forward(Mat& out)
{
    if (!hasInit)
    {
        this->init();
        hasInit = true;
    }
    out = this->forward();
}

Mat Net::NetImpl::forward()
{
    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        if (benchmarkEnabled_)
        {
            auto t0 = std::chrono::steady_clock::now();
            it->layer->forward(it->inputs, it->outputs);
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            profiler_.recordForward(it->layerId, us);
        }
        else
        {
            it->layer->forward(it->inputs, it->outputs);
        }
    }

    M_Assert(outputMatId.size() == 1);

    Mat* m = this->getMat(outputMatId[0]);
    M_Assert(m && "m can not be empty!");
    return *m;
}

void Net::NetImpl::init()
{
    // build layerId 2 Custom
    // TODO 优化下面代码，添加forward layer order部分优化，找到最佳的forward顺序。
    // 为什么需要调用两次这个代码？考虑到创建是的layerId和Params中的LayerId是不一样的，创建好的和实际运行速度最优的layer order也是不一样。
    // 所以这两个不一样需要调用两次这部分代码
    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        std::vector<int> currCustom = {};

        // 如果只有一个输出，只需要找哪些层需要这个输出作为输入就行
        const std::vector<int>& outputIndex = it->outputsIdx;

        for (int i = 0; i < outputIndex.size(); i++)
        {
            int currOutIdx = outputIndex[i];
            for (auto it2 = lds.begin(); it2 != lds.end(); it2++)
            {
                const std::vector<int> inputIndex = it2->inputsIdx;
                auto itFind = std::find(inputIndex.begin(), inputIndex.end(), currOutIdx);
                if (itFind != inputIndex.end())
                {
                    currCustom.push_back(it2->layerId);
                }
            }
        }

        it->layerCustomers = currCustom;
    }

    // TODO 考虑将下面这段代码加入GPU
    // 建立Mat的使用表格，能达到最好的复用策略。
    // Mat表格指的是创建的Mat能在哪一层被释放
    std::map<int, int> matReleaseAtLayer; // mat Id to layerId
    for (auto it = mats.begin(); it != mats.end(); it++)
    {
        auto itLy = matId2layer.find(it->first);
        M_Assert(itLy != matId2layer.end());

        auto itLd = lds[itLy->second];

        if (itLd.layerCustomers.size() == 0)
            matReleaseAtLayer[it->first] = -1;
        else
        {
            int lastCustomId = -1;
            for (int i = 0; i < itLd.layerCustomers.size(); i++)
            {
                // ⚠️ 这里把最后一个包含此Mat的customId为释放的flag
                int customId = itLd.layerCustomers[i];
                if (customId > lastCustomId)
                {
                    // 查找这个custome layer的input是否包含这个Mat
                    auto ld = lds[customId];

                    const std::vector<int>& inputsIds = ld.inputsIdx;
                    if (std::find(inputsIds.begin(), inputsIds.end(), it->first) != inputsIds.end())
                    {
                        lastCustomId = customId;
                    }
                }
            }

            // 找到这个mat能在那一层运行完之后被释放
            matReleaseAtLayer[it->first] = lastCustomId;
        }
    }

    // 调用runtime 分配和释放内存
    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        it->layer->init(it->inputs, it->outputs); // 计算shape

        // 分配内存
        for (int i = 0; i < it->outputsIdx.size(); i++)
        {
            runtime->allocMat(it->outputs[i]);

            int outId = it->outputsIdx[i];
            auto it2 = matReleaseAtLayer.find(outId);

            M_Assert(it2 != matReleaseAtLayer.end());

            if (it2->second == it->layerId)
            {
                runtime->deallocMat(it->outputs[i]); // 回收当前资源
            }
        }
        //
        // // 释放不用的资源，查找释放flag，确定是否在当前layerId释放
        // for (int i = 0; i < it->outputsIdx.size(); i++)
        // {
        //     int outId = it->outputsIdx[i];
        //     auto it2 = matReleaseAtLayer.find(outId);
        //
        //     M_Assert(it2 != matReleaseAtLayer.end());
        //
        //     if (it2->second == it->layerId)
        //     {
        //         runtime->deallocMat(it->outputs[i]); // 回收当前资源
        //     }
        // }
    }
    hasInit = true;

    // Initialize profiler with layer info
    profiler_.resize((int)lds.size());
    for (const auto& ld : lds)
    {
        profiler_.setLayerInfo(ld.layerId, ld.layer->getName(), ld.layer->getType());
    }
}

void Net::NetImpl::createLayerRecurve(int layerIdx, std::vector<int>& isLayerCreated, const std::map<int,
        std::vector<int> >& layer2Parent, const std::vector<std::shared_ptr<LayerParams> >& allLayerParams)
{
    if (isLayerCreated[layerIdx])
    {
        return;
    }

    // create layer's parents first
    auto it = layer2Parent.find(layerIdx);
    M_Assert(it != layer2Parent.end());

    for (int i = 0; i < it->second.size(); i++)
    {
        createLayerRecurve(it->second[i], isLayerCreated, layer2Parent, allLayerParams);
    }

    if (createLayer(allLayerParams[layerIdx]) >= 0)
    {
        isLayerCreated[layerIdx] = 1;
    }
}

// 此函数保证在 allLayerParams乱序情况下，仍然能够让模型从input层一层层创建，从而让后面层的创建滞后于前面的层。
// 此部分代码有待测试
void Net::NetImpl::createNet(const std::vector<std::shared_ptr<LayerParams> >& allLayerParams)
{
    M_Assert(!graphCreated_ && "Net has already been created, precision can not change after createNet/readNet.");
    // find every layer's parent layer index.
    std::vector<int> outLayerIndex;
    std::map<int, std::vector<int> > layer2Parent; // 建立layer -> parent 的映射
    for (int i = 0; i < allLayerParams.size(); i++)
    {
        std::vector<int> curParent = {};
        auto& cur = allLayerParams[i];

        // loop cur layer's input mat, and find the layer index which output these mat.
        for (int k = 0; k < cur->inputIndex.size(); k++)
        {
            int currInIdx = allLayerParams[i]->inputIndex[k];
            for (int j = 0; j < allLayerParams.size(); j++)
            {
                auto itFind = std::find(allLayerParams[j]->outputIndex.begin(), allLayerParams[j]->outputIndex.end(), currInIdx);

                if (itFind != allLayerParams[j]->outputIndex.end())
                {
                    curParent.push_back(j);
                }
            }
        }

        layer2Parent[i] = curParent;
        if (cur->type == Output)
        {
            outLayerIndex.push_back(i);
        }
    }

    M_Assert(outLayerIndex.size() > 0 && "Model is broken, it does not have output!!");

    std::vector<int> isLayerCreated(allLayerParams.size(), 0);
    // 递归的调用createLayerParents，建立是否创建表格。
    for (int i = 0; i < outLayerIndex.size(); i++)
    {
        createLayerRecurve(outLayerIndex[i], isLayerCreated, layer2Parent, allLayerParams);
    }
}

void Net::NetImpl::createNet(const std::vector<std::shared_ptr<LayerParams> >& allLayerParams, RuntimePrecision precision)
{
    setRuntimePrecision(precision);
    createNet(allLayerParams);
}

int Net::NetImpl::createLayer(std::shared_ptr<LayerParams> param)
{
    AutoLock lk(mutex);
    // TODO 对inputlayer和outputlayer的特殊处理

    // Check if the input layer has been created.
    int inputSize = param->inputIndex.size();
    int outputSize = param->outputIndex.size();

    LayerData ld = {};
    int layerId = lds.size();
    param->precision = runtimePrecision_;
    std::shared_ptr<Layer> layer = runtime->createLayer(param);

    if (!layer)
    {
        M_Error_(Error::Code::StsBadType, ("Fail to create layer instance with type = %d!", (int)param->type));
    }
    // 对输入输出对特殊处理
    // 输入将会在setinput中进行初始化。
    if (param->type == LayerType::Input)
    {
        inputMatClone.push_back(Mat());
        inputMatId.push_back(param->inputIndex[0]);
        M_Assert(param->inputIndex.size() == 1);
        mats[param->inputIndex[0]] = &inputMatClone[inputMatClone.size() - 1];
        inputLayers.push_back(layerId);
        matId2layer[param->inputIndex[0]] = layerId;
    }
    else if (param->type == LayerType::Output)
    {
        M_Assert(param->outputIndex.size() == 1);
        outputMatId.push_back(param->outputIndex[0]);
        outputLayers.push_back(layerId);
    }

    // 每一个层只管理自己的outputMat，而inputMat是由上面传下来的
    std::vector<Mat*> inps(inputSize, nullptr);
    for (int i = 0; i < inputSize; ++i)
    {
        int inputId = param->inputIndex[i];
        Mat* m = getMat(inputId);
        M_Assert(m && "The input Mat has not been created!");
        inps[i] = m;
    }

    layer->setId(layerId);
    std::vector<Mat*> outs(outputSize, nullptr);
    for (int i = 0; i < outputSize; ++i)
    {
        int outputMatId = param->outputIndex[i];
        outs[i] = new Mat();
        mats[outputMatId] = outs[i];
        matId2layer[outputMatId] = layerId;
    }

    ld.layerId = layerId;
    ld.layer = layer;
    ld.inputs = inps;
    ld.inputsIdx = param->inputIndex;
    ld.outputs = outs;
    ld.outputsIdx = param->outputIndex;

    lds.push_back(ld);
    graphCreated_ = true;

    // Register layer info for profiler
    profiler_.setLayerInfo(layerId, layer->getName(), layer->getType());
    // Create the layer.
    return layerId;
}

Mat *Net::NetImpl::getMat(const int matIdx)
{
    auto it = mats.find(matIdx);

    if (it != mats.end())
    {
        return it->second;
    }
    else
    {
        return nullptr;
    }
}

void Net::NetImpl::getMats(const std::vector<int> matsIdx, std::vector<Mat *> &mats)
{
    mats.clear();
    mats.resize(matsIdx.size(), nullptr);

    for (int i = 0; i < matsIdx.size(); i++)
    {
        mats[i] = getMat(matsIdx[i]);
    }
}

void Net::NetImpl::decode(const std::vector<int> &out_ids, std::string &out_text)
{
    M_Assert(gguf_vocab && "gguf_vocab is empty, can not decode!");
    gguf_vocab->decode(out_ids, out_text);
}

void Net::NetImpl::encode(const std::string text, std::vector<int> &out_ids)
{
    M_Assert(gguf_vocab && "gguf_vocab is empty, can not encode!");
    gguf_vocab->encode(text, out_ids);
}

void Net::NetImpl::setRuntimePrecision(RuntimePrecision precision)
{
    M_Assert(!graphCreated_ &&
             "Net precision is immutable after createNet/readNet");
    runtimePrecision_ = precision;
}

RuntimePrecision Net::NetImpl::getRuntimePrecision() const
{
    return runtimePrecision_;
}

void Net::NetImpl::setKVCacheConfigPath(const std::string& cfg_path)
{
    M_Assert(!graphCreated_ &&
             "KV cache cfg path is immutable after createNet/readNet");
    kv_cache_cfg_path_ = cfg_path;
    kv_cache_cfg_text_.clear();
}

void Net::NetImpl::initPhaseThreadPolicy()
{
    if (phaseThreadPolicyInited_)
    {
        return;
    }
    phaseThreadPolicyInited_ = true;

    const int base_threads = std::max(get_num_threads(), 1);
    phaseThreadPolicyEnabled_ = env_policy_enabled();
    if (!phaseThreadPolicyEnabled_)
    {
        prefillThreads_ = base_threads;
        decodeThreads_ = base_threads;
        return;
    }

    const int prefill_env = parse_positive_int_env("MINFER_PREFILL_THREADS");
    const int decode_env = parse_positive_int_env("MINFER_DECODE_THREADS");

    const int prefill_auto = std::min(base_threads, 8);
    const int decode_auto = std::min(base_threads, 16);

    prefillThreads_ = clamp_threads(prefill_env > 0 ? prefill_env : prefill_auto, base_threads);
    decodeThreads_ = clamp_threads(decode_env > 0 ? decode_env : decode_auto, base_threads);

    if (prefillThreads_ == base_threads && decodeThreads_ == base_threads)
    {
        phaseThreadPolicyEnabled_ = false;
    }
}

void Net::NetImpl::applyPhaseThreads(InferPhase phase)
{
    initPhaseThreadPolicy();
    if (!phaseThreadPolicyEnabled_)
    {
        return;
    }

    const int target_threads = (phase == InferPhase::Prefill) ? prefillThreads_ : decodeThreads_;
    if (target_threads <= 0 || activePhaseThreads_ == target_threads)
    {
        return;
    }

    set_num_threads(target_threads);
    activePhaseThreads_ = target_threads;
}

// ====== Chat 生成接口实现 ======

Mat Net::NetImpl::prefill(const std::vector<int>& token_ids)
{
    int seq_len = token_ids.size();
    M_Assert(seq_len > 0 && "Token ids must not be empty!");

    // 设置推理上下文
    ctx_.phase = InferPhase::Prefill;
    ctx_.start_pos = 0;
    ctx_.seq_len = seq_len;

    applyPhaseThreads(InferPhase::Prefill);

    // 构造输入 Mat: token ids as int Mat [1, seq_len]
    std::vector<int> input_shape = {1, seq_len};
    Mat input_mat(input_shape, DT_32S, (void*)token_ids.data());

    // 设置输入并初始化
    this->setInput(input_mat, -1);

    if (!hasInit)
    {
        this->init();
        hasInit = true;
    }

    // 遍历所有层，使用带 context 的 forward
    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        if (benchmarkEnabled_)
        {
            auto t0 = std::chrono::steady_clock::now();
            it->layer->forward(it->inputs, it->outputs, ctx_);
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            profiler_.record(it->layerId, InferPhase::Prefill, us);
        }
        else
        {
            it->layer->forward(it->inputs, it->outputs, ctx_);
        }
    }

    // 更新 start_pos
    ctx_.start_pos += seq_len;

    M_Assert(outputMatId.size() == 1);
    Mat* m = this->getMat(outputMatId[0]);
    M_Assert(m && "Output Mat can not be empty!");
    return *m;
}

Mat Net::NetImpl::step(int token_id)
{
    // 设置推理上下文
    ctx_.phase = InferPhase::Decode;
    ctx_.seq_len = 1;

    applyPhaseThreads(InferPhase::Decode);

    // 构造输入 Mat: single token id [1, 1]
    std::vector<int> input_shape = {1, 1};
    Mat input_mat(input_shape, DT_32S, (void*)&token_id);

    // 直接更新输入数据，绕过 setInput 的 size 检查以避免 hasInit 被重置
    M_Assert(inputMatId.size() == 1);
    inputMatClone[0] = input_mat.clone();

    // 更新输入指针
    int mIndx = inputMatId[0];
    auto itLayerId = matId2layer.find(mIndx);
    M_Assert(itLayerId != matId2layer.end());
    auto& ld_input = lds[itLayerId->second];
    ld_input.inputs[0] = &inputMatClone[0];
    mats[0] = &inputMatClone[0];

    // 重新 init 所有层的 shape（不重新分配整个网络，只更新 shape 和分配 output）
    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        it->layer->init(it->inputs, it->outputs);
        for (int i = 0; i < (int)it->outputsIdx.size(); i++)
        {
            // 仅当 output Mat 为空时才分配内存
            if (it->outputs[i]->empty())
            {
                Runtime::getRuntime()->allocMat(it->outputs[i]);
            }
        }
    }

    // 遍历所有层，使用带 context 的 forward
    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        if (benchmarkEnabled_)
        {
            auto t0 = std::chrono::steady_clock::now();
            it->layer->forward(it->inputs, it->outputs, ctx_);
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            profiler_.record(it->layerId, InferPhase::Decode, us);
        }
        else
        {
            it->layer->forward(it->inputs, it->outputs, ctx_);
        }
    }

    // 更新 start_pos
    ctx_.start_pos += 1;

    M_Assert(outputMatId.size() == 1);
    Mat* m = this->getMat(outputMatId[0]);
    M_Assert(m && "Output Mat can not be empty!");
    return *m;
}

void Net::NetImpl::resetKVCache()
{
    ctx_.start_pos = 0;
    ctx_.seq_len = 0;
    ctx_.phase = InferPhase::Prefill;

    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        it->layer->resetKVCache();
    }
}

// ====== Benchmark Profiling ======

void Net::NetImpl::enableBenchmark(bool enable)
{
    benchmarkEnabled_ = enable;
}

void Net::NetImpl::printBenchmark() const
{
    profiler_.printReport();
}

void Net::NetImpl::resetBenchmark()
{
    profiler_.reset();
}

}
