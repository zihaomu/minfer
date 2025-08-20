# this file contains python test data generated code.
# It's used to generate test input and output code for C++ layer defined in layer.h

import numpy as np
import math
import sys
import os

ROOT_PATH = "./data"
# random seed
np.random.seed(0)

def generate_lower_triangular_mask(N):
    mask = np.tril(np.ones((N, N), dtype=bool))
    return mask

# set with the maximum number of sequences
MASK = generate_lower_triangular_mask(256)
print(MASK)

def printArray(data, name = None):
    d = data.flatten()
    print(name, " shape:", data.shape, ", v: ", d[:10])


class RMSNorm:
    def __init__(self, dim : int, eps=1e-6):
        self.eps = eps
        self.dim = dim
        self.weight = None

    def load(self, params):
        self.weight = params

    def _norm(self, x):
        return x * np.reciprocal(np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps))

    def forward(self, x):
        assert self.weight is not None, "rms norm not initialized"

        out = self._norm(x)
        printArray(out, "RMSNorm before")
        data = out * self.weight
        printArray(data, "RMSNorm")
        return data
        # return out * self.weight

class Linear:
    def __init__(self, in_features, out_features, hasBias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.hasBias = hasBias
        self.weight = None
        self.bias = None

    def forward(self, x):
        assert self.weight is not None, "linear not initialized"
        if self.hasBias:
            return np.matmul(x, self.weight.T) + self.bias
        else:
            printArray(x, "Linear input")
            printArray(self.weight, "Linear weight")
            printArray(self.weight.T, "Linear weight .T")
            data = np.matmul(x, self.weight.T)
            
            # printArray(data[:, 1:, :], "Linear 1 output")
            # printArray(data[:, 2:, :], "Linear 2 output")
            return np.matmul(x, self.weight.T)

    def load(self, params):
        # if params is a list, then it's [weight, bias]
        if isinstance(params, list):
            self.weight = params[0].reshape(self.out_features, self.in_features)
            if self.hasBias:
                self.bias = params[1].reshape(self.out_features)
        else:
            assert not self.hasBias, "linear with bias should have two params"
            self.weight = params.reshape(self.out_features, self.in_features)

    def parameters(self):
        return [self.weight, self.bias] if self.hasBias else [self.weight]

# softmax implementation
class Softmax:
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        x = x - np.max(x, axis=self.dim, keepdims=True)
        x = np.exp(x)
        x = x / np.sum(x, axis=self.dim, keepdims=True)
        return x

# convert word to embedding
# the vocab_size = 2048, and the feature size = 288.
class WordEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = None

    def forward(self, x):
        assert self.weight is not None, "embedding not initialized"
        return self.weight[x]# * math.sqrt(self.d_model)

    def load(self, params):
        self.weight = params.reshape(self.vocab_size, self.d_model)

    def parameters(self):
        return [self.weight]

# Compute Rope position encoding.
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** ((np.arange(0, dim, 2)) / dim)) # dim = 288, freqs.shape = (144,)

    t = np.arange(end) # end = 256

    freqs = np.outer(t, freqs) # freqs.shape = (256, 144), 这里的256是sequence 长度，而不是词表长度。sequence长度不足时需要被padding起来。
    freqs_sin = np.sin(freqs)
    freqs_cos = np.cos(freqs)
    return freqs_sin, freqs_cos

# The following function is to broadcast the freqs_cis to the shape of x.
def reshape_for_broadcast(freqs_cis, x):
    ndim = len(x.shape)
    # print("code in reshape for broadcast = ", freqs_cis.shape, (x.shape[1], x.shape[-1]))
    assert (freqs_cis.shape == (x.shape[1], x.shape[-1]))
    freqs_cis = freqs_cis[np.newaxis, :, np.newaxis, :]
    assert (len(freqs_cis.shape) == len(x.shape))
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_sin, freqs_cos):

    xq_r, xq_i = xq[..., 0::2], xq[..., 1::2] # 拆成实部和虚部
    xk_r, xk_i = xk[..., 0::2], xk[..., 1::2]

    printArray(xq_r, "xq_r")
    # freqs_sin shape = (256, 144)
    # xq_r shape = (1, 6, 24)

    print("before reshape_for_broadcast")
    printArray(freqs_sin, "freqs_sin")
    printArray(xq_r, "xq_r")

    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)

    printArray(freqs_sin, "freqs_sin")
    printArray(freqs_cos, "freqs_cos")
    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    printArray(xq_out_r, "xq_out_r")
    printArray(xq_out_i, "xq_out_i")
    # flatten last two dimensions

    xq_out = np.stack((xq_out_r, xq_out_i), axis=-1) # shape: (1, 256, 8, 8, 2)
    # printArray(xq_out, "xq_out 0")
    xq_out = xq_out.reshape(1, xq_out.shape[1], xq_out.shape[2], xq_out.shape[3] * xq_out.shape[4]) # shape: (1, 256, 8, 16)
    # printArray(xq_out, "xq_out 1")
    # stack 之后，维度从 (1, 1, 6, 24)变成(1, 1, 6, 24, 2)，需要将最后两个维度融合。
    xk_out = np.stack((xk_out_r, xk_out_i), axis=-1)
    # 下面输出维度为：(1, 1, 6, 24 * 2)
    xk_out = xk_out.reshape(1, xk_out.shape[1], xk_out.shape[2], xk_out.shape[3] * xk_out.shape[4])

    printArray(xq_out, "xq_out")
    # sys.exit(0)
    return xq_out, xk_out

# test
# precompute_freqs_cis(48, 256)

# Scalling Dot-Product Attention
class Attention:
    def __init__(self):
        self.soft_max = Softmax(dim=-1) # Keep

    def forward(self, query, key, value):
        d_k = query.shape[-1] # d_k是每个头的维度 # Keep
        keyT = np.transpose(key, axes=(0, 1, 3, 2))

        '''
        query: 1 x num_heads x seq_len x d_k
        keyT: 1 x num_heads x d_k x seq_len
        np.matmul(query, keyT): 1 x num_heads x seq_len x seq_len
        '''
        ddata = np.matmul(query, keyT)
        # printArray(ddata, "query keyT")
        
        scores = np.matmul(query, keyT) / math.sqrt(d_k)
        printArray(scores, "scores T")
        
        mask = MASK[:scores.shape[-2], :scores.shape[-1]]
        printArray(mask, "mask")
        scores = scores * mask - 1e20 * (1 - mask)
        printArray(scores, "scores 2")
        printArray(scores[:, 1:, :, :], "scores 22")
        printArray(scores[:, 2:, :, :], "scores 22")
        
        scores = self.soft_max.forward(scores)
        printArray(scores[:, 1:, :, :], "scores 22")
        printArray(scores[:, 2:, :, :], "scores 22")
        printArray(scores[:, 3:, :, :], "scores 22")
        return np.matmul(scores, value)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # number of dimensions per head
        self.q_linear = Linear(d_model, d_model, False)
        self.k_linear = Linear(d_model, d_model, False)
        self.v_linear = Linear(d_model, d_model, False)
        self.out_linear = Linear(d_model, d_model, False)

        self.attn = Attention()

    def load(self, params):
        self.q_linear.load(params[0])
        self.k_linear.load(params[1])
        self.v_linear.load(params[2])
        self.out_linear.load(params[3])

    def forward(self, x, freqs_sin, freqs_cos):
        bsz, seq_len, _ = x.shape
        print("seq_len = ", seq_len, ", x shape = ", x.shape, "self.d_k = ", self.d_k, "self.num_heads = ", self.num_heads)
        assert self.q_linear.weight is not None, "multi-head attention not initialized"

        q = self.q_linear.forward(x).reshape(1, seq_len, self.num_heads, self.d_k)
        k = self.k_linear.forward(x).reshape(1, seq_len, self.num_heads, self.d_k)
        v = self.v_linear.forward(x).reshape(1, seq_len, self.num_heads, self.d_k)
        
        q, k = apply_rotary_emb(q, k, freqs_sin, freqs_cos)
        
        printArray(q[:, 1:, :, :], "q0 reshape")
        printArray(k[:, 1:, :, :], "k0 reshape")
        
        # printArray(q, "xq")
        # Transpose for attention dot product: (bsz, num_heads, seq_len, d_k)
        v = np.transpose(v, axes=(0, 2, 1, 3)) # 1 x num_heads x seq_len x d_k
        q = np.transpose(q, axes=(0, 2, 1, 3)) # from  shape: (1, 256, 8, 16) to (1, 8, 256, 16)
        k = np.transpose(k, axes=(0, 2, 1, 3))

        q1 = q[:, 1:, :, :]
        k1 = k[:, 1:, :, :]
        printArray(q, "q reshape")
        printArray(k, "k reshape")
        printArray(v, "v reshape")
        printArray(q1, "q1 reshape")
        printArray(k1, "k1 reshape")
        

        # print("xq = ", q.shape)
        # print("xk = ", k.shape)
        # print("xv = ", v.shape)

        # 为啥新的不需要mask？
        scores = self.attn.forward(q, k, v)
        printArray(scores, "scores")
        concat = np.transpose(scores, axes=(0, 2, 1, 3)).reshape(1, seq_len, self.d_model)
        printArray(concat[:, 0:, :], "concat reshape")
        printArray(concat[:, 1:, :], "concat reshape")
        printArray(concat[:, 2:, :], "concat reshape")
        printArray(concat[:, 3:, :], "concat reshape")
        printArray(concat[:, 4:, :], "concat reshape")
        output = self.out_linear.forward(concat)
        printArray(output, "atten out")
        return output

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff, hasBias=False)
        self.linear2 = Linear(d_ff, d_model, hasBias=False)
        self.linear3 = Linear(d_model, d_ff, hasBias=False)
        self.relu = lambda x : np.maximum(0, x)
        self.silu = lambda x : x / (1 + np.exp(-x))
        # self.relu = np.vectorize(lambda x: max(0, x))

    def load(self, params):
        self.linear1.load(params[0])
        self.linear2.load(params[1])
        self.linear3.load(params[2])

    def forward(self, x):
        data1 = self.linear2.forward(self.silu(self.linear1.forward(x)) * self.linear3.forward(x))
        data2 = x + data1
        # data2 = self.linear3.forward(x)
        # print("data 1 shape = ", data1.shape)
        # print("data 2 shape = ", data2.shape)
        # return data1 * data2
        printArray(data1, "ff out")
        return data2
        # return self.linear2.forward(self.silu(self.linear1.forward(x))) * self.linear3.forward(x)

def Attention_layer_data_generater():
    # random attention layer
    d_model = 128
    num_heads = 8
    max_len = 256
    seq_len = 256

    atten_norm = RMSNorm(d_model)
    atten = MultiHeadAttention(d_model, num_heads)

    # random params
    params = []
    params.append(np.random.rand(d_model, d_model))
    params.append(np.random.rand(d_model, d_model))
    params.append(np.random.rand(d_model, d_model))
    params.append(np.random.rand(d_model, d_model))

    atten.load(params)

    params_atten_norm = np.random.rand(d_model)
    atten_norm.load(params_atten_norm)

    # random input
    x = np.random.rand(1, seq_len, d_model)
    printArray(x, "Attention input")

    # precompute freqs_cis
    freqs_sin, freqs_cos = precompute_freqs_cis(d_model // num_heads, max_len)
    
    # forward
    out = atten.forward(atten_norm.forward(x), freqs_sin, freqs_cos) + x
    printArray(out, "Attention output")
    printArray(out[:, 1:, :], "Attention output 1")
    printArray(out[:, 2:, :], "Attention output 1")
    printArray(out[:, 3:, :], "Attention output 1")

    # save input
    np.save(ROOT_PATH + "/atten_input.npy", x.astype(np.float32))

    # save params
    np.save(ROOT_PATH + "/atten_params_0.npy", np.array(params[0]).astype(np.float32))
    np.save(ROOT_PATH + "/atten_params_1.npy", np.array(params[1]).astype(np.float32))
    np.save(ROOT_PATH + "/atten_params_2.npy", np.array(params[2]).astype(np.float32))
    np.save(ROOT_PATH + "/atten_params_3.npy", np.array(params[3]).astype(np.float32))
    np.save(ROOT_PATH + "/atten_rms_params.npy", np.array(params_atten_norm).astype(np.float32))

    # # save output
    np.save(ROOT_PATH + "/atten_output.npy", out.astype(np.float32))

def FeedForward_layer_data_generater():

    # random FFN layer
    d_model = 128
    d_ff = 256
    ffn_norm = RMSNorm(d_model)
    ffn = FeedForward(d_model, d_ff)

    # random params
    params = []
    params.append(np.random.rand(d_ff, d_model))
    params.append(np.random.rand(d_model, d_ff))
    params.append(np.random.rand(d_ff, d_model))

    ffn.load(params)

    params_ffn_norm = np.random.rand(d_model)
    ffn_norm.load(params_ffn_norm)
    
    # random input
    x = np.random.rand(1, 6, d_model)
    printArray(x, "FFN input")

    # forward
    out = ffn.forward(ffn_norm.forward(x)) + x
    # printArray(out, "FFN output")

    # check if the output folder is exist
    if not os.path.exists(ROOT_PATH):
        os.makedirs(ROOT_PATH)

    # store all data in npy
    # input
    np.save(ROOT_PATH + "/ffn_input.npy", x.astype(np.float32))

    # params
    np.save(ROOT_PATH + "/ffn_params_0.npy", np.array(params[0]).astype(np.float32))
    np.save(ROOT_PATH + "/ffn_params_1.npy", np.array(params[1]).astype(np.float32))
    np.save(ROOT_PATH + "/ffn_params_2.npy", np.array(params[2]).astype(np.float32))
    np.save(ROOT_PATH + "/ffn_rms_params.npy", np.array(params_ffn_norm).astype(np.float32))

    # output
    np.save(ROOT_PATH + "/ffn_output.npy", out.astype(np.float32))

def generate_random_numpy_npy():
    data = np.random.rand(10, 12).astype(np.float32)
    print(data)
    np.save(ROOT_PATH + "/random10x12.npy", data)

def WordEmbedding_layer_data_generater():
    # random word embedding layer
    d_model = 128
    vocab_size = 2048
    word_emb = WordEmbedding(vocab_size, d_model)

    # random numpy array with shape [vocab_size, d_model]
    params = np.random.rand(vocab_size, d_model)

    printArray(params, "WordEmbedding params")
    word_emb.load(params)

    # random input
    x = np.random.randint(0, vocab_size, (1, 10))
    printArray(x, "WordEmbedding input")

    # forward
    out = word_emb.forward(x)
    printArray(out, "WordEmbedding output")

    # save input
    np.save(ROOT_PATH + "/word_emb_input.npy", x.astype(np.int32))

    # save params
    np.save(ROOT_PATH + "/word_emb_params_0.npy", params.astype(np.float32))

    # output
    np.save(ROOT_PATH + "/word_emb_output.npy", out.astype(np.float32))

def RMSNorm_layer_data_generater():
    # random RMSNorm layer
    d_model = 128
    rms_norm = RMSNorm(d_model)

    # random params
    params = np.random.rand(d_model)

    printArray(params, "RMSNorm params")
    rms_norm.load(params)

    # random input
    x = np.random.rand(1, 6, d_model)
    printArray(x, "RMSNorm input")

    # forward
    out = rms_norm.forward(x)
    printArray(out, "RMSNorm output")

    # save input
    np.save(ROOT_PATH + "/rms_norm_input.npy", x.astype(np.float32))

    # save params
    np.save(ROOT_PATH + "/rms_norm_params.npy", params.astype(np.float32))

    # output
    np.save(ROOT_PATH + "/rms_norm_output.npy", out.astype(np.float32))

def Linear_layer_data_generater():
    # random Linear layer
    input_feature = 128
    output_feature = 256
    linear = Linear(input_feature, output_feature, hasBias=True)

    # random params
    params = []
    params.append(np.random.rand(input_feature, output_feature))
    params.append(np.random.rand(output_feature))

    linear.load(params)

    # random input
    x = np.random.rand(1, 6, input_feature)
    printArray(x, "Linear input")

    # forward
    out = linear.forward(x)
    printArray(out, "Linear output")

    # save input
    np.save(ROOT_PATH + "/linear_input.npy", x.astype(np.float32))

    # save params
    np.save(ROOT_PATH + "/linear_params_0.npy", params[0].astype(np.float32))
    np.save(ROOT_PATH + "/linear_params_1.npy", params[1].astype(np.float32))

    # output
    np.save(ROOT_PATH + "/linear_output.npy", out.astype(np.float32))

def main():
    # FeedForward_layer_data_generater()
    # generate_random_numpy_npy()
    # Attention_layer_data_generater()
    # WordEmbedding_layer_data_generater()
    # RMSNorm_layer_data_generater()
    Linear_layer_data_generater()

main()