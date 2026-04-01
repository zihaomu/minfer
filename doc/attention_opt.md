一个更适合端侧的最终形态

如果你是端侧、B=1、ring-buffer、CPU/xsimd，我更建议最终走到这个版本：

KV 存储
K/V: [S_max, Hkv, D]
ring-buffer 只维护：
write_head
valid_len
decode kernel

对每个 q_head：

kv_head = q_head / repeat_kv
两段顺序扫：
A: [write_head, cap)
B: [0, write_head)
直接对 K[t][kv_head][:] 做 dot
softmax
直接对 V[t][kv_head][:] 做 weighted sum
不再出现的东西
gather_head_segment()
tmp_ka/tmp_kb/tmp_va/tmp_vb
qk_a/qk_b -> memcpy -> qk_buf
GQA repeat materialization
fallback full slice copy
一个很实际的判断

你现在这版代码适合作为：

正确性版本
过渡版本
验证 ring-buffer 语义版本

但还不是最终高性能版本。
因为它的核心还是：

用额外拷贝把数据修成 GEMM 喜欢的样子。

而端侧 decode 更好的思路通常是：

直接写一个消费原始 KV layout 的专用 kernel。

落地顺序建议

最稳的改法是：

V1

先保留 ring-buffer 逻辑不变，只做一件事：

去掉 gather_head_segment()
写 dot_strided_two_segments() 和 weighted_sum_two_segments()
V2

去掉 qk_buf

改成在线 softmax + 累加 V
V3

统一 fallback 和 mobilekv 的 KV layout

都走 [S,Hkv,D]
共用 decode kernel
V4

彻底移除显式 GQA repeat