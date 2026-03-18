# FasterTransformer 单次推理流程详解

本文以 **ParallelGpt** 模型为例，详细分析一次完整的推理流程。重点关注两种典型场景的区别：
- **冷启动 (Cold Start)**: 全新请求，无历史上下文
- **继续对话 (Continue from Previous Turn)**: 基于之前对话的 KV Cache 继续生成

---

## 1. 推理入口

入口函数: `ParallelGpt<T>::forward()` (文件: `src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.cc`)

**输入张量：**
```
input_ids:        [batch_size, max_input_length]    // token IDs
input_lengths:    [batch_size]                       // 每个样本的实际长度
output_seq_len:   [batch_size]                       // 期望的输出长度
// 可选参数:
runtime_top_k, runtime_top_p, temperature, repetition_penalty, ...
```

**输出张量：**
```
output_ids:       [batch_size, beam_width, max_output_seq_len]
sequence_length:  [batch_size, beam_width]
cum_log_probs:    [batch_size, beam_width]  // 可选
```

---

## 2. 冷启动完整流程

### 阶段 0: 初始化

```
┌──────────────────────────────────────────────────────┐
│ 0a. Buffer 分配                                       │
│     allocateBuffer(batch_size, beam_width,            │
│                    max_seq_len, max_input_length)     │
│                                                       │
│ 0b. KV Cache 分配                                     │
│     key_cache_:   [num_layer, batch*beam,             │
│                    local_head_num, head_size/x,       │
│                    max_cache_seq_len, x]              │
│     value_cache_: [num_layer, batch*beam,             │
│                    local_head_num,                    │
│                    max_cache_seq_len, head_size]      │
│     (x = 16/sizeof(T), 保证16字节对齐访存)            │
│                                                       │
│ 0c. Beam Search 缓冲区 (beam_width > 1 时)           │
│     cache_indirections_[2]:                           │
│       [batch*beam, memory_len] × 2 (双缓冲)          │
└──────────────────────────────────────────────────────┘
```

**性能关键点：Buffer 预分配**
- 所有中间 tensor（`decoder_normed_input_`, `self_attn_output_`, `ffn_output_`, `inter_buf_` 等）一次性分配
- `is_free_buffer_after_forward_` 控制是否在推理后释放：设为 false 可复用 buffer（适合连续推理），设为 true 可省显存
- CUDA 11.2+ 使用 `cudaMallocAsync` 异步分配，与首次 kernel 启动重叠

### 阶段 1: Context Phase (Prefill)

这是冷启动中最耗时的阶段，需要一次性处理整个输入 prompt。

```
输入: input_ids [batch_size, max_input_length]
  │
  ▼
┌──────────────────────────────────────────────────────┐
│ Step 1.1: Tiling (Beam Width 展开)                    │
│   invokeTileGptInputs()                               │
│   [batch, seq_len] → [batch × beam_width, seq_len]   │
│   ⚡ 优化: 单 kernel 完成复制展开                      │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Step 1.2: Embedding + Position Encoding               │
│   invokeInputIdsEmbeddingLookupPosEncoding()          │
│                                                       │
│   token_emb = embedding_table[input_ids]              │
│   pos_emb   = position_table[0:seq_len]              │
│   output    = token_emb + pos_emb                    │
│                                                       │
│   ⚡ 融合: embedding lookup + position add 在一个     │
│           kernel 中完成                                │
│   ⚡ Soft Prompt: 如果有 prompt_learning_table，      │
│           直接拼接到 embedding 前面                    │
│                                                       │
│   输出: [batch×beam, max_input_length, hidden_units] │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Step 1.3: Build Attention Mask                        │
│   invokeBuildDecoderAttentionMask()                   │
│                                                       │
│   生成下三角因果掩码:                                  │
│   [1, 1, 0, 0]                                       │
│   [1, 1, 1, 0]                                       │
│   [1, 1, 1, 1]                                       │
│                                                       │
│   ⚡ 对于 padding: 利用 input_lengths 生成            │
│     combined mask = causal_mask & padding_mask        │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Step 1.4: Context Decoder Forward                     │
│   gpt_context_decoder_->forward()                     │
│                                                       │
│   对于每一层 l = 0 ... num_layers-1:                  │
│   ┌────────────────────────────────────────────┐     │
│   │ 1. Pre-LayerNorm                           │     │
│   │    invokeGeneralLayerNorm()                 │     │
│   │                                             │     │
│   │ 2. Self-Attention (全序列)                  │     │
│   │    ┌─ QKV GEMM: [b×s, h] × [h, 3h] → [b×s, 3h]│
│   │    ├─ Fused QKV Bias + Transpose            │     │
│   │    ├─ 存储 K, V 到 KV Cache                 │     │
│   │    ├─ Q × K^T → attention scores           │     │
│   │    ├─ Mask + Softmax                        │     │
│   │    ├─ Scores × V → attention output        │     │
│   │    └─ Output Projection GEMM               │     │
│   │                                             │     │
│   │ 3. Residual + Post-LayerNorm               │     │
│   │    invokeAddBiasResidualLayerNorm()         │     │
│   │    (融合: bias + residual + layernorm)       │     │
│   │                                             │     │
│   │ 4. FFN                                      │     │
│   │    ┌─ Up GEMM: [b×s, h] × [h, 4h] → [b×s, 4h] │
│   │    ├─ Activation (GELU/SiLU + bias 融合)   │     │
│   │    └─ Down GEMM: [b×s, 4h] × [4h, h]      │     │
│   │                                             │     │
│   │ 5. Residual Addition                        │     │
│   │    invokeAddBiasAttentionFfnResidual()      │     │
│   │    (融合: FFN bias + attention residual      │     │
│   │     + FFN residual)                         │     │
│   │                                             │     │
│   │ 6. [TP] AllReduce (如果 tensor_para > 1)    │     │
│   │    ftNcclAllReduceSum()                     │     │
│   │                                             │     │
│   │ 7. [PP] Send/Recv (如果 pipeline_para > 1)  │     │
│   │    首层 Recv，末层 Send                      │     │
│   └────────────────────────────────────────────┘     │
│                                                       │
│   输出: [batch×beam, max_input_length, hidden_units] │
│   副产物: KV Cache 已填充所有 input positions         │
└──────────┬───────────────────────────────────────────┘
```

**Context 阶段的关键优化：**

#### ⚡ 优化 1: GEMM 效率
Context 阶段 seq_len 大，GEMM 的 M 维度大，GPU 利用率高。这是 FT 的效率优势所在 — 批量处理整个序列。

#### ⚡ 优化 2: Fused MHA (TRT Kernel)
当满足条件时 (FP16 + seq_len ≤ 512 + 特定 SM)，使用 TensorRT 的融合 MHA kernel：
```
标准路径:  QKV_GEMM → Bias+Transpose → QK_GEMM → Softmax → AV_GEMM → Transpose
融合路径:  QKV_GEMM → FusedMHA(一个 kernel 完成所有注意力计算)
```
节省 4+ 次全局内存读写。

#### ⚡ 优化 3: Shared Context 去重
```cpp
invokeFindContextDups()  // 检测 batch 中重复的 input 序列
```
如果多个请求共享相同 prompt（如系统 prompt），只计算一次 context，结果复制给所有相同请求。由 `shared_contexts_ratio_` 控制触发阈值。

#### ⚡ 优化 4: Padding 消除
对于 batch 中长度不一的序列，FT 使用 `remove_padding` 模式：
- 去掉 padding tokens，拼接有效 tokens
- GEMM 只计算有效 tokens
- 通过 `padding_offset` 数组还原原始位置
- 节省 (1 - 有效率) 的计算量

### 阶段 2: Generation Phase (Decode)

从 context 输出的最后一个 token 的 hidden state 开始，逐 token 生成。

```
for step = max_input_length to max_output_seq_len:
  │
  ▼
┌──────────────────────────────────────────────────────┐
│ Step 2.1: Embedding Lookup (单个 token)               │
│   invokeEmbeddingLookupPosEncodingPadCount()          │
│   输入: output_ids_buf_[:, step-1] — 上一步生成的 ID │
│   输出: [batch×beam, 1, hidden_units]                │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Step 2.2: Decoder Forward (单 token)                  │
│   gpt_decoder_->forward()                             │
│                                                       │
│   对于每一层 l:                                       │
│   ┌────────────────────────────────────────────┐     │
│   │ 1. Pre-LayerNorm                           │     │
│   │                                             │     │
│   │ 2. Self-Attention (KV Cache 读取)           │     │
│   │    ★ 这里是 Context 和 Generation 的        │     │
│   │      核心区别！                              │     │
│   │                                             │     │
│   │    a) QKV GEMM: [b, h] × [h, 3h]          │     │
│   │       (只有1个token，M=batch_size)          │     │
│   │                                             │     │
│   │    b) 写入当前 K, V 到 Cache[step]          │     │
│   │                                             │     │
│   │    c) decoder_masked_multihead_attention:   │     │
│   │       - Q: 当前 token [b, num_head, d]     │     │
│   │       - K: Cache[0:step] 所有历史          │     │
│   │       - V: Cache[0:step] 所有历史          │     │
│   │       → 单个融合 kernel 完成:               │     │
│   │         Q×K^T → scale → softmax → ×V       │     │
│   │                                             │     │
│   │    d) Output Projection GEMM               │     │
│   │                                             │     │
│   │ 3. Residual + LayerNorm (融合)              │     │
│   │ 4. FFN (up GEMM → activation → down GEMM) │     │
│   │ 5. Residual (融合)                          │     │
│   │ 6. [TP] AllReduce                          │     │
│   └────────────────────────────────────────────┘     │
│                                                       │
│   输出: [batch×beam, hidden_units]                   │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Step 2.3: Logits 计算                                 │
│   1. Final LayerNorm                                  │
│   2. Logits GEMM:                                     │
│      [batch×beam, h] × [h, vocab_size_padded]        │
│                                                       │
│   ⚡ vocab_size 对齐到 8/16 的倍数，提高 GEMM 效率    │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Step 2.4: Dynamic Decode                              │
│   dynamic_decode_layer_->forward()                    │
│                                                       │
│   路由逻辑:                                           │
│   if (beam_width == 1) {                              │
│     → TopK/TopP Sampling                              │
│       1. Temperature scaling: logits /= temperature  │
│       2. Repetition penalty (惩罚已生成 token)        │
│       3. TopK 过滤 → TopP 过滤 → 采样               │
│   } else {                                            │
│     → Beam Search                                     │
│       1. Log-softmax                                  │
│       2. TopK per beam                                │
│       3. 更新 cum_log_probs                           │
│       4. 更新 cache_indirections (beam 重排)          │
│   }                                                   │
│                                                       │
│   输出: next_token_id [batch×beam]                   │
│   更新: finished_buf, sequence_lengths                │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Step 2.5: 终止检查                                    │
│   if all(finished_buf) || step >= max_output_seq_len: │
│       break                                           │
│   else:                                               │
│       next_token → output_ids_buf_[:, step]           │
│       continue loop                                   │
└──────────────────────────────────────────────────────┘
```

**Generation 阶段的关键优化：**

#### ⚡ 优化 5: Decoder Masked MHA (融合注意力 Kernel)

这是 generation 阶段最关键的 kernel，位于 `kernels/decoder_masked_multihead_attention/`。

```
传统实现 (4步):
  Step 1: QKV_GEMM             → 读写全局内存
  Step 2: Q × K^T (batch GEMM) → 读写全局内存
  Step 3: Softmax              → 读写全局内存
  Step 4: Score × V            → 读写全局内存

FT 融合实现 (2步):
  Step 1: QKV_GEMM                              → 读写全局内存
  Step 2: 融合 kernel (Q×K^T + softmax + ×V)   → 只写一次全局内存
          ├─ Q 从寄存器加载
          ├─ K 从 Cache 流式读取, 计算 QK
          ├─ 在线 Softmax (避免两遍扫描)
          ├─ V 从 Cache 流式读取, 加权求和
          └─ 最终输出写回全局内存
```

**核心技巧：**
- **按 head_size 模板特化**: 每种 head_size 有专用 kernel，寄存器分配在编译时确定
- **在线 Softmax**: 不需要先算最大值再算 exp，一遍完成
- **向量化访存**: 使用 `half2`/`float4` 减少内存事务数
- **Warp 级归约**: 利用 `__shfl_xor_sync` 做 warp 内归约，避免共享内存
- **Cache 友好**: KV Cache 的内存布局针对 generation 阶段的顺序读取优化

#### ⚡ 优化 6: GEMM 尺寸问题与应对

Generation 阶段 GEMM 的 M 维度 = batch_size（通常很小），导致 GPU 利用率低。

应对策略：
1. **增大 batch_size**: 连续 batching（Triton 后端支持动态 batch）
2. **cuBLAS 算法选择**: `cublasAlgoMap` 预 profile 小 M GEMM 的最优算法
3. **使用 cublasLt**: 对 FP16 优先使用 cublasLt 的 matmul API，支持更灵活的 tiling

#### ⚡ 优化 7: Beam Search Cache 管理

Beam search 中，不同 beam 可能选择不同 parent，KV Cache 需要重排：

```
step t-1: beam 0 → "hello"    beam 1 → "hi"
step t:   beam 0 选了 beam 1 的 parent  →  需要读 beam 1 的 KV Cache

FT 的做法: 不复制 Cache 数据！
而是维护 cache_indirections_ 数组:
  cache_indirections_[beam_id][time_step] = source_beam_id
注意力 kernel 根据 indirection 表动态路由读取

双缓冲: cache_indirections_[0] 和 [1] 交替使用，避免读写冲突
```

---

## 3. 继续对话 (从之前的 KV Cache 恢复)

### 3.1 与冷启动的关键区别

```
                    冷启动                        继续对话
                    ──────                        ────────
KV Cache 状态:     空，需要从 0 填充              已有之前 turn 的 cache
Context 输入:      完整 prompt                    仅新增 tokens (新 user query)
Cache 起始位置:    position 0                     position = prev_turn_length
Context 计算量:    O(full_prompt_len²)            O(new_query_len × total_len)
总延迟:           首 token 慢 (需要 prefill)      首 token 可能更快 (增量 prefill)
```

### 3.2 继续对话的流程

```
已有状态:
  KV Cache: [num_layer, batch, head, size, prev_seq_len, ...]
  已知: prev_context_length (之前对话的总 token 数)

新输入:
  new_input_ids: [batch, new_input_length]  (新的 user query)
```

#### Step 1: 增量 Context 处理

```
┌──────────────────────────────────────────────────────┐
│ Position Encoding 偏移:                               │
│   position_ids = [prev_context_length,                │
│                   prev_context_length + 1,            │
│                   ...,                                │
│                   prev_context_length + new_len - 1]  │
│                                                       │
│ Attention Mask 扩展:                                  │
│   新 tokens 可以 attend to:                           │
│   - 所有之前 turn 的 tokens (通过 KV Cache)           │
│   - 当前 turn 的因果掩码                              │
│                                                       │
│ Context Decoder:                                      │
│   - Q: 只有新 tokens [batch, new_len, hidden]        │
│   - K, V: Cache[0:prev_len] + 新 tokens              │
│   - 新的 K, V 追加到 Cache[prev_len:prev_len+new_len]│
│                                                       │
│ ⚡ 计算量: new_len × (prev_len + new_len)             │
│   vs 冷启动: (prev_len + new_len)²                   │
│   节省比例: ≈ prev_len / total_len                   │
└──────────────────────────────────────────────────────┘
```

#### Step 2: 继续 Generation

与冷启动完全相同，但 KV Cache 中已包含所有历史 tokens。

### 3.3 关键实现细节

#### KV Cache 地址计算

```cpp
// KV Cache 是预分配的固定大小 buffer
// max_cache_seq_len = max(total conversation length across all turns)
// 写入时通过 step offset 确定位置:

// Context phase 写入:
k_cache[layer][batch][head][:][prev_len : prev_len + new_input_len][:] = new_K
v_cache[layer][batch][head][prev_len : prev_len + new_input_len][:] = new_V

// Generation phase 写入:
k_cache[layer][batch][head][:][current_step][:] = current_K
v_cache[layer][batch][head][current_step][:] = current_V
```

#### memory_len 参数

```cpp
// memory_len 控制 KV Cache 可回看的最大长度
// 用于 sliding window attention 或限制内存:
if (memory_len > 0) {
    attention_window = min(current_step, memory_len);
    // 只读取最近 memory_len 个位置的 KV
}
```

### 3.4 Session 管理

FT 本身不管理会话状态（stateless 设计）。KV Cache 的持久化由上层负责：

```
方式 1: Triton Backend 维护 session → KV Cache 映射
         ├─ 收到请求时查找对应 session 的 KV Cache
         └─ 超时或手动释放

方式 2: 客户端传入 KV Cache
         ├─ 首次请求: server 返回 KV Cache (或 token)
         └─ 后续请求: 客户端带上之前的 cache 数据

方式 3: 固定 batch slot
         ├─ 每个 slot 绑定一个对话
         └─ slot 的 KV Cache 在 GPU 上驻留
```

---

## 4. 性能关键点深度分析

### 4.1 Memory Bandwidth 是瓶颈

Generation 阶段的核心矛盾：**每生成一个 token，需要读取所有层的全部参数 + 历史 KV Cache**。

```
读取量估算 (以 GPT-J 6B 为例):
  - 模型参数: 6B × 2 bytes (FP16) = 12 GB
  - KV Cache: 2 × num_layers × batch × seq_len × hidden × 2 bytes

A100 80GB HBM 带宽: 2 TB/s
  → 每个 token 最少需要 12GB / 2TB/s = 6ms (仅参数读取)
  → batch_size 越大，参数读取被摊薄，效率越高
```

### 4.2 Kernel Launch Overhead

Generation 阶段每个 token 的 kernel 数量：

```
每层:
  LayerNorm: 1 kernel
  QKV GEMM: 1 cuBLAS call
  Fused Attention: 1 kernel
  Output GEMM: 1 cuBLAS call
  Residual+LayerNorm: 1 kernel (融合)
  FFN Up GEMM: 1 cuBLAS call
  Activation: 1 kernel
  FFN Down GEMM: 1 cuBLAS call
  Residual: 1 kernel
  AllReduce: 1 NCCL call (如果 TP)
  ≈ 9 calls/layer

32 层: ≈ 288 kernel launches + Logits GEMM + Decode
每次 launch ~5μs → 总 launch overhead ≈ 1.5ms

占比: 1.5ms / 6ms ≈ 25% (显著！)
```

**FT 的应对：**
1. Kernel 融合减少 launch 次数 (bias+residual+layernorm 三合一)
2. 使用 CUDA Graph 可以捕获并回放 kernel 序列（外层框架实现）
3. Stream 异步执行，pipeline 重叠

### 4.3 量化的性能收益

```
INT8 Weight-Only (int8_mode=1):
  - 参数读取量减半: 12GB → 6GB
  - GEMM 使用 CUTLASS fpA_intB: FP16激活 × INT8权重
  - 反量化在 GEMM 内完成，无额外开销
  - Token latency ≈ 降低 40-50%

INT8 SmoothQuant (int8_mode=2):
  - 激活也量化为 INT8
  - 使用 cublasINT8MMWrapper 的 INT8 GEMM
  - 需要预先校准量化参数
  - 吞吐量提升 ≈ 1.5-2x

FP8 (Hopper GPU):
  - 8-bit 浮点，保持动态范围
  - Tensor Core 原生支持
  - 吞吐量 ≈ 2x vs FP16
```

### 4.4 Tensor Parallelism 的通信开销

```
每层需要 2 次 AllReduce:
  1. Attention output 后 (合并 head 并行的结果)
  2. FFN output 后 (合并列并行的结果)

AllReduce 耗时:
  NCCL AllReduce (NVLink):
    batch=1, hidden=4096, FP16: ≈ 10-20μs (小消息)
    batch=32, hidden=4096, FP16: ≈ 20-30μs

  自定义 AllReduce (custom_ar):
    对小消息可能更快 (避免 NCCL 初始化开销)

通信占比:
  32层 × 2次 × 20μs = 1.28ms
  vs 计算 6ms → 约 20% 开销

  ⚡ FT 用 ftNcclGroupStart/End 批量提交 NCCL 操作
```

### 4.5 Pipeline Parallelism 的 Bubble

```
Pipeline 4 阶段示例:

时间 →
GPU 0: [Stage 0] ─────── idle ────── [Stage 0] ───
GPU 1:  idle  [Stage 1] ──── idle ── [Stage 1] ───
GPU 2:  idle ─── [Stage 2] ── idle ─ [Stage 2] ───
GPU 3:  idle ────── [Stage 3] ────── [Stage 3] ───

Bubble 比例 = (PP_size - 1) / PP_size
PP=4: 75% bubble (灾难性)

应对: Microbatching
  将 batch 拆分为多个 micro-batch，流水执行
  local_batch_size = batch_size / pipeline_para.world_size
  Bubble 比例 = (PP_size - 1) / (PP_size + num_microbatches - 1)
```

---

## 5. 完整时序图

### 5.1 冷启动时序

```
Time ──────────────────────────────────────────────────→

[Buffer Alloc]
         [Embedding + PosEnc (fused kernel)]
                  [Build Attention Mask]
                           [Context Decoder: Layer 0...N]
                           ├─ [LayerNorm][QKV GEMM][Fused Attn][O GEMM]
                           ├─ [Residual+LN (fused)][Up GEMM][Act][Down GEMM]
                           └─ [Residual (fused)][AllReduce]
                                                            [Init Decode State]
                                                                     [Gen Step 0]
                                                                     ├─ [Embed]
                                                                     ├─ [Decoder Layer 0..N]
                                                                     ├─ [Logits GEMM]
                                                                     └─ [Sample/BeamSearch]
                                                                              [Gen Step 1]
                                                                              ...
|←──── 首 Token 延迟 (TTFT) ────→|←── Token 间延迟 ──→|
```

### 5.2 继续对话时序

```
Time ──────────────────────────────────────────────────→

[Incremental Context: new_query only]
├─ Layer 0..N: Q=new_tokens, KV=cache+new
├─ 计算量 ≈ new_len × total_len (远小于冷启动)
                                    [Gen Step 0]
                                    [Gen Step 1]
                                    ...
|←── TTFT (更短) ──→|←── 相同 ───→|
```

---

## 6. 冷启动 vs 继续对话对比总结

| 维度 | 冷启动 | 继续对话 |
|------|--------|----------|
| **KV Cache** | 空，需完整构建 | 已有历史，仅追加 |
| **Context 计算量** | O(prompt_len²) | O(new_len × total_len) |
| **首 Token 延迟** | 高 (与 prompt 长度平方相关) | 低 (仅处理新增部分) |
| **GPU 显存** | 按最大长度分配 | 已分配，可能需扩展 |
| **Position Encoding** | 从 0 开始 | 从 prev_len 偏移 |
| **Attention Mask** | 标准下三角 | 扩展掩码 (新→全部) |
| **Generation 性能** | 相同 | 相同 (取决于总 cache 长度) |
| **Buffer 分配** | 需要 allocateBuffer | 如果 shape 不变可复用 (`reMalloc` REUSE) |
| **GEMM 算法** | 可能 cache miss，首次较慢 | 已在 algo_map_ 中缓存 |
| **总结** | 适合首次交互 | 适合多轮对话，显著降低 TTFT |

---

## 7. 性能调优 Checklist

1. **增大 batch size**: 最直接的吞吐量提升手段，摊薄参数读取
2. **启用量化**: INT8 weight-only 几乎无精度损失，延迟降低 40%+
3. **选择正确的并行策略**: 小模型用 TP，大模型 TP + PP
4. **预热 GEMM**: 运行 `gemm_config.in` profiling，找到最优算法
5. **启用 `remove_padding`**: batch 中长度差异大时效果显著
6. **调整 `memory_len`**: 限制 KV Cache 长度可节省显存和注意力计算
7. **使用 FP16/BF16**: 避免 FP32 (2x 带宽浪费)
8. **复用 Buffer**: `is_free_buffer_after_forward_ = false` 用于连续推理
9. **Shared Context**: 相同 system prompt 的请求合并处理
10. **Triton 动态 Batching**: 利用 Triton 的 dynamic batching 最大化 GPU 利用率
