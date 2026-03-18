# FasterTransformer 系统总体设计文档

## 1. 项目概述

FasterTransformer (FT) 是 NVIDIA 开发的高性能 Transformer 推理加速库，核心目标是通过 CUDA kernel 融合、量化、多 GPU 并行等手段，实现 Transformer 类模型在 GPU 上的极致推理性能。

**核心设计理念：**
- 面向推理而非训练，所有数据结构和计算路径围绕推理场景优化
- 模板化设计 (`<T>`)，支持 FP32/FP16/BF16/INT8/FP8 多精度
- 计算与通信重叠，减少 GPU 空闲时间
- 最大化 kernel 融合，减少 HBM 读写次数

---

## 2. 目录结构与模块划分

```
FasterTransformer/
├── 3rdparty/                  # 第三方依赖 (CUTLASS, TRT fused MHA)
├── src/fastertransformer/
│   ├── kernels/               # CUDA kernel 实现 (核心计算层)
│   │   ├── decoder_masked_multihead_attention/  # 按 head_size 特化的注意力 kernel
│   │   ├── cutlass_kernels/   # CUTLASS GEMM (MoE, 混合精度, weight-only quant)
│   │   ├── *_int8_kernels.cu  # INT8 量化 kernel
│   │   └── *_fp8_kernels.cu   # FP8 量化 kernel
│   │
│   ├── layers/                # 层级抽象 (中间抽象层)
│   │   ├── BaseLayer.h        # 所有 layer 的基类
│   │   ├── attention_layers/  # 注意力层 (Fused/Unfused/Context/Decoder)
│   │   ├── attention_layers_int8/  # INT8 注意力
│   │   ├── attention_layers_fp8/   # FP8 注意力
│   │   ├── FfnLayer.h         # FFN 层 (Gelu/Relu/Silu 变体)
│   │   ├── sampling_layers/   # TopK/TopP 采样
│   │   ├── beam_search_layers/ # Beam Search
│   │   ├── DynamicDecodeLayer.h  # 统一解码层 (调度采样/beam search)
│   │   └── TensorParallel*.h  # 张量并行封装层
│   │
│   ├── models/                # 完整模型实现 (最高层抽象)
│   │   ├── multi_gpu_gpt/     # 多 GPU GPT (核心模型)
│   │   ├── gptj/              # GPT-J
│   │   ├── gptneox/           # GPT-NeoX
│   │   ├── bert/              # BERT 编码器
│   │   ├── t5/                # T5 编解码
│   │   ├── vit/, swin/        # 视觉 Transformer
│   │   └── gpt_fp8/           # FP8 GPT
│   │
│   ├── utils/                 # 基础设施
│   │   ├── allocator.h        # 内存分配器 (CUDA/TF/PyTorch)
│   │   ├── cublasMMWrapper.h  # cuBLAS 矩阵乘封装 + 算法选择
│   │   ├── cublasAlgoMap.h    # GEMM 算法缓存与自动调优
│   │   ├── Tensor.h           # 张量抽象 (CPU/GPU, 数据类型)
│   │   ├── nccl_utils.h       # NCCL 通信封装
│   │   └── custom_ar_comm.h   # 自定义 AllReduce 通信
│   │
│   ├── triton_backend/        # Triton 推理服务后端
│   ├── th_op/                 # PyTorch 自定义算子
│   ├── tf_op/                 # TensorFlow 自定义算子
│   └── tensorrt_plugin/       # TensorRT 插件
│
├── examples/                  # C++/PyTorch/TensorFlow/TensorRT 使用示例
├── benchmarks/                # 性能基准测试
└── docs/                      # 各模型文档
```

---

## 3. 核心抽象层次

FT 的代码组织体现了三层抽象：

```
┌─────────────────────────────────────────────────────┐
│  Model 层: ParallelGpt, GptJ, Bert, T5...          │  ← 完整模型，管理推理循环
├─────────────────────────────────────────────────────┤
│  Layer 层: Attention, FFN, DynamicDecode...         │  ← 可组合的层级构建块
├─────────────────────────────────────────────────────┤
│  Kernel 层: CUDA kernels, cuBLAS GEMM, CUTLASS     │  ← 底层计算原语
└─────────────────────────────────────────────────────┘
```

### 3.1 Kernel 层

位于 `src/fastertransformer/kernels/`，是所有计算的底层实现。

**关键设计决策：**
- **按 head_size 模板特化**: `decoder_masked_multihead_attention/` 目录下对每种 head_size (32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256) 编译专用 kernel，编译时确定寄存器和共享内存分配
- **向量化类型**: FP16 使用 `half2` 类型，每个线程处理两个元素，带宽翻倍
- **融合模式**: 尽可能在一个 kernel 内完成 bias + activation + quantization + residual

**Kernel 分类：**

| 类别 | 典型 Kernel | 功能 |
|------|-------------|------|
| 注意力 | `decoder_masked_multihead_attention` | 自回归注意力 (含 KV cache 读写) |
| 注意力 | `unfused_attention_kernels` | 非融合多头注意力 (支持任意 head_size) |
| LayerNorm | `generalAddBiasResidualLayerNormOpt` | 融合 bias + residual + LayerNorm |
| 激活 | `invokeGenericActivation` | GELU/ReLU/SiLU + bias + gating + 量化 |
| 解码 | `sampling_topk/topp_kernels` | TopK/TopP 采样 |
| 解码 | `beam_search_topk_kernels` | Beam Search TopK |
| 量化 | `calibrate_quantize_weight_kernels` | 权重量化与校准 |

### 3.2 Layer 层

位于 `src/fastertransformer/layers/`，基于 `BaseLayer` 抽象：

```cpp
class BaseLayer {
protected:
    cudaStream_t stream_;
    cublasMMWrapper* cublas_wrapper_;
    IAllocator* allocator_;
    bool is_free_buffer_after_forward_;  // 控制 buffer 生命周期

    virtual void allocateBuffer() = 0;   // 预分配计算 buffer
    virtual void freeBuffer() = 0;       // 释放 buffer
};
```

**核心 Layer 类型：**

1. **注意力层** — 根据场景选择不同实现：
   - `FusedAttentionLayer`: TRT fused MHA (FP16, seq_len ≤ 512, 特定 SM)
   - `UnfusedAttentionLayer`: 通用多步注意力
   - `GptContextAttentionLayer`: Context 阶段的全序列注意力
   - `DecoderSelfAttentionLayer`: 生成阶段单 token 注意力 (读 KV cache)

2. **FFN 层** — 按激活函数特化：
   - `GeluFfnLayer`, `ReluFfnLayer`, `SiluFfnLayer`
   - Gated 变体: `GeGLU`, `ReGLU`, `SiGLU` (双 GEMM + 门控)

3. **解码层** — 统一调度：
   - `DynamicDecodeLayer`: 运行时根据 `beam_width` 和参数路由到采样或 beam search
   - 支持 temperature, repetition_penalty, length_penalty 等参数

4. **张量并行封装层**：
   - `TensorParallelDecoderSelfAttentionLayer`: 注意力的 TP 封装
   - `TensorParallelGeluFfnLayer`: FFN 的 TP 封装
   - 在计算后执行 AllReduce 归约

### 3.3 Model 层

位于 `src/fastertransformer/models/`，是最高级别抽象。以 `ParallelGpt` 为例：

```
ParallelGpt (模型入口)
  ├── ParallelGptContextDecoder (Context 阶段)
  │   └── N 层 { TensorParallelGptContextAttentionLayer + TensorParallelFfnLayer }
  ├── ParallelGptDecoder (Generation 阶段)
  │   └── N 层 { TensorParallelDecoderSelfAttentionLayer + TensorParallelFfnLayer }
  └── DynamicDecodeLayer (采样/beam search)
```

**Model 的职责：**
- 管理完整推理循环 (context + generation loop)
- 管理 KV cache 的分配和生命周期
- Embedding lookup + Position encoding
- 输出 logits 计算
- 协调 pipeline parallelism 的跨阶段通信

---

## 4. 支持的模型

| 类别 | 模型 | 精度支持 | 多 GPU |
|------|------|----------|--------|
| **自回归生成** | GPT / OPT | FP16, FP8 | TP + PP |
| | GPT-J | FP16 | TP + PP |
| | GPT-NeoX | FP16 | TP + PP |
| | GPT-MoE | FP16 | TP + PP |
| **编码器** | BERT | FP16, INT8, FP8 | TP + PP |
| | DeBERTa | FP16 | - |
| | Longformer | FP16 | - |
| | XLNet | FP16 | - |
| **编解码** | T5 / UL2 | FP16 | TP + PP |
| | BART / mBART | FP16 | TP + PP |
| **视觉** | ViT | FP16, INT8 | - |
| | Swin Transformer | FP16, INT8 | - |

---

## 5. 内存管理

### 5.1 IAllocator 接口

```cpp
class IAllocator {
    virtual void* malloc(size_t size, bool is_set_zero, bool is_host) = 0;
    virtual void  free(void** ptr, bool is_host) = 0;
    virtual void* reMalloc(T* ptr, size_t size, bool is_set_zero) = 0; // 智能重分配
};
```

**三种后端实现：**
- `Allocator<CUDA>`: 原生 CUDA 内存，支持 CUDA 11.2+ 异步内存池
- `Allocator<TF>`: 通过 TensorFlow OpKernelContext 分配
- `Allocator<TH>`: 通过 PyTorch torch::Tensor 分配

### 5.2 智能重分配策略

`reMalloc()` 通过比较已有 buffer 大小与请求大小，选择策略：
- **REUSE**: 已有 buffer 足够大，直接复用（可能 memset 清零）
- **INCREASE**: buffer 不够大，释放旧的并分配更大的
- **DECREASE**: CUDA 11.2+ 可释放多余部分回内存池

**关键优化: 32 字节对齐** — 所有分配对齐到 32 字节，保证 GPU 内存访问合并。

### 5.3 CUDA 异步内存池 (CUDA 11.2+)

```cpp
cudaMemPool_t mempool;
cudaDeviceGetDefaultMemPool(&mempool, device_id);
// 防止内存池收缩
cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, UINT64_MAX);
// 异步分配，与 kernel 执行重叠
cudaMallocAsync(&ptr, size, stream_);
```

### 5.4 Layer 的 Buffer 管理模式

每个 Layer 遵循统一的 buffer 管理模式：
1. `allocateBuffer(batch_size, seq_len, ...)`: 预分配所有中间 buffer
2. `forward()`: 使用预分配的 buffer 执行计算
3. `freeBuffer()`: 可选释放（由 `is_free_buffer_after_forward_` 控制）

**权衡**: 保留 buffer 可避免重复分配开销（适合相同 shape 的连续推理），释放 buffer 可降低峰值内存（适合间歇性推理）。

---

## 6. 多精度支持

### 6.1 精度类型

```cpp
enum DataType {
    TYPE_FP32,      // 默认精度
    TYPE_FP16,      // Volta+ GPU, 主要优化目标
    TYPE_BF16,      // CUDA 11.0+, 范围更大
    TYPE_FP8_E4M3,  // Hopper+ GPU (H100), CUDA 11.8+
    TYPE_INT8,      // Turing+ GPU 量化
    TYPE_INT32, TYPE_UINT8, ...
};
```

### 6.2 模板化精度处理

FT 通过 C++ 模板在编译时确定精度：
- `class ParallelGpt<T>`: T 为 float / half / __nv_bfloat16
- INT8 和 FP8 有独立的 Layer 实现（如 `FfnLayerINT8<T>`, `FfnFP8Layer<T1, T2>`）

### 6.3 量化策略

| 模式 | `int8_mode` 值 | 描述 |
|------|----------------|------|
| 无量化 | 0 | FP16/BF16 全精度推理 |
| Weight-Only | 1 | 权重 INT8，激活 FP16，GEMM 使用 CUTLASS fpA_intB |
| SmoothQuant | 2 | 权重和激活都 INT8，使用 cublasINT8MMWrapper |
| FP8 | - | 权重和激活 FP8，使用 cublasFP8MMWrapper |

### 6.4 cuBLAS GEMM 算法管理

`cublasAlgoMap` 提供两层缓存：
1. **文件缓存**: `gemm_config.in` (FP16), `igemm_config.in` (INT8) — 存储预先 profile 的最优算法
2. **内存缓存**: `algo_map_` — 运行时按 (batch, M, N, K, dtype) 查找

```cpp
struct cublasAlgoConfig_t { int batch_count, m, n, k; CublasDataType data_type; };
struct cublasLtMatmulAlgo_info {
    int algoId, tile, splitK_val, stages, swizzle;
    float exec_time;  // profile 时间
};
```

---

## 7. 多 GPU 并行

### 7.1 两种并行模式

```
张量并行 (TP, Tensor Parallelism):
  - 水平切分: 注意力的 head 维度 / FFN 的中间维度
  - 每层末尾做 AllReduce
  - 降低每 GPU 计算量和显存

流水线并行 (PP, Pipeline Parallelism):
  - 垂直切分: 将层均匀分配到不同 GPU
  - 跨阶段做 Send/Recv
  - 降低单 GPU 显存，增加通信延迟

组合使用: TP × PP = total_gpus
  rank 计算: tensor_para_rank = rank % tp_size
             pipeline_para_rank = rank / tp_size
```

### 7.2 NCCL 通信

```cpp
struct NcclParam {
    int rank_, world_size_;
    ncclComm_t nccl_comm_;
};
```

**封装的通信原语：**
- `ftNcclAllReduceSum()`: TP 层后的梯度归约
- `ftNcclSend() / ftNcclRecv()`: PP 阶段间传递隐状态
- `ftNcclAllGather()`: 收集分布式输出
- `ftNcclGroupStart/End()`: 分组执行减少通信开销

### 7.3 自定义 AllReduce

`custom_ar_comm.h` 提供自定义 AllReduce 接口，在小规模张量通信时可能比 NCCL 更快（避免 NCCL 启动开销）。

---

## 8. Kernel 融合策略

这是 FT 性能的核心来源之一。主要融合模式：

### 8.1 注意力融合

| 场景 | 实现 | 融合范围 |
|------|------|----------|
| Context 阶段 + FP16 + seq≤512 | TRT fused MHA (`MHARunner`) | QKV proj → Softmax → Output proj 全融合 |
| Context 阶段 + 通用 | `GptContextAttentionLayer` | QKV bias+transpose 融合; QK GEMM; Softmax; AV GEMM |
| Generation 阶段 | `decoder_masked_multihead_attention` | Q·K^T + Softmax + ·V 在单 kernel 内 (读 KV cache) |

### 8.2 LayerNorm + Residual 融合

```
标准流程:     residual → bias_add → layernorm  (3次 HBM 读写)
融合 kernel:  generalAddBiasResidualLayerNormOpt   (1次 HBM 读写)
```

### 8.3 激活函数融合

```
标准流程:     GEMM → bias_add → activation → quantize  (4次内存操作)
融合 kernel:  invokeGenericActivation                    (1次内存操作)
              支持: bias + GELU/ReLU/SiLU + gating + IA3 scaling + INT8量化
```

### 8.4 QKV 融合

```
标准:   Q = W_q * x + b_q;  K = W_k * x + b_k;  V = W_v * x + b_v  (3次GEMM)
融合:   [Q;K;V] = W_qkv * x + b_qkv                                  (1次GEMM)
        + invokeAddFusedQKVBiasTranspose (bias + transpose 融合)
```

---

## 9. 服务集成

### 9.1 多框架支持

```
┌────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐
│ Triton 后端 │  │ PyTorch 算子  │  │ TF 自定义Op │  │ TensorRT 插件│
│ triton_     │  │ th_op/       │  │ tf_op/      │  │ tensorrt_    │
│ backend/    │  │              │  │             │  │ plugin/      │
└─────┬──────┘  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘
      │                │                  │                 │
      └────────────────┴──────────────────┴─────────────────┘
                              │
                   ┌──────────┴──────────┐
                   │  FT 核心库           │
                   │  (models + layers    │
                   │   + kernels)         │
                   └─────────────────────┘
```

### 9.2 Triton Backend

- 每个模型有独立的 Triton 后端实现
- 支持动态 batch、流式输出（callback）
- 自动处理 GPU 多设备 NCCL 初始化
- 数据类型自动转换 (TRITONSERVER_DataType ↔ FT DataType)

---

## 10. 构建系统

### 10.1 关键 CMake 选项

| 选项 | 功能 |
|------|------|
| `BUILD_MULTI_GPU` | 启用 MPI + NCCL 多 GPU |
| `BUILD_PYT` | 构建 PyTorch 扩展 |
| `BUILD_TF / BUILD_TF2` | 构建 TensorFlow Op |
| `BUILD_TRT` | 构建 TensorRT 插件 |
| `ENABLE_FP8` | 启用 FP8 (CUDA 11.8+) |
| `ENABLE_BF16` | 启用 BF16 (CUDA 11.0+) |
| `SPARSITY_SUPPORT` | 启用 Ampere 稀疏 (cuSPARSELt) |
| `BUILD_FAST_MATH` | `--use_fast_math` 编译标记 |
| `BUILD_CUTLASS_MOE` | CUTLASS MoE GEMM |
| `BUILD_CUTLASS_MIXED_GEMM` | CUTLASS 混合精度 GEMM |

### 10.2 GPU 架构支持

SM 52, 60, 61, 70 (Volta), 75 (Turing), 80 (Ampere), 86, 89 (Ada), 90 (Hopper)

C++ 标准: C++17

输出: 单一共享库 `transformer-shared`

---

## 11. 权重管理

每个模型有对应的 Weight 类，结构如下：

```cpp
struct ParallelGptWeight<T> {
    std::vector<ParallelGptDecoderLayerWeight<T>> decoder_layer_weights;
    const T* pre_decoder_embedding_table;       // 词嵌入表
    const T* position_encoding_table;           // 位置编码表
    LayerNormWeight<T> post_decoder_layernorm;  // 最终 LayerNorm
    DenseWeight<T> post_decoder_embedding;      // LM head
    std::vector<std::pair<const T*, int>> prompt_learning_table;  // soft prompt
};

struct ParallelGptDecoderLayerWeight<T> {
    LayerNormWeight<T> pre_layernorm_weights;
    AttentionWeight<T> self_attention_weights;   // Q/K/V/O 权重 + bias
    LayerNormWeight<T> post_attn_layernorm_weights;
    FfnWeight<T> ffn_weights;                    // FFN up/down 权重 + bias
};
```

**权重加载流程:**
1. 构造函数分配 GPU 内存 (`mallocWeights()`)
2. 设置指针 (`setWeightPtr()`)
3. 从文件加载 (`loadModel(dir_path)`)
4. 对于 TP，构造时自动切片

---

## 12. 整体数据流

```
用户请求 (input_ids, 采样参数)
    │
    ▼
┌───────────────────────────────────────────────────┐
│ Model.forward()                                   │
│                                                   │
│  1. Embedding + Position Encoding                 │
│  2. Context Decoder (全序列并行处理)              │
│     └→ 填充 KV Cache                              │
│  3. Generation Loop:                              │
│     ├─ Embedding (单 token)                       │
│     ├─ Decoder (读 KV Cache, 追加新 KV)          │
│     ├─ LayerNorm → Logits GEMM                   │
│     ├─ DynamicDecode (采样/beam search)           │
│     └─ 检查 end_id / max_length → 继续或结束     │
│  4. 输出 output_ids, sequence_lengths             │
└───────────────────────────────────────────────────┘
```

---

## 13. 设计特点总结

1. **计算密集优先**: 通过 kernel 融合将 memory-bound 操作转为 compute-bound
2. **编译时特化**: 模板 + head_size 特化避免运行时开销
3. **内存效率**: 预分配 + 复用 buffer，异步内存池
4. **灵活并行**: TP + PP 可组合，适配 1~数百 GPU
5. **多框架兼容**: 核心库与框架无关，通过薄 wrapper 集成
6. **渐进量化**: 从 FP32 → FP16 → INT8 → FP8 多档位，精度-性能可选
