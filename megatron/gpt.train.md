# Megatron-LM GPT 训练深度解析

> 以 `pretrain_gpt.py` 为入口，深入剖析 Megatron-LM 框架训练 GPT 模型的完整过程，涵盖各种并行策略、切分机制、Checkpoint 保存与故障恢复。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [训练入口：pretrain_gpt.py](#2-训练入口pretrain_gptpy)
3. [训练主循环：training.py](#3-训练主循环trainingpy)
4. [GPT 模型架构](#4-gpt-模型架构)
5. [并行策略深入解析](#5-并行策略深入解析)
   - 5.1 [张量并行 (Tensor Parallelism, TP)](#51-张量并行-tensor-parallelism-tp)
   - 5.2 [流水线并行 (Pipeline Parallelism, PP)](#52-流水线并行-pipeline-parallelism-pp)
   - 5.3 [数据并行 (Data Parallelism, DP)](#53-数据并行-data-parallelism-dp)
   - 5.4 [序列并行 (Sequence Parallelism, SP)](#54-序列并行-sequence-parallelism-sp)
   - 5.5 [上下文并行 (Context Parallelism, CP)](#55-上下文并行-context-parallelism-cp)
   - 5.6 [专家并行 (Expert Parallelism, EP)](#56-专家并行-expert-parallelism-ep)
   - 5.7 [虚拟流水线并行 (Virtual Pipeline Parallelism, VPP)](#57-虚拟流水线并行-virtual-pipeline-parallelism-vpp)
6. [完整示例：128 GPU 训练 GPT-175B](#6-完整示例128-gpu-训练-gpt-175b)
7. [Checkpoint 保存机制](#7-checkpoint-保存机制)
8. [故障恢复机制](#8-故障恢复机制)
9. [关键通信模式总结](#9-关键通信模式总结)

---

## 1. 整体架构概览

Megatron-LM 的训练流程可以概括为以下调用链：

```
pretrain_gpt.py::main()
  ├── set_startup_timestamps()                 # 记录启动时间戳
  ├── inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)  # 包装进程内重启
  └── pretrain(                                # 核心训练入口
        train_valid_test_datasets_provider,     # 数据集构建器
        model_provider,                         # 模型构建器
        ModelType.encoder_or_decoder,           # 模型类型
        forward_step,                           # 前向计算步骤
      )
        ├── initialize_megatron()              # 初始化分布式环境、参数解析
        │     ├── _initialize_distributed()    # 初始化 torch.distributed
        │     └── _set_random_seed()           # 设置随机种子
        ├── _init_parallel_state()             # 创建 TP/PP/DP/CP/EP 进程组
        ├── get_model()                        # 构建模型（含并行切分）
        ├── get_optimizer()                    # 构建优化器
        ├── load_checkpoint()                  # 加载检查点（如果有）
        └── train()                            # 训练主循环
              └── while iteration < train_iters:
                    ├── train_step()           # 单步训练
                    │     ├── forward_backward_func()  # 前向+反向
                    │     └── optimizer.step()         # 参数更新
                    ├── training_log()         # 日志记录
                    ├── evaluate()             # 定期验证
                    └── save_checkpoint()      # 定期保存
```

---

## 2. 训练入口：pretrain_gpt.py

### 2.1 主入口

```python
# pretrain_gpt.py, line 399-421
if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # 允许进程内重启（故障恢复）
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,    # 数据集提供函数
        partial(model_provider, gpt_builder),  # 模型构建函数
        ModelType.encoder_or_decoder,          # GPT 是 decoder-only
        forward_step,                          # 前向步骤函数
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        get_embedding_ranks=get_embedding_ranks,
    )
```

`pretrain()` 接收四个核心回调函数，将 **数据**、**模型**、**前向计算** 的逻辑与训练框架解耦。

### 2.2 数据获取：get_batch()

```python
# pretrain_gpt.py, line 65-163
def get_batch(data_iterator, vp_stage=None):
    # 1. 非首尾流水线阶段 → 不需要数据，返回 None
    if not is_first_or_last_pipeline_stage(vp_stage) and not is_packed_sequence:
        return None, None, None, None, None, None

    # 2. 根据 TP rank 获取对应的 batch（只有 TP rank 0 从 iterator 取数据，然后广播）
    batch = get_batch_on_this_tp_rank(data_iterator)

    # 3. 根据 CP（上下文并行）对序列进行切分
    if cu_seqlens is None and local_cp_size is None:
        batch = get_batch_on_this_cp_rank(batch)     # 标准 CP 切分
    elif local_cp_size is None:
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(...)  # THD 格式
    else:
        batch, packed_seq_params = get_batch_on_this_hybrid_cp_rank(...)  # 混合 CP

    return (tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params)
```

**关键点：**
- 只有 **TP rank 0** 从 data_iterator 读取数据，然后通过 `broadcast` 分发给同一 TP 组的其他 rank
- 只有 **PP 首尾阶段** 需要完整的 batch 数据（中间阶段通过 P2P 通信获取激活值）
- **CP 切分**：将序列维度按 `cp_size` 分割，每个 CP rank 处理序列的一个片段

### 2.3 前向计算：forward_step()

```python
# pretrain_gpt.py, line 232-268
def forward_step(data_iterator, model, return_schedule_plan=False):
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator)

    output_tensor = model(
        tokens, position_ids, attention_mask,
        labels=labels, loss_mask=loss_mask,
        packed_seq_params=packed_seq_params
    )

    return output_tensor, partial(loss_func, loss_mask, model=model)
```

### 2.4 损失函数：loss_func()

```python
# pretrain_gpt.py, line 170-229
def loss_func(loss_mask, output_tensor, model=None):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)       # 掩码加权求和
    num_tokens = loss_mask.sum()

    # NaN/Inf 检测 → 触发重跑机制
    rerun_state_machine.validate_result(result=loss, rejection_func=torch.isnan, ...)
    # Spiky Loss 检测 → 触发重跑机制
    rerun_state_machine.validate_result(result=loss, rejection_func=is_unexpectedly_large, ...)

    return loss, num_tokens, {'lm loss': torch.cat([loss, num_tokens])}
```

---

## 3. 训练主循环：training.py

### 3.1 pretrain() 函数：整体编排

```python
# megatron/training/training.py, line 400+
def pretrain(train_valid_test_datasets_provider, model_provider, model_type, forward_step_func, ...):
    # Phase 1: 初始化
    initialize_megatron(...)           # 分布式环境、参数解析
    _init_parallel_state(...)          # 创建并行进程组

    # Phase 2: 构建模型和优化器
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type
    )
    # 内部调用链：
    #   get_model()           → 创建模型实例（含 TP/PP 切分）
    #   get_optimizer()       → 创建分布式优化器（含 DP shard）
    #   load_checkpoint()     → 恢复训练状态

    # Phase 3: 构建数据集
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(...)
    train_data_iterator = build_train_valid_test_data_iterators(...)

    # Phase 4: 训练
    iteration, num_flops = train(
        forward_step_func, model, optimizer,
        opt_param_scheduler, train_data_iterator, ...
    )

    # Phase 5: 最终评估
    evaluate_and_print_results(...)
```

### 3.2 train_step()：单步训练

```python
# megatron/training/training.py, line 1737-1904
def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
    # 1. 清零梯度
    for m in model:
        m.zero_grad_buffer()
    optimizer.zero_grad()

    # 2. 获取前向-反向函数（根据 PP 配置）
    forward_backward_func = get_forward_backward_func()
    # 返回值：
    #   PP=1  → forward_backward_no_pipelining
    #   PP>1  → forward_backward_pipelining_without_interleaving (1F1B)
    #   PP>1 + VPP → forward_backward_pipelining_with_interleaving

    # 3. 执行前向-反向传播（含微批次调度）
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        dtype=config.pipeline_dtype,
        forward_only=False,
    )

    # 4. 参数更新
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

    # 5. 学习率调度
    if update_successful:
        opt_param_scheduler.step(increment=num_microbatches)

    return losses_reduced, skipped_iter, grad_norm, num_zeros_in_grad
```

### 3.3 train()：主循环

```python
# megatron/training/training.py, line 2562-3218
def train(...):
    while iteration < args.train_iters:
        # 1. 更新微批次数（支持 batch size ramp-up）
        update_num_microbatches(...)

        # 2. 执行单步训练
        losses_reduced, skipped_iter, grad_norm, ... = train_step(...)

        # 3. 日志记录（loss、learning rate、throughput、内存）
        training_log(...)

        # 4. 定期评估
        if iteration % args.eval_interval == 0:
            evaluate_and_print_results(...)

        # 5. 定期保存检查点
        if iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, ...)

        # 6. 非持久化检查点（高频保存用于快速恢复）
        if iteration % args.non_persistent_save_interval == 0:
            save_checkpoint(..., non_persistent_ckpt=True)

        iteration += 1
```

---

## 4. GPT 模型架构

### 4.1 模型结构

```python
# megatron/core/models/gpt/gpt_model.py
class GPTModel(LanguageModule):
    def __init__(self, config, transformer_layer_spec, vocab_size, max_sequence_length,
                 pre_process=True, post_process=True, ...):

        # Embedding 层（仅在 PP 首阶段 / MTP 阶段）
        if pre_process or mtp_process:
            self.embedding = LanguageModelEmbedding(config, vocab_size, ...)

        # RoPE 旋转位置编码
        if config.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(config)

        # Transformer 解码器（核心，按 PP 划分层数）
        self.decoder = TransformerBlock(config, transformer_layer_spec, ...)
        # 内部包含 config.num_layers 个 TransformerLayer
        # 每个 TransformerLayer = Self-Attention + MLP + LayerNorm

        # 输出层（仅在 PP 末阶段）
        if post_process:
            self.output_layer = ColumnParallelLinear(
                config.hidden_size, vocab_size, ...)
```

### 4.2 单个 TransformerLayer

```
输入: [seq_len, batch, hidden_size]
  │
  ├── LayerNorm
  ├── Self-Attention (含 TP 切分)
  │     ├── QKV Projection: ColumnParallelLinear [hidden→3*hidden/tp]
  │     ├── Attention Core: 每个 TP rank 处理 num_heads/tp 个头
  │     └── Output Projection: RowParallelLinear [hidden/tp→hidden]
  │              └── AllReduce / ReduceScatter (TP 通信)
  ├── Residual Connection
  │
  ├── LayerNorm
  ├── MLP (含 TP 切分)
  │     ├── FC1 (Up): ColumnParallelLinear [hidden→ffn_hidden/tp]
  │     ├── Activation (GeLU/SwiGLU)
  │     └── FC2 (Down): RowParallelLinear [ffn_hidden/tp→hidden]
  │              └── AllReduce / ReduceScatter (TP 通信)
  └── Residual Connection

输出: [seq_len, batch, hidden_size]
```

---

## 5. 并行策略深入解析

### 5.1 张量并行 (Tensor Parallelism, TP)

**核心思想：** 将单个 Transformer 层内的矩阵运算切分到多个 GPU 上并行执行。

**实现文件：**
- `megatron/core/tensor_parallel/layers.py` — ColumnParallelLinear, RowParallelLinear
- `megatron/core/tensor_parallel/mappings.py` — 通信原语

#### 5.1.1 ColumnParallelLinear（列并行）

将权重矩阵 **按列（输出维度）** 切分：

```
完整权重 W: [hidden_size, output_size]

TP=4 切分后：
  GPU 0: W_0 = W[:, 0:output_size/4]
  GPU 1: W_1 = W[:, output_size/4:output_size/2]
  GPU 2: W_2 = W[:, output_size/2:3*output_size/4]
  GPU 3: W_3 = W[:, 3*output_size/4:output_size]
```

**前向传播：**
```
输入 X: [seq, batch, hidden_size]  （每个 GPU 上都有完整副本）

GPU 0: Y_0 = X @ W_0  → [seq, batch, output_size/4]
GPU 1: Y_1 = X @ W_1  → [seq, batch, output_size/4]
GPU 2: Y_2 = X @ W_2  → [seq, batch, output_size/4]
GPU 3: Y_3 = X @ W_3  → [seq, batch, output_size/4]

如果 gather_output=True: AllGather → Y = [Y_0, Y_1, Y_2, Y_3]
如果 gather_output=False: 保持分片状态（给后续 RowParallel 使用）
```

**反向传播：**
```
收到梯度 dY: [seq, batch, output_size]
各 GPU:
  dX_i = dY_i @ W_i^T     → 局部梯度
  dW_i = X^T @ dY_i       → 局部权重梯度

最终: dX = AllReduce(dX_0, dX_1, ..., dX_p)  （跨 TP 组归约输入梯度）
```

**应用场景：** QKV 投影、MLP 的上投影（FC1）、Embedding 输出层

#### 5.1.2 RowParallelLinear（行并行）

将权重矩阵 **按行（输入维度）** 切分：

```
完整权重 W: [input_size, output_size]

TP=4 切分后：
  GPU 0: W_0 = W[0:input_size/4, :]
  GPU 1: W_1 = W[input_size/4:input_size/2, :]
  GPU 2: W_2 = W[input_size/2:3*input_size/4, :]
  GPU 3: W_3 = W[3*input_size/4:input_size, :]
```

**前向传播：**
```
输入已经是切分的: X_i: [seq, batch, input_size/4]

GPU 0: Y_0 = X_0 @ W_0  → [seq, batch, output_size]
GPU 1: Y_1 = X_1 @ W_1  → [seq, batch, output_size]
GPU 2: Y_2 = X_2 @ W_2  → [seq, batch, output_size]
GPU 3: Y_3 = X_3 @ W_3  → [seq, batch, output_size]

最终: Y = AllReduce(Y_0, Y_1, Y_2, Y_3)  （跨 TP 组归约输出）
```

**应用场景：** Attention 的输出投影、MLP 的下投影（FC2）

#### 5.1.3 Attention 的 TP 切分

```python
# megatron/core/transformer/attention.py, line 267-302
self.num_attention_heads_per_partition = divide(
    config.num_attention_heads, tp_world_size
)
# 例如: 96 个 attention head, TP=8 → 每个 GPU 处理 12 个 head
```

**完整流程（以 TP=4, num_heads=32 为例）：**

```
输入: [seq, batch, hidden=4096]  （所有 TP rank 相同）
                │
    ┌───────────┴───────────┐
    │   QKV ColumnParallel  │
    │   W_qkv: [4096, 3*4096] │
    │   切分: 每个 GPU [4096, 3*1024]  │
    └───────────┬───────────┘
                │
    各 GPU: q,k,v 各 [seq, batch, 1024] → 8 个 head × 128 dim
                │
    ┌───────────┴───────────┐
    │   Scaled Dot-Product  │
    │   Attention           │
    │   (FlashAttention)    │
    │   各 GPU 独立计算 8 个 head │
    └───────────┬───────────┘
                │
    各 GPU: attn_output [seq, batch, 1024]
                │
    ┌───────────┴───────────┐
    │   Output RowParallel  │
    │   W_o: [4096, 4096]    │
    │   切分: 每个 GPU [1024, 4096]  │
    │   → AllReduce 得到完整输出 │
    └───────────┬───────────┘
                │
    输出: [seq, batch, 4096]  （所有 TP rank 相同）
```

#### 5.1.4 MLP 的 TP 切分

```
输入: [seq, batch, hidden=4096]  （所有 TP rank 相同）
                │
    ┌───────────┴───────────┐
    │   FC1 ColumnParallel  │
    │   W1: [4096, 16384]    │
    │   切分: 每 GPU [4096, 4096]  │  (TP=4)
    └───────────┬───────────┘
                │
    各 GPU: [seq, batch, 4096]
                │
    ┌───────────┴───────────┐
    │   GeLU / SwiGLU       │
    │   （逐元素，无通信）    │
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │   FC2 RowParallel     │
    │   W2: [16384, 4096]    │
    │   切分: 每 GPU [4096, 4096]  │
    │   → AllReduce 得到完整输出 │
    └───────────┬───────────┘
                │
    输出: [seq, batch, 4096]
```

**每个 Transformer 层的 TP 通信量：**
- **2 次 AllReduce**（无 SP）或 **2 次 ReduceScatter + 2 次 AllGather**（有 SP）
- 每次通信数据量: `2 × seq_len × batch × hidden_size` bytes (AllReduce)

### 5.2 流水线并行 (Pipeline Parallelism, PP)

**核心思想：** 将模型的层按深度分割到不同的 GPU 组上，通过微批次流水线隐藏通信延迟。

**实现文件：**
- `megatron/core/pipeline_parallel/schedules.py` — 调度算法
- `megatron/core/pipeline_parallel/p2p_communication.py` — P2P 通信

#### 5.2.1 层的划分

```python
# 96 层模型, PP=8
# 每个 PP stage 获得 96/8 = 12 层

PP Stage 0 (GPU 0-7):  Layer 0-11   + Embedding
PP Stage 1 (GPU 8-15): Layer 12-23
PP Stage 2 (GPU 16-23): Layer 24-35
...
PP Stage 7 (GPU 56-63): Layer 84-95 + Output Layer + Loss
```

`pre_process=True` 表示该 stage 包含 Embedding 层，`post_process=True` 表示包含输出层。

#### 5.2.2 1F1B 调度（非交错）

```python
# megatron/core/pipeline_parallel/schedules.py
# forward_backward_pipelining_without_interleaving

# 三个阶段：
# 1. Warmup: 连续执行 forward，填充流水线
# 2. 1F1B Steady State: 交替执行 1 个 forward + 1 个 backward
# 3. Cooldown: 连续执行 backward，排空流水线
```

**以 PP=4, 8 个微批次为例：**

```
时间步 →  1    2    3    4    5    6    7    8    9   10   11
Stage 0: F0   F1   F2   F3   F4  B0   F5  B1   F6  B2   F7  B3  B4  B5  B6  B7
Stage 1:      F0   F1   F2   F3  B0   F4  B1   F5  B2   F6  B3  F7  B4  B5  B6  B7
Stage 2:           F0   F1   F2  B0   F3  B1   F4  B2   F5  B3  F6  B4  F7  B5  B6  B7
Stage 3:                F0   F1  B0   F2  B1   F3  B2   F4  B3  F5  B4  F6  B5  F7  B6  B7

Legend: F0=Forward microbatch 0, B0=Backward microbatch 0
```

**Warmup 微批次数：**
```python
num_warmup_microbatches = pp_size - pp_rank - 1
# Stage 0: 3 个 warmup
# Stage 1: 2 个 warmup
# Stage 2: 1 个 warmup
# Stage 3: 0 个 warmup
```

**流水线气泡：**
```
bubble_ratio = (pp_size - 1) / num_microbatches
# PP=4, M=8: bubble = 3/8 = 37.5%
# PP=4, M=32: bubble = 3/32 = 9.4%
# PP=4, M=64: bubble = 3/64 = 4.7%
```

→ 增大微批次数可有效减少气泡率

#### 5.2.3 P2P 通信

```python
# megatron/core/pipeline_parallel/p2p_communication.py
class P2PCommunicator:
    def recv_forward(self):
        """从上一阶段接收激活值"""
        # torch.distributed.irecv(tensor, src=prev_rank)

    def send_forward(self, output_tensor):
        """发送激活值到下一阶段"""
        # torch.distributed.isend(tensor, dst=next_rank)

    def send_forward_recv_backward(self, output_tensor):
        """重叠: 发送前向激活 + 接收反向梯度"""
        # batch_isend_irecv([send_op, recv_op])  # 一次系统调用完成两个操作
```

**通信数据量：** 每个微批次的阶段间通信 = `seq_len × micro_batch × hidden_size × dtype_size`

#### 5.2.4 重叠通信与计算

在 1F1B 稳态阶段：
```
Stage i 时间线:
  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Compute  │  │  Comm   │  │ Compute  │  │  Comm   │
  │ Forward  │  │Send+Recv│  │ Backward │  │Send+Recv│
  └─────────┘  └─────────┘  └─────────┘  └─────────┘

通过 send_forward_recv_backward() 实现：
  - 前向计算完成 → 同时发送激活 + 接收上一个微批次的梯度
  - 反向计算完成 → 同时发送梯度 + 接收下一个微批次的激活
```

### 5.3 数据并行 (Data Parallelism, DP)

**核心思想：** 每个 DP rank 持有完整的模型副本，处理不同的数据分片，通过梯度同步保持模型一致。

#### 5.3.1 基础 DP

```
假设全局 batch = 512, DP=8
每个 DP rank 处理 micro_batch = 512 / 8 = 64 个样本

DP Rank 0: data[0:64]    → forward → backward → grad_0
DP Rank 1: data[64:128]  → forward → backward → grad_1
...
DP Rank 7: data[448:512] → forward → backward → grad_7

AllReduce(grad_0, grad_1, ..., grad_7) → 平均梯度 → 所有 rank 一致更新
```

#### 5.3.2 分布式优化器 (Distributed Optimizer)

```python
# megatron/core/optimizer/distrib_optimizer.py
# 核心优化：将优化器状态（Adam 的 m, v）按 DP 维度分片
```

**内存节省原理：**

对于 Adam 优化器，每个参数需要存储：
- FP32 master weight (4 bytes)
- FP32 momentum m (4 bytes)
- FP32 variance v (4 bytes)
- 总计: 12 bytes/param

**未分片（标准 DP）：** 每个 GPU 存储全部优化器状态
```
每 GPU 优化器内存 = num_params × 12 bytes
```

**分片后（Distributed Optimizer）：** 只存自己负责的分片
```
每 GPU 优化器内存 = num_params × 12 / dp_size bytes
```

**通信模式变化：**
```
标准 DP:    AllReduce(gradients)    → 全量梯度同步
分布式优化器: ReduceScatter(gradients) → 每个 rank 只拿自己需要的梯度分片
           + AllGather(parameters)   → 更新后广播新参数
```

```
例: 175B 参数, DP=64, FP16 训练
标准 DP 每 GPU 优化器内存: 175B × 12 = 2100 GB  ← 单 GPU 装不下
分布式优化器每 GPU:        175B × 12 / 64 = 32.8 GB  ← 可行
```

#### 5.3.3 梯度桶通信

```python
# megatron/training/training.py, line 1391-1465
# 将参数分组到桶(bucket)中，桶满即发起通信
# 实现梯度归约与反向传播的重叠

config = DistributedDataParallelConfig(
    grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
    overlap_grad_reduce=args.overlap_grad_reduce,  # 反向传播时重叠梯度归约
    use_distributed_optimizer=args.use_distributed_optimizer,
    bucket_size=args.ddp_bucket_size,  # 桶大小（参数数量）
)
```

**重叠通信：**
```
反向传播时间线:
Layer N   反向 → 梯度放入桶 → 桶满 → 异步 ReduceScatter ─┐
Layer N-1 反向 → 梯度放入桶 →  ...                        │ 通信与计算重叠
Layer N-2 反向 → ...           ← ReduceScatter 完成 ────┘
...
```

### 5.4 序列并行 (Sequence Parallelism, SP)

**核心思想：** 在 TP 通信之间的 **非并行区域**（LayerNorm、Dropout）上，沿序列维度切分，减少激活内存。

SP 总是与 TP 搭配使用，它改变了 TP 的通信模式：

```
不使用 SP（标准 TP）:
  ColumnParallel → [独立计算] → RowParallel → AllReduce → LayerNorm
  通信: 2 × AllReduce per layer

使用 SP:
  ColumnParallel → [独立计算] → RowParallel → ReduceScatter → LayerNorm（序列切分）→ AllGather → ColumnParallel
  通信: 2 × ReduceScatter + 2 × AllGather per layer
  通信量相同，但激活内存减少 tp_size 倍！
```

**为什么内存更少？**

```
标准 TP - LayerNorm 输入: [seq_len, batch, hidden]     ← 全序列，所有 TP rank 相同
SP      - LayerNorm 输入: [seq_len/tp, batch, hidden]  ← 只存 1/tp 的序列

激活内存节省 = (tp_size - 1) / tp_size
TP=8: 节省 87.5% 的 LayerNorm/Dropout 激活内存
```

**通信原语（mappings.py）：**

```python
# scatter_to_sequence_parallel_region: 沿序列维度(dim=0)将数据等分给 TP 组
#   前向: Split → 每个 rank 拿 1/tp
#   反向: AllGather → 重组完整梯度

# gather_from_sequence_parallel_region: 从 TP 组收集完整序列
#   前向: AllGather → 合并为完整序列
#   反向: ReduceScatter → 分散梯度
```

### 5.5 上下文并行 (Context Parallelism, CP)

**核心思想：** 沿序列维度将输入切分到多个 GPU 上，支持超长序列训练。与 SP 不同，CP 切分的是 **输入数据**，而非中间激活。

```
原始序列长度: 128K tokens
CP=8: 每个 CP rank 处理 16K tokens

CP Rank 0: tokens[0:16K]
CP Rank 1: tokens[16K:32K]
...
CP Rank 7: tokens[112K:128K]
```

**Attention 的处理：**

由于 self-attention 需要每个位置关注所有其他位置，CP 使用 **Ring Attention** 机制：

```
Ring Attention (CP=4):

Round 0: Rank 0 本地 Q×K → 部分 attention
Round 1: K,V 环形传递 → Rank 0 用 Rank 3 的 K,V 计算
Round 2: K,V 环形传递 → Rank 0 用 Rank 2 的 K,V 计算
Round 3: K,V 环形传递 → Rank 0 用 Rank 1 的 K,V 计算
聚合所有 round 的结果 → 完整 attention output

通信: 每个 round 传递 K,V，通过环形重叠通信与计算
```

**数据切分（pretrain_gpt.py）：**

```python
# pretrain_gpt.py, line 154-161
if cu_seqlens is None and local_cp_size is None:
    batch = get_batch_on_this_cp_rank(batch)  # 标准 CP: 直接切序列
elif local_cp_size is None:
    batch, packed_seq_params = get_thd_batch_on_this_cp_rank(...)  # THD 打包格式
else:
    batch, packed_seq_params = get_batch_on_this_hybrid_cp_rank(...)  # 混合 CP
```

### 5.6 专家并行 (Expert Parallelism, EP)

**核心思想：** 对 Mixture-of-Experts (MoE) 模型中的专家进行分布式存放，每个 EP rank 仅持有部分专家。

```python
# megatron/core/transformer/moe/moe_layer.py, line 119-128
assert config.num_moe_experts % ep_size == 0
num_local_experts = config.num_moe_experts // ep_size
# 例如: 64 个专家, EP=8 → 每个 EP rank 持有 8 个专家
```

**Token 路由流程：**

```
输入: [seq_len, batch, hidden]
         │
    ┌────┴────┐
    │  Router │  → 每个 token 选择 top-k 个专家
    └────┬────┘
         │
    ┌────┴────────────────────────────────────┐
    │  All-to-All Token Dispatch              │
    │  每个 EP rank 发送 token 到目标专家所在 rank │
    │  同时接收来自其他 rank 的 token           │
    └────┬────────────────────────────────────┘
         │
    各 EP rank: 本地专家处理收到的 token
         │
    ┌────┴────────────────────────────────────┐
    │  All-to-All Token Combine               │
    │  将处理结果发送回原始 rank               │
    └────┬────────────────────────────────────┘
         │
    输出: [seq_len, batch, hidden]  （原始顺序恢复）
```

**通信模式：**
- **All-to-All**: 每个 rank 发送不同大小的数据到不同 rank
- 通信量取决于 token 路由的分布（负载均衡越好，通信越均匀）

**Token Dispatcher 类型：**

| Dispatcher | 方式 | 适用场景 |
|---|---|---|
| `MoEAlltoAllTokenDispatcher` | All-to-All | 常规场景 |
| `MoEAllGatherTokenDispatcher` | AllGather | 专家间负载方差小 |
| `MoEFlexTokenDispatcher` | 自适应 | 动态选择 |

### 5.7 虚拟流水线并行 (Virtual Pipeline Parallelism, VPP)

**核心思想：** 在每个物理 PP stage 上放置多个不连续的模型块（virtual stages），通过交错调度减少流水线气泡。

```python
# megatron/training/training.py, line 1299-1319
if vp_size > 1:
    model = []
    for i in range(vp_size):
        pre_process = is_pp_first_stage() and is_vp_first_stage(i)
        post_process = is_pp_last_stage() and is_vp_last_stage(i)
        model.append(model_provider(..., vp_stage=i))
```

**示例: 32 层, PP=4, VPP=2**

```
物理 Stage 0 (chunk 0): Layer 0-3    (Virtual Stage 0)
物理 Stage 1 (chunk 0): Layer 4-7    (Virtual Stage 1)
物理 Stage 2 (chunk 0): Layer 8-11   (Virtual Stage 2)
物理 Stage 3 (chunk 0): Layer 12-15  (Virtual Stage 3)
物理 Stage 3 (chunk 1): Layer 16-19  (Virtual Stage 4)  ← 反转
物理 Stage 2 (chunk 1): Layer 20-23  (Virtual Stage 5)
物理 Stage 1 (chunk 1): Layer 24-27  (Virtual Stage 6)
物理 Stage 0 (chunk 1): Layer 28-31  (Virtual Stage 7)

实际层分配:
  物理 Stage 0: Layer 0-3 + Layer 28-31  (两个不连续的块)
  物理 Stage 1: Layer 4-7 + Layer 24-27
  物理 Stage 2: Layer 8-11 + Layer 20-23
  物理 Stage 3: Layer 12-15 + Layer 16-19
```

**气泡减少：**

```
标准 PP=4:       bubble = (4-1)/M = 3/M
VPP=2, PP=4:     bubble ≈ (4-1)/(2*M) = 3/(2M)  ← 减半！
VPP=4, PP=4:     bubble ≈ (4-1)/(4*M) = 3/(4M)  ← 减至 1/4！
```

**交错调度示意（PP=4, VPP=2, M=4）：**

```
                    Forward                           Backward
Stage 0: F(0,c0) F(1,c0) F(0,c1) F(1,c1)  B(0,c1) B(1,c1) B(0,c0) B(1,c0) ...
Stage 1:   F(0,c0) F(1,c0) F(0,c1) F(1,c1)  B(0,c1) B(1,c1) B(0,c0) B(1,c0) ...
Stage 2:     F(0,c0) F(1,c0) F(0,c1) F(1,c1)  B(0,c1) B(1,c1) B(0,c0) ...
Stage 3:       F(0,c0) F(1,c0) F(0,c1) F(1,c1)  B(0,c1) B(1,c1) ...

F(m,c) = Forward microbatch m, chunk c
B(m,c) = Backward microbatch m, chunk c
```

---

## 6. 完整示例：128 GPU 训练 GPT-175B

### 6.1 模型配置

```bash
GPT-175B 模型参数:
  num_layers = 96
  hidden_size = 12288
  num_attention_heads = 96
  ffn_hidden_size = 49152  (4 × hidden_size)
  seq_length = 2048
  vocab_size = 50257
  total_params ≈ 175 Billion
```

### 6.2 并行配置

```bash
# 128 个 A100-80GB GPU (16 个节点, 每节点 8 GPU)
TP=8    # 节点内 8 GPU 用 NVLink 做张量并行
PP=4    # 4 个流水线阶段
DP=4    # 4 路数据并行
VPP=2   # 虚拟流水线并行（减少气泡）
SP=True # 序列并行（配合 TP）

验证: TP × PP × DP = 8 × 4 × 4 = 128 GPU ✓
```

### 6.3 启动命令

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=16 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=12345 \
  pretrain_gpt.py \
  --num-layers 96 \
  --hidden-size 12288 \
  --num-attention-heads 96 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --micro-batch-size 1 \
  --global-batch-size 1536 \
  --train-iters 500000 \
  --lr 6.0e-5 \
  --min-lr 6.0e-6 \
  --lr-warmup-iters 2000 \
  --lr-decay-iters 430000 \
  --lr-decay-style cosine \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --num-layers-per-virtual-pipeline-stage 12 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --bf16 \
  --use-flash-attn \
  --save-interval 1000 \
  --save /checkpoints/gpt-175b \
  --load /checkpoints/gpt-175b \
  --data-path /data/gpt_text_document \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file /data/gpt2-vocab.json \
  --merge-file /data/gpt2-merges.txt \
  --split 98,2,0 \
  --log-interval 10 \
  --eval-interval 500 \
  --eval-iters 10 \
  --ckpt-format torch_dist \
  --non-persistent-save-interval 100 \
  --async-save
```

### 6.4 GPU 编排详解

```
128 GPU 编号: 0-127

═══════════════════════════════════════════════════
TP 组（节点内 NVLink 连接, 8 GPU 一组, 共 16 组）:
═══════════════════════════════════════════════════
TP Group  0: GPU [0,  1,  2,  3,  4,  5,  6,  7]     ← Node 0
TP Group  1: GPU [8,  9,  10, 11, 12, 13, 14, 15]    ← Node 1
TP Group  2: GPU [16, 17, 18, 19, 20, 21, 22, 23]    ← Node 2
...
TP Group 15: GPU [120, 121, 122, 123, 124, 125, 126, 127]  ← Node 15

═══════════════════════════════════════════════════
PP 组（4 个阶段, 跨节点 IB 连接, 共 32 组）:
═══════════════════════════════════════════════════
PP Group 0:  GPU [0,  8,  16, 24]    ← TP rank 0, DP rank 0 的 4 个 PP stage
PP Group 1:  GPU [1,  9,  17, 25]    ← TP rank 1, DP rank 0
PP Group 2:  GPU [2,  10, 18, 26]    ← TP rank 2, DP rank 0
...
PP Group 7:  GPU [7,  15, 23, 31]    ← TP rank 7, DP rank 0
PP Group 8:  GPU [32, 40, 48, 56]    ← TP rank 0, DP rank 1
...
PP Group 31: GPU [103, 111, 119, 127] ← TP rank 7, DP rank 3

═══════════════════════════════════════════════════
DP 组（4 个副本, 跨节点 IB 连接, 共 32 组）:
═══════════════════════════════════════════════════
DP Group 0:  GPU [0,  32, 64, 96]    ← TP rank 0, PP stage 0 的 4 个 DP 副本
DP Group 1:  GPU [1,  33, 65, 97]    ← TP rank 1, PP stage 0
...
DP Group 31: GPU [31, 63, 95, 127]   ← TP rank 7, PP stage 3
```

### 6.5 层切分详解

```
═══════════════════════════════════════════════════
PP=4, VPP=2 的层分配 (96 层)
═══════════════════════════════════════════════════

每个虚拟 stage 的层数: 96 / (PP × VPP) = 96 / 8 = 12 层

物理 Stage 0 (GPU 0-7,  32-39, 64-71,  96-103):
  Chunk 0: Layer 0-11  (前 12 层)   + Embedding
  Chunk 1: Layer 84-95 (后 12 层)   + Output Layer

物理 Stage 1 (GPU 8-15, 40-47, 72-79,  104-111):
  Chunk 0: Layer 12-23
  Chunk 1: Layer 72-83

物理 Stage 2 (GPU 16-23, 48-55, 80-87,  112-119):
  Chunk 0: Layer 24-35
  Chunk 1: Layer 60-71

物理 Stage 3 (GPU 24-31, 56-63, 88-95,  120-127):
  Chunk 0: Layer 36-47
  Chunk 1: Layer 48-59
```

### 6.6 前向传播的一个微批次

```
数据流（单个微批次, micro_batch_size=1, seq_len=2048）:

═══ PP Stage 0 (Chunk 0): Layer 0-11 + Embedding ═══

[tokens: 2048 个 token ID]
        │
  Embedding: token_id → [2048, 1, 12288]
        │
  如果 SP=True: ScatterToSP → [2048/8, 1, 12288] = [256, 1, 12288]
        │
  Layer 0-11: 每层内:
    │ AllGather(SP→full) → [2048, 1, 12288]    ← SP 恢复全序列
    │ ColumnParallel QKV → [2048, 1, 12288*3/8] = [2048, 1, 4608]
    │ Attention(12 heads per TP rank) → [2048, 1, 12288/8]
    │ RowParallel Output → ReduceScatter → [256, 1, 12288]  ← 回到 SP
    │ LayerNorm (SP 域)
    │ AllGather(SP→full) → [2048, 1, 12288]
    │ ColumnParallel FC1 → [2048, 1, 49152/8]
    │ GeLU
    │ RowParallel FC2 → ReduceScatter → [256, 1, 12288]
    │ LayerNorm (SP 域)
        │
  P2P Send → [256, 1, 12288] 发送到 Stage 1  (via IB)

═══ PP Stage 1 (Chunk 0): Layer 12-23 ═══

  P2P Recv ← 从 Stage 0 接收
        │
  Layer 12-23 ... (同上)
        │
  P2P Send → Stage 2

═══ PP Stage 2 (Chunk 0): Layer 24-35 ═══
  ... (同上)

═══ PP Stage 3 (Chunk 0): Layer 36-47 ═══
  ... Layer 36-47

═══ PP Stage 3 (Chunk 1): Layer 48-59 ═══
  ... Layer 48-59
  P2P Send → Stage 2  (反向传递！)

═══ PP Stage 2 (Chunk 1): Layer 60-71 ═══
  ... 同上

═══ PP Stage 1 (Chunk 1): Layer 72-83 ═══
  ... 同上

═══ PP Stage 0 (Chunk 1): Layer 84-95 + Output ═══
  Layer 84-95
        │
  如果 SP=True: Gather 回 [2048, 1, 12288]
        │
  Output Layer: ColumnParallel → [2048, 1, 50257/8]
        │
  交叉熵损失 → loss scalar
```

### 6.7 微批次与全局批次的关系

```
global_batch_size = 1536
micro_batch_size = 1
DP = 4

每个 DP rank 处理: 1536 / 4 = 384 个样本
微批次数: 384 / 1 = 384 个微批次

即每个训练步:
  - 每个 DP rank 执行 384 次微批次的前向+反向
  - 梯度在 384 个微批次上累积
  - DP AllReduce/ReduceScatter 同步梯度
  - 优化器更新一次
```

### 6.8 内存分析

```
GPT-175B 每 GPU 内存估算（BF16, TP=8, PP=4, DP=4, VPP=2）:

模型参数:
  每 PP stage 参数量 ≈ 175B / 4 = 43.75B
  每 GPU (TP=8) 参数量 ≈ 43.75B / 8 = 5.47B
  VPP=2 → 每 GPU 持有 2 个 chunk ≈ 10.94B 参数
  BF16 存储: 10.94B × 2 bytes ≈ 21.9 GB

优化器状态 (分布式优化器, DP=4):
  每参数需 12 bytes (fp32 master weight + m + v)
  总计: 10.94B × 12 / 4 ≈ 32.8 GB → 分片后每 GPU ≈ 8.2 GB

梯度:
  BF16 梯度: 10.94B × 2 ≈ 21.9 GB
  分布式优化器 ReduceScatter: 实际只存 1/4 ≈ 5.5 GB

激活 (SP 开启):
  每层激活(粗略): seq_len × batch × hidden × dtype ÷ TP
  = 2048 × 1 × 12288 × 2 / 8 ≈ 6.3 MB/层
  24 层 × 微批次(若不做 checkpoint): ≈ 150 MB
  (激活检查点进一步减少)

总估算: ≈ 21.9 + 8.2 + 5.5 + 少量激活 ≈ ~40 GB
→ 可以 fit 在 80 GB A100 中
```

---

## 7. Checkpoint 保存机制

### 7.1 Checkpoint 格式

Megatron-LM 支持多种格式：

| 格式 | 类型 | 特点 |
|---|---|---|
| `torch` (Legacy) | 单文件/rank | 传统 torch.save, 每个 rank 一个文件 |
| `torch_dist` (Global) | 分布式 | 分片存储, 支持拓扑变更(resharding) |
| `torch_dcp` | PyTorch 原生 | 使用 torch.distributed.checkpoint |
| `local` | 本地非持久化 | 存在本地 SSD, 快速恢复 |

### 7.2 保存流程

```python
# megatron/training/checkpointing.py, line 479
def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, ...):

    # 1. 收集 RNG 状态（跨 TP/PP rank）
    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.random.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': get_cuda_rng_tracker().get_states(),
    }

    # 2. 收集 Rerun State Machine 状态（数据迭代器位置）
    rerun_state = rerun_state_machine.state_dict(data_iterator=train_data_iterator)

    # 3. 组装 state_dict
    state_dict = {
        'args': args,
        'checkpoint_version': 3.0,
        'iteration': iteration,
        'model': model.state_dict(),      # 模型权重
        'optimizer': optimizer.state_dict(),  # 优化器状态
        'opt_param_scheduler': scheduler.state_dict(),
        'rng_state': rng_state,
        'rerun_state_machine': rerun_state,
        'num_floating_point_operations_so_far': flops,
    }

    # 4. 根据格式保存
    if ckpt_type == CheckpointType.GLOBAL:  # torch_dist
        dist_checkpointing.save(state_dict, checkpoint_dir,
                               async_sharded_save=args.async_save)
    elif ckpt_type == CheckpointType.LEGACY:
        torch.save(state_dict, checkpoint_file)

    # 5. 更新 tracker 文件 (仅 rank 0)
    write_tracker_file('latest_checkpointed_iteration.txt', iteration)

    # 6. 清理旧检查点
    cleanup_old_checkpoints(...)
```

### 7.3 Checkpoint 目录结构

```
/checkpoints/gpt-175b/
├── latest_checkpointed_iteration.txt     # 记录最新迭代号: "10000"
│
├── iter_0010000/                          # 分布式格式
│   ├── metadata.json                      # 分片元数据
│   ├── common.pt                          # 非分片状态 (args, iteration 等)
│   └── __0_0.distcp ... __N_M.distcp     # 分片的张量文件
│
├── iter_0009000/
│   └── ...
│
└── non_persistent/                        # 非持久化检查点
    └── iter_0009500/
        └── ...
```

### 7.4 分布式 Checkpoint 的分片

```
torch_dist 格式下，每个 rank 保存自己持有的参数分片：

GPU 0 (TP=0, PP=0, DP=0):
  model.embedding.word_embeddings.weight  → 分片 [0:6282, :]  (TP 切分)
  model.decoder.layers.0.self_attention.linear_qkv.weight → 分片 [0:4608, :]
  optimizer.state.exp_avg → 对应参数的 DP 分片
  ...

GPU 1 (TP=1, PP=0, DP=0):
  model.embedding.word_embeddings.weight  → 分片 [6282:12564, :]
  model.decoder.layers.0.self_attention.linear_qkv.weight → 分片 [4608:9216, :]
  ...

metadata.json 记录:
  - 每个张量的全局 shape
  - 每个分片对应的全局坐标范围
  - 支持加载时拓扑不同（resharding）
```

### 7.5 异步保存

```python
# 异步保存流程:
# 1. 主线程准备 state_dict（拷贝到 CPU 或 pinned memory）
# 2. 启动后台线程/进程执行 I/O
# 3. 主线程继续训练
# 4. 下次保存前检查上次异步是否完成

save_checkpoint(..., async_save=True)
    │
    ├── state_dict 拷贝到 CPU  ← 同步（短暂暂停训练）
    │
    ├── 提交到 AsyncCallsQueue  ← 返回，训练继续
    │
    └── 后台: 写入磁盘          ← 与训练重叠
```

### 7.6 拓扑变更（Resharding）

分布式格式支持在加载时改变 TP/PP 配置：

```bash
# 保存时: TP=8, PP=4
# 加载时: TP=4, PP=8  ← 拓扑不同！

# 前提: 使用 --ckpt-fully-parallel-save 保存
# 加载流程:
#   1. 读取 metadata.json，获取每个张量的全局 shape 和分片信息
#   2. 根据新拓扑计算每个 rank 需要的分片范围
#   3. 从相应文件加载对应分片
#   4. 优化器状态需要 'fully_reshardable' 格式
```

**限制：**
- RNG 状态在拓扑变更时被忽略（随机性无法完美恢复）
- 优化器状态必须用 `--ckpt-fully-parallel-save` 才支持 resharding
- 模型权重始终支持 resharding

---

## 8. 故障恢复机制

### 8.1 检查点恢复（标准方式）

```python
# megatron/training/checkpointing.py, line 1558
def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load'):

    # 1. 查找最新检查点
    #    优先级: 非持久化 ckpt > 持久化 ckpt > pretrained ckpt > 从头训练
    iteration = read_tracker_file(checkpoint_dir)

    # 2. 加载 state_dict
    state_dict = _load_base_checkpoint(checkpoint_dir, ...)

    # 3. 恢复模型权重
    model.load_state_dict(state_dict['model'], strict=True)

    # 4. 恢复优化器状态
    optimizer.load_state_dict(state_dict['optimizer'])

    # 5. 恢复学习率调度器
    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])

    # 6. 恢复 RNG 状态（仅在拓扑完全匹配时）
    if ckpt_tp_pp == run_tp_pp and ckpt_world_size == run_world_size:
        random.setstate(rng_state['random_rng_state'])
        np.random.set_state(rng_state['np_rng_state'])
        torch.set_rng_state(rng_state['torch_rng_state'])
        torch.cuda.set_rng_state(rng_state['cuda_rng_state'])

    # 7. 恢复 Rerun State Machine（数据迭代器位置）
    if topology_matches:
        rerun_state_machine.load_state_dict(state_dict['rerun_state_machine'])

    return iteration, num_flops
```

### 8.2 非持久化检查点（快速恢复）

非持久化检查点用于频繁保存（如每 100 步），仅保留最新一个，实现快速故障恢复。

```
保存策略:
  - 持久化检查点: 每 1000 步，长期保留
  - 非持久化检查点: 每 100 步，只保留最新 1-2 个

恢复优先级:
  非持久化 iter 9500 > 持久化 iter 9000
  → 最多损失 100 步的训练，而非 1000 步
```

**两种非持久化格式：**

```
1. Global 非持久化（存共享文件系统）:
   save_dir/non_persistent/iter_xxx/
   自动清理旧版本: cleanup_old_non_persistent_checkpoint(leave_ckpt_num=1)

2. Local 非持久化（存本地 SSD/内存）:
   各节点本地存储，使用 nvidia_resiliency_ext 的 LocalCheckpointManager
   支持节点间复制（CliqueReplicationStrategy）以容忍节点故障
   恢复速度: 本地 SSD 读取 >> 共享文件系统读取
```

### 8.3 Rerun State Machine（重跑机制）

当检测到 NaN loss 或异常损失时，自动触发重跑：

```python
# megatron/core/rerun_state_machine.py
class RerunStateMachine:
    def validate_result(self, result, rejection_func, message, tolerance, fatal):
        """
        检查训练结果是否异常:
        - rejection_func=torch.isnan: 检测 NaN
        - rejection_func=torch.isinf: 检测 Inf
        - rejection_func=is_unexpectedly_large: 检测 Spiky Loss (>10x 历史最大)
        """
        if rejection_func(result):
            if fatal:
                self.request_rerun(reason=message)  # 请求重跑当前步
            else:
                self.log_warning(message)  # 仅警告
```

**重跑流程：**

```
正常训练: step 100 → step 101 → step 102 (NaN detected!)
                                    │
                                    ├── rerun_state_machine 记录异常
                                    ├── 回滚到 step 102 的初始状态
                                    ├── 恢复保存的 RNG 状态
                                    └── 重新执行 step 102 (使用保存的数据迭代器位置)
                                         │
                                         ├── 仍然 NaN? → 标记为 fatal, 保存 ckpt 并退出
                                         └── 正常 → 继续训练 step 103
```

### 8.4 进程内重启 (In-Process Restart)

```python
# pretrain_gpt.py, line 410
pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
```

当训练遇到可恢复的错误时，无需终止整个作业：

```
传统方式:
  训练崩溃 → 作业终止 → 重新排队 → 重新初始化 → 从 ckpt 恢复
  耗时: 分钟级（初始化）+ 加载时间

进程内重启:
  训练异常 → 回滚状态 → 从最近的 ckpt 恢复 → 继续训练
  耗时: 秒级
```

### 8.5 Fault Tolerance 集成

```python
# megatron/training/training.py
# 与外部容错系统（如 NVIDIA Fault Tolerance）集成:

ft_integration.setup()                   # 早期初始化
ft_integration.on_checkpointing_start()  # 保存前调整超时
ft_integration.on_checkpoint_loaded()    # 加载后通知
```

### 8.6 从检查点恢复的完整流程

```
作业中断: GPU 故障 / NCCL 超时 / OOM / 节点挂掉
    │
    ▼
作业调度器重启所有进程
    │
    ▼
pretrain_gpt.py::main()
    │
    ├── initialize_megatron()  → 重建分布式环境
    ├── _init_parallel_state() → 重建 TP/PP/DP/CP/EP 进程组
    │
    ├── setup_model_and_optimizer()
    │     ├── get_model()      → 重建模型（随机初始化）
    │     ├── get_optimizer()   → 重建优化器
    │     └── load_checkpoint() → 恢复训练状态
    │           │
    │           ├── 查找最新 ckpt
    │           │     非持久化 iter 9500 vs 持久化 iter 9000
    │           │     → 选择 iter 9500
    │           │
    │           ├── 加载模型权重（支持 resharding）
    │           ├── 加载优化器状态（含 Adam m, v）
    │           ├── 加载学习率调度器
    │           ├── 恢复 RNG 状态（确保 dropout 等可复现）
    │           └── 恢复数据迭代器位置
    │
    ├── build_data_iterators()
    │     └── 跳过已处理的样本（按恢复的迭代号）
    │
    └── train(start_iteration=9500)  → 从 step 9500 继续
          └── while iteration < 500000:
                train_step(...)  → 正常训练
```

---

## 9. 关键通信模式总结

### 9.1 通信原语一览

| 原语 | 方向 | 数据量 | 用途 |
|---|---|---|---|
| **AllReduce** | 所有→所有(全量) | 不变 | TP 梯度同步(无SP), DP 梯度同步(标准) |
| **ReduceScatter** | 所有→所有(分片) | 缩减为 1/N | TP+SP 反向, 分布式优化器梯度归约 |
| **AllGather** | 所有→所有(聚合) | 放大为 N 倍 | SP 前向恢复全序列, 分布式优化器参数广播 |
| **All-to-All** | 点对点(异构) | 不变 | MoE token dispatch/combine |
| **P2P Send/Recv** | 点对点 | 不变 | PP 阶段间激活/梯度传递 |
| **Broadcast** | 一→所有 | 放大为 N 倍 | 数据加载(TP rank 0 → 其他 TP rank) |

### 9.2 单个训练步的通信总结

以 6.2 节配置为例 (TP=8, PP=4, DP=4, SP=True, VPP=2):

```
一个 Transformer 层的通信 (TP 维度, 8 GPU, NVLink):
  ├── 2× AllGather (SP→full):  2 × seq × batch × hidden × dtype = 2 × 2048 × 1 × 12288 × 2 ≈ 100 MB
  └── 2× ReduceScatter (full→SP): 同上 ≈ 100 MB
  总计: ~200 MB/层

PP 阶段间通信 (IB):
  每个微批次: seq/tp × batch × hidden × dtype = 256 × 1 × 12288 × 2 ≈ 6.3 MB
  384 个微批次: 384 × 6.3 ≈ 2.4 GB (每个方向)

DP 梯度同步 (IB, 分布式优化器):
  ReduceScatter: total_params × dtype / dp_size = 10.94B × 2 / 4 ≈ 5.5 GB
  AllGather (参数更新后): 同上 ≈ 5.5 GB
```

### 9.3 通信拓扑优化

```
最优通信部署:
  TP (高带宽需求) → 节点内 NVLink (600-900 GB/s)
  PP (中等带宽需求) → 节点间 IB (相邻节点)
  DP (可重叠) → 节点间 IB (可与反向传播重叠)
  CP (高带宽需求) → 尽量节点内或相邻节点
  EP (All-to-All) → 取决于专家数量和路由模式
```

---

## 附录: 关键文件索引

| 组件 | 文件路径 | 描述 |
|---|---|---|
| 训练入口 | `pretrain_gpt.py` | GPT 训练入口脚本 |
| 模型构建 | `megatron/core/models/gpt/gpt_model.py` | GPT 模型定义 |
| 训练循环 | `megatron/training/training.py` | 训练主循环、train_step |
| 张量并行层 | `megatron/core/tensor_parallel/layers.py` | ColumnParallel, RowParallel |
| 通信原语 | `megatron/core/tensor_parallel/mappings.py` | AllReduce, ReduceScatter 等 |
| 流水线调度 | `megatron/core/pipeline_parallel/schedules.py` | 1F1B, 交错调度 |
| P2P 通信 | `megatron/core/pipeline_parallel/p2p_communication.py` | Send/Recv |
| 分布式优化器 | `megatron/core/optimizer/distrib_optimizer.py` | DP 梯度/参数分片 |
| 并行状态 | `megatron/core/parallel_state.py` | 进程组创建与管理 |
| 检查点 | `megatron/training/checkpointing.py` | 保存/加载/恢复 |
| 分布式检查点 | `megatron/core/dist_checkpointing/serialization.py` | 分布式 save/load |
| Attention | `megatron/core/transformer/attention.py` | 多头注意力(含 TP) |
| MLP | `megatron/core/transformer/mlp.py` | FFN 层(含 TP) |
| MoE 层 | `megatron/core/transformer/moe/moe_layer.py` | MoE + EP |
| Token 路由 | `megatron/core/transformer/moe/token_dispatcher.py` | All-to-All dispatch |
| 重跑机制 | `megatron/core/rerun_state_machine.py` | NaN/Spiky Loss 恢复 |
| 异步保存 | `megatron/training/async_utils.py` | 后台 checkpoint I/O |
