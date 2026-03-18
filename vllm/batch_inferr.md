## vLLM Batch 推理完整流程

以下以 **离线 batch**（`LLM.generate()`）为主线，兼顾 OpenAI 兼容 batch API，详细说明整个推理流程。

---

### 一、请求提交阶段

#### 1. 入口：`LLM.generate()`
**文件：** `vllm/entrypoints/llm.py`

```python
llm = LLM(model="xxx")
outputs = llm.generate(["prompt1", "prompt2", ...], sampling_params)
```

调用链：
```
generate()
  → _run_completion()
    → _add_completion_requests()   # 逐个处理每个 prompt
      → _add_request(prompt, params)
        → llm_engine.add_request(request_id, prompt, params)
```

对于 **OpenAI 兼容 batch API**（`/v1/batch`），入口在 `vllm/entrypoints/openai/run_batch.py`：
- 读取 JSONL 文件，每行一个 `BatchRequestInput`
- 通过 `asyncio.gather()` 并发提交所有请求到 `AsyncLLM`

---

### 二、引擎接收与预处理

#### 2. LLMEngine.add_request()
**文件：** `vllm/v1/engine/llm_engine.py`

每个请求经历：

1. **InputProcessor 预处理**（`vllm/v1/engine/input_processor.py`）
   - Tokenization：将文本 prompt 转为 `prompt_token_ids`
   - 多模态特征提取（如有图片/音频）
   - 参数校验
   - 构建 `EngineCoreRequest` 对象

2. **OutputProcessor 注册**（`vllm/v1/engine/output_processor.py`）
   - 为每个请求创建 `RequestState` 跟踪输出状态
   - 设置 `output_kind = FINAL_ONLY`（batch 模式只返回最终结果）

3. **提交到 EngineCore**
   - `engine_core.add_request(request)` → 请求进入 **Scheduler 的 waiting 队列**

---

### 三、调度阶段

#### 3. Scheduler.schedule()
**文件：** `vllm/v1/core/sched/scheduler.py`

Scheduler 维护三个核心队列：
- **waiting**：新到达，尚未开始执行的请求
- **running**：正在生成 token 的请求
- **finished**：已完成的请求

每次调度（每个 step）执行两阶段选择：

**阶段一：调度 RUNNING 请求（decode）**
- 优先保障已在执行的请求继续生成
- 检查 KV cache 是否充足
- 每个 running 请求生成 1 个 token

**阶段二：调度 WAITING 请求（prefill）**
- 从 waiting 队列取新请求
- 为其分配 KV cache blocks
- 计算本轮可调度的 token 数（支持 chunked prefill）
- 直到 **token budget** (`max_num_batched_tokens`) 耗尽

输出 `SchedulerOutput`：
```python
SchedulerOutput:
  scheduled_new_reqs      # 首次调度的新请求
  scheduled_cached_reqs   # 之前已调度过的请求
  num_scheduled_tokens    # {req_id: token_count}
  total_num_scheduled_tokens
  finished_req_ids
```

> **关键点：vLLM 使用 continuous batching（连续批处理）**——不是凑齐一批再执行，而是每个 step 动态选择 prefill 和 decode 请求的最优组合。

---

### 四、模型执行阶段

#### 4. EngineCore.step()
**文件：** `vllm/v1/engine/core.py`

主循环：
```python
while has_unfinished_requests():
    scheduler_output = scheduler.schedule()        # 选择本轮请求
    output = executor.execute_model(scheduler_output)  # GPU 推理
    scheduler.update_from_output(output)           # 更新状态
```

#### 5. Executor → Worker → ModelRunner
**文件：** `vllm/v1/executor/abstract.py`

```
Executor.execute_model(scheduler_output)
  → Worker.execute_model()
    → ModelRunner.execute_model()
      → model.forward()           # 实际 GPU 前向计算
      → Sampler.forward()         # 采样下一个 token
```

- **Executor** 抽象了分布式后端（单进程/多进程/Ray）
- **Worker** 管理单个 GPU 上的执行
- **ModelRunner** 准备输入张量、执行 attention（含 KV cache 读写）、采样

返回 `ModelRunnerOutput`：
- 每个请求的新生成 token IDs
- logprobs（可选）
- 停止原因

---

### 五、输出处理阶段

#### 6. Scheduler 状态更新
```
scheduler.update_from_output(model_output)
```
- 将新 token 追加到每个请求的 output_token_ids
- 检查终止条件：
  - 遇到 EOS/stop token
  - 达到 max_tokens
  - 其他停止条件
- 满足条件的请求标记为 **FINISHED**

#### 7. OutputProcessor 处理
**文件：** `vllm/v1/engine/output_processor.py`

- **增量解码（detokenization）**：token_ids → 文本
- **Logprobs 格式化**
- 构建 `RequestOutput` 对象：
  ```python
  RequestOutput:
    request_id: str
    prompt: str
    outputs: [CompletionOutput]
      └── text: str          # 生成的文本
          token_ids: list     # token ID 序列
          finish_reason: str  # "stop" / "length"
          logprobs: dict      # 可选
  ```

---

### 六、结果收集与返回

#### 8. _run_engine() 收集结果
**文件：** `vllm/entrypoints/llm.py`

```python
while llm_engine.has_unfinished_requests():
    step_outputs = llm_engine.step()
    for output in step_outputs:
        if output.finished:
            outputs.append(output)
```

- 请求**不一定按提交顺序完成**（短 prompt 先完成）
- 最终按 `request_id` 排序后返回
- tqdm 显示进度条

---

### 七、完整流程图

```
用户调用 generate([prompts], params)
         │
         ▼
┌──────────────────────────────┐
│  逐个 add_request 到 Engine   │
│  InputProcessor: tokenize     │
│  OutputProcessor: 注册跟踪    │
│  → 全部进入 Scheduler.waiting │
└──────────┬───────────────────┘
           │
     ┌─────▼──────┐
     │  Main Loop  │◄──────────────────────────┐
     └─────┬──────┘                             │
           │                                    │
     ┌─────▼──────────────┐                     │
     │ Scheduler.schedule()│                    │
     │ ├─ RUNNING (decode) │ → 已运行的继续      │
     │ └─ WAITING (prefill)│ → 新请求加入        │
     └─────┬──────────────┘                     │
           │ SchedulerOutput                    │
     ┌─────▼──────────────┐                     │
     │ Executor→Worker     │                    │
     │ →ModelRunner.forward│ → GPU 前向+采样     │
     └─────┬──────────────┘                     │
           │ ModelRunnerOutput                  │
     ┌─────▼──────────────┐                     │
     │ 更新 Scheduler 状态  │                    │
     │ 标记 FINISHED 请求   │                    │
     └─────┬──────────────┘                     │
           │                                    │
     ┌─────▼──────────────┐    还有未完成请求？   │
     │ OutputProcessor     │────── Yes ─────────┘
     │ detokenize + 输出   │
     └─────┬──────────────┘
           │ No
     ┌─────▼──────────┐
     │ 排序，返回结果   │
     │ list[RequestOutput]│
     └────────────────┘
```

---

### 八、关键设计要点

| 特性 | 说明 |
|------|------|
| **Continuous Batching** | 不等凑齐一批，每个 step 动态组合 prefill + decode 请求 |
| **Token Budget** | `max_num_batched_tokens` 控制每步最大 token 数，防 OOM |
| **KV Cache 管理** | Scheduler 分配/回收 cache blocks，支持 PagedAttention |
| **Prefix Caching** | 共享前缀的请求复用 KV cache blocks |
| **Chunked Prefill** | 长 prompt 分多个 step 完成 prefill，避免阻塞 decode |
| **Preemption** | 内存不足时可暂停低优先级请求，释放 KV cache |
