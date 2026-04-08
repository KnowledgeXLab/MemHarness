# MemAdaptor 代码库下一步修改规划

本文档汇总 Memory Adaptor 与 Reasoning agent 在 MemAdaptor（verl + agent）中的集成计划：**可训练/可冻结**、双 WorkerGroup、I/O 契约与落地顺序。不写具体实现代码，仅作架构与任务清单。

---

## 1. 目标能力

### 1.1 Reasoning agent

- 负责环境动作与（可选）同模型下的「总结经验」类生成。
- 支持 **可训练 / 可冻结**（不进 optimizer 或 `requires_grad=False`）。
- 冻结时：**不需要**完整训练栈（优化器等），但 **仍需** 能 `generate_sequences`（或等价推理），除非完全改为闭源 API 等外包推理。

### 1.2 Memory Adaptor（独立 0.5B～1.5B LM；亦可与 Reasoning 同模）

**形式化 I/O（与实现对齐时写入 prompt 模板）：**

- **输入三元组** \((S_{curr}, S_{old}, P_{old})\)  
  - \(S_{curr}\)：当前环境可观测量（与 Reasoning 输入一致，如 AlfWorld 拼好的 text obs 或等价摘要）。  
  - \(S_{old}\)：检索命中对应的历史情境。  
  - \(P_{old}\)：历史经验原则（Principle / memory 文本）。
- **输出** \(P_{new}\)  
  - **Adaptation**：改写后的原则文本，注入 Reasoning 上下文。  
  - **Rejection**：输出约定字面 `<EMPTY>`（由 **system prompt** 约束即可，无需特殊 tokenizer token）；不注入原则内容，可对 Reasoning 贴 **固定「无相关经验」说明**，鼓励其仅靠观测推理。

**训练目标：** 最大化 **Reasoning agent 在当前任务上的 Reward**（轨迹级/步级与现有 GRPO/PPO 一致）。

**实现顺序建议：**

1. **推理-only**：跑通 I/O、注入、日志。  
2. **路径 B**：Reasoning 冻结，仅训练 Adaptor（REINFORCE/GRPO 等，视实现而定）。  
3. **路径 A**：Reasoning 与 Adaptor **联合 GRPO**。

### 1.3 双 `generate_sequences` / 双 WorkerGroup

- **Reasoning** 与 **Adaptor** 各可对应 **`actor_rollout_wg` 风格的一组 `RayWorkerGroup`**（名称上可区分为 reasoning / adaptor）。
- **同模**（同一 checkpoint、同一角色）时 **合并为一个 WorkerGroup**，总结与适配都走同一 `generate_sequences`，由 **prompt 角色** 区分任务。
- **经验总结若用 Reasoning**：调 Reasoning 的 wg；**记忆适配** 调 Adaptor 的 wg（或共享 wg + 不同 prompt）。

### 1.4 调度策略（缓解「Reasoning 冻结时 adaptor 有效样本少」）

- **部分 step 区间**：**每环境步都调用 Adaptor**（可无检索时用空占位 + prompt，让其多输出 EMPTY）。  
- **部分区间**：**仅在 Reasoning 触发记忆检索时** 调用 Adaptor，与产品行为一致。  
- 实现：在配置中增加调度开关/区间（如按全局 step 或局内步数），在 `rollout_loop` 或等价处分支。

### 1.5 多条检索（top-k）

- **语义**：一步内可有多条 \((S_{old}, P旧)\)。  
- **策略**：**每条分别调用 Adaptor**（各自 accept/reject），便于与你方 `<EMPTY>` 逐条丢弃一致；**实现上**将多条 prompt **打包成 batch 一次 `generate`** 加速。  
- **落地顺序**：先 **top-1**，再扩 **top-k 并行**、reject **逐条**。

### 1.6 vLLM 与 batch

- **Policy（Reasoning）** 的 `generate_sequences`：可继续 **vLLM / HF / openai_api**。  
- **Adaptor**：可 **HF batched generate** 或 **单独再起一套 vLLM 引擎**（与 policy 的 vLLM **两套部署**，不是单次调用自动推理两颗模型）。  
- **训练反向**：Adaptor/Reasoning 的梯度仍在 **PyTorch + FSDP（等）**；vLLM 仅服务 **采样/推理吞吐**。若采样用 vLLM、训练用 HF，需注意 **概率与超参一致性**（off-policy 风险）。

---

## 2. 配置与契约（优先完成）

| 项 | 说明 |
|----|------|
| `ppo_trainer.yaml` + Hydra | Reasoning / Adaptor：`model.path`、`trainable`、LoRA/FSDP；Adaptor `enable`、与同模共享 wg；prompt、最大生成长度、`<EMPTY>` 规则与「无经验」固定话术；**Adaptor 调度**；top-k / 先 top-1。 |
| I/O 文档 | \(S_{curr}, S_{old}, P_{old}\) 与现有 **obs / 检索字段** 的字段级映射，供 `rollout_loop` 填 batch。 |

---

## 3. Ray / Worker 拓扑

| 文件 | 修改要点 |
|------|----------|
| `verl/trainer/main_ppo.py` | 增加 **可选第二 Role**（如 Adaptor rollout）与 **resource_pool**；**同模**时不建第二组 worker。 |
| `verl/trainer/ppo/ray_trainer.py` | `self.actor_rollout_wg`（Reasoning）+ 可选 `self.adaptor_rollout_wg`；冻结侧 **跳过** `update_policy`，**保留** `generate_sequences`（除非完全 API 化）。 |

---

## 4. 模型与优化器

| 文件 | 修改要点 |
|------|----------|
| `verl/workers/fsdp_workers.py`（及拆出的工具模块） | 加载 **单/双 LM**；FSDP/PEFT 兼容；**param 分组**；frozen **不进 optimizer**；Adaptor 的 `compute_log_prob` / GRPO 字段随路径 B/A 逐步接入。 |

---

## 5. 数据流与 rollout

| 文件 | 修改要点 |
|------|----------|
| `agent_system/multi_turn_rollout/rollout_loop.py` 与记忆相关 env 路径 | **拼 Reasoning 的 `input_ids` 之前**：按调度决定是否跑 Adaptor；构造 **Adaptor 用 DataProto**，**batch 维** = 待适配条数；调用 **对应 wg 的 `generate_sequences`**；解析 `<EMPTY>` / \(P_{new}\) 与固定话术注入。 |
| `verl/protocol.py` / collate | 按需扩展 **adaptor 相关 tensor / non_tensor**，便于日志与 RL。 |

---

## 6. Rollout 引擎与训推一致

- 路径 B/A 过程中评估 **训练期是否统一 HF rollout**、或 **Adaptor 独立 vLLM 仅用于采样** 等取舍。  
- 联合训练阶段收紧 **sample 与 `compute_log_prob` 分布** 的一致性说明。

---

## 7. 检查点与评测

| 项 | 说明 |
|----|------|
| Checkpoint | Reasoning / Adaptor **分 key** 或 `extra`；支持只恢复 Adaptor。 |
| `examples/grpo_trainer/eval_alfworld-preexp.sh`（等） | Hydra 覆盖：**双路径**、**双冻结**、adaptor 调度、top-k 等。 |

---

