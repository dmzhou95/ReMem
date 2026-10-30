# ReMem-LLM：可逆情景记忆路由参考实现

根据用户提供的论文 **ReMem-LLM: Reversible Episodic Memory Routing for Efficient Long-Context Reasoning**（`sn-article(10).pdf`）编写。

本项目是基于论文方法章节重新编写的参考代码，**不是作者原始代码，也不是已复现论文表格结果的官方实现**。包含可在 CPU 上运行的轻量 Transformer 演示，以及面向本地 Llama-3.2-3B-Instruct 的可选 LoRA 接口。论文没有充分规定的实现细节已在本文和 `docs/IMPLEMENTATION_NOTES.md` 中列出。

## 1. 已包含的内容

- 推理状态条件化的历史行为效用估计与自适应阈值路由。
- 活动行为、压缩记忆和恢复行为三类状态；显式保留原始索引。
- 连续非活动片段的效用加权压缩，以及熵、语义冲突和最大效用联合恢复。
- 完整历史自教师分支；任务、预测一致性、状态一致性、记忆、预算与恢复六项损失。
- 训练、检查点保存/加载、单样本推理与可读的路由轨迹。
- AUC、LogLoss、NDCG@10、Recall@10，以及上下文长度、延迟和 GPU 显存统计。
- 标准 CSV 数据转换、独立合成数据生成、配置文件和自动化测试。

压缩包不包含真实用户数据、受许可约束的 Llama 权重、第三方基线源码或论文结果的伪造日志。合成数据仅用于验证代码流程。

## 2. 文件结构

| 路径 | 用途 |
|---|---|
| `train.py` | 梯度累积训练、验证和模型保存 |
| `evaluate.py` | 点式/排序指标及实际运行开销 |
| `predict.py` | 单请求预测，输出每轮索引和恢复分数 |
| `prepare_data.py` | 标准交互 CSV → 按用户时间划分的 JSONL |
| `make_demo_data.py` | 生成明确标记的合成演示数据 |
| `remem/config.py` | 参数定义和校验 |
| `remem/backbones.py` | 轻量 Transformer 与本地 HF/LoRA 接口 |
| `remem/router.py` | 效用估计、阈值及直通门控 |
| `remem/memory.py` | 情景分组、压缩、来源索引 |
| `remem/restoration.py` | 不确定性感知恢复控制器 |
| `remem/model.py` | route–reason–restore 主循环 |
| `remem/losses.py` | 六项损失及加权总损失 |
| `remem/data.py` | JSONL 验证与读写 |
| `remem/metrics.py` | AUC、LogLoss 和分组排序指标 |
| `remem/utils.py` | 随机种子、设备和检查点 |
| `configs/tiny.json` | CPU 演示配置 |
| `configs/llama_lora.json` | Llama/LoRA 起始配置，不是已调优参数 |
| `examples/request.json` | 推理请求样例 |
| `tests/test_core.py` | 数值、恢复、梯度及检查点测试 |
| `tests/test_data.py` | 时间切分与历史泄漏检查 |
| `docs/IMPLEMENTATION_NOTES.md` | 公式对应、工程选择和实现边界 |
| `docs/DATA_PROTOCOL.md` | 真实数据准备与评价协议说明 |
| `docs/VALIDATION.md` | 本次实际执行的检查及限制 |

## 3. 安装

建议使用 Python 3.10 或更高版本；先解压并进入项目根目录。

```bash
cd remem_llm
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。GPU 版本的 PyTorch 应根据本机 CUDA 环境安装；基础演示不需要 GPU。项目仅依赖 PyTorch，指标计算无需安装 sklearn。

## 4. 一次跑通 CPU 演示

在项目根目录依次运行：

```bash
python -m unittest discover -s tests -v
python make_demo_data.py --output data/demo
python train.py --config configs/tiny.json --train data/demo/train.jsonl --valid data/demo/valid.jsonl --output runs/demo --device cpu
python evaluate.py --checkpoint runs/demo/best.pt --data data/demo/test.jsonl --output runs/demo/evaluation.json --device cpu
python predict.py --checkpoint runs/demo/best.pt --input examples/request.json --device cpu
```

产物包括 `best.pt`、`last.pt`、完整配置、逐轮训练日志与评估 JSON。`best.pt` 按验证集 AUC 选取；单类别验证集无法定义 AUC 时，退回到负 LogLoss。当前入口用于从头训练，不提供优化器状态断点续训。

演示只有 32 个训练候选、12 个验证候选和 12 个测试候选。每个候选组仅 4 项，因此 Recall@10 恒为 1，不能用于判断排序质量或论文性能；这里只检查数据—训练—保存—加载—评估链路。真实排序实验应提供有意义的候选集合。

## 5. 数据格式

每一行是一个候选样本，UTF-8 JSONL：

```json
{"group_id":"user42:request7","target":"A science documentary","history":["Read a book about planets","Watched an astronomy lecture"],"label":1}
```

- `target`：当前候选的文字描述。
- `history`：按时间由早到晚排列的历史行为文字列表，不能为空；超长时保留最近 `max_history` 项。
- `label`：二分类标签，0 或 1。
- `group_id`：同一次排序请求的候选集合标识；同组样本应共享历史和请求条件。
- 可额外记录 `user_id`、`item_id`，模型不会将其直接作为输入。

不要把当前候选的点击标签、未来行为或测试期之后的信息写入 `history`。排序指标按 `group_id` 分别计算后宏平均；AUC 和 LogLoss 则在全部候选上计算。AUC/NDCG/Recall 返回 0–1，若制表需要百分数，再乘 100；LogLoss 不乘 100。无正样本或单候选的排序组跳过，并记录数量；单类别 AUC 返回 JSON `null`。

如果已有统一格式的交互 CSV：

```bash
python prepare_data.py --input data/interactions.csv --output data/processed --min-interactions 10 --max-history 512
```

CSV 必需列：`user_id,item_id,timestamp,text,label`；可选 `impression_id`。时间必须可转换为数值，`label` 需提前转换为 0/1。脚本按每位用户的时间事件组进行 8:1:1 划分，同一时间点的候选不拆开，历史只纳入严格早于当前时刻的正反馈。它不是四个公开数据集的原始文件专用下载器。详见 `docs/DATA_PROTOCOL.md`。

## 6. 接入 Llama 与 LoRA

先准备具有使用权限的本地模型目录，然后安装可选依赖：

```bash
python -m pip install -r requirements-llm.txt
```

编辑 `configs/llama_lora.json` 中的 `model_path`，指向本地 Llama-3.2-3B-Instruct 目录，再运行：

```bash
python train.py --config configs/llama_lora.json --train data/processed/train.jsonl --valid data/processed/valid.jsonl --output runs/llama --device cuda
python evaluate.py --checkpoint runs/llama/best.pt --data data/processed/test.jsonl --output runs/llama/evaluation.json --device cuda
```

此接口以同一 LLM 的输入词嵌入均值表示每个行为，再将行为/记忆向量送入 LLM，最后一个状态 token 作为推理读出。因此这里的“token”指**行为级路由单元**，不等同于原始文本的每一个 BPE token。该编码与状态注入方式是为补足论文描述而采用的工程实现，详见实现说明。

采用本地加载、SDPA attention、可选 LoRA。GPU 支持时使用 bfloat16 主干；路由与损失使用 float32；不支持 bfloat16 的设备回退到 float32。没有集成量化加载、FlashAttention-2 或分布式训练。每次前向处理一个样本，`batch_size` 通过梯度累积实现有效批量，便于处理动态长度。

HF 检查点只保存可训练参数与 ReMem 模块；重新加载时仍须保留原始本地基础模型，且路径与模型版本一致。本次未运行 Llama 权重或 GPU 测试，不能保证论文所述 24GB 显存配置一定适配此参考实现。

## 7. 参数与训练目标

| 参数 | 演示默认值 | 含义 |
|---|---:|---|
| `steps` | 3 | 迭代推理轮数；论文未给出唯一值 |
| `episode_size` | 8 | 连续非活动片段最多包含的行为数；工程设定 |
| `kappa` | 0.5 | 阈值 `mean(utility) + kappa * std(utility)` |
| `temperature` | 0.3 | 软路由门温度 |
| `restore_threshold` | 0.5 | 恢复概率阈值 |
| `budget_ratio` | 0.3 | 目标预算与截断后历史长度的比值 |
| `margin` | 0.02 | 恢复一致性的改进 margin |
| `lambda_pred` | 1.0 | 预测分布一致性 |
| `lambda_state` | 0.5 | 推理状态一致性 |
| `lambda_mem` | 0.5 | 记忆语义一致性 |
| `lambda_budget` | 0.05 | 上下文预算超额惩罚 |
| `lambda_rest` | 0.5 | 恢复有效性 |

总损失：`Ltask + λp Lpred + λs Lstate + λm Lmem + λb Lbudget + λr Lrest`。

完整历史分支只在训练时计算，并将教师输出与状态停止梯度；没有单独预训练的教师。预测一致性为 `KL(teacher || routed)`，而非反向 KL。

预算是**软惩罚而非硬长度上限**。式 (20)、(33) 的预算按恢复前上下文计数；恢复后可能超出该预算。评估同时报告恢复前和恢复后长度，不能仅凭压缩率声称速度提升。所有原始行为嵌入仍保留以支持恢复，路由也扫描完整历史，教师和恢复重算都带来额外开销。

## 8. 论文一致性与已知边界

核心公式对应表、离散索引的梯度近似、状态集合优先级和工程补充均见 `docs/IMPLEMENTATION_NOTES.md`。

本项目没有自动复现 MovieLens-1M、MovieLens-25M、Amazon-Book、MIND 的结果，也没有实现论文中的 12 个外部基线。真实实验还需确定数据版本、候选构造、负样本规则、检索协议及超参数选择。提供的配置是起始配置，不能称作验证集最优配置。

本次实际检查以 `docs/VALIDATION.md` 为准。项目是可扩展的参考实现，不能把演示数据上的输出替换成论文表中的真实实验结果。
