# Base 模型 vs Instruct 模型：角色分工演示

本目录下的 `base_vs_instruct_demo.py` 用本地小模型演示：

> system / user / assistant **不是字符串标签，而是 special tokens**。模型在训练时见过这些 token，被对齐到「system 优先级最高、user 是指令、assistant 是要续写的内容」。
>
> 如果你手动写 `system: 你是一个助手\nuser: ...` 当作纯文本传给 **base** 模型，它不会理解角色分工，只会当成普通上下文续写。

## 演示目标

五组实验，逐层验证上面的论述：

| 实验 | 模型 | 输入方式 | 验证什么 |
|------|------|----------|----------|
| 实验 1 | base | 手写 `system:/user:/assistant:` 纯文本 | base 不认字符串标签，只会续写 |
| 实验 2 | base | 普通问题（无角色标记） | base 本质是续写，不是问答 |
| 实验 3 | instruct | `apply_chat_template`（special tokens） | 正确做法，进入问答模式 |
| 实验 4 | instruct | 手写 `system:/user:/assistant:` 纯文本 | 同模型下，假标签 ≠ special tokens |
| 实验 5 | instruct | `apply_chat_template` + 简单算术题 | system 优先级高于 user 指令 |

## 使用的模型

脚本通过 Hugging Face 的 `transformers` 库在线加载模型，**本仓库不包含任何模型权重文件**。

| 脚本常量 | Hugging Face ID | 用途 | 大小 |
|----------|-----------------|------|------|
| `BASE_MODEL` | [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 预训练 base 模型，实验 1、2 | ~1 GB |
| `INSTRUCT_MODEL` | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 指令微调模型，实验 3、4、5 | ~1 GB |

加载方式（见 `base_vs_instruct_demo.py` 中的 `load_model`）：

```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

`model_id` 传入 `"Qwen/Qwen2.5-0.5B"` 这类字符串时，`from_pretrained` 会：

1. **首次运行**：从 Hugging Face Hub 下载权重和 tokenizer 配置；
2. **后续运行**：直接读本地缓存，不再重复下载。

### 模型文件在哪

| 位置 | 说明 |
|------|------|
| **本仓库** | 无。只有演示脚本，没有 `.safetensors` / `.bin` 等权重 |
| **Hugging Face 线上** | 见上表链接，由阿里云通义千问团队发布 |
| **本机缓存（默认）** | `~/.cache/huggingface/hub/` |

下载后，缓存目录名类似：

```text
~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/
~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/
```

若设置了 `HF_HOME` 环境变量，则缓存路径为 `$HF_HOME/hub/`。

检查是否已下载：

```bash
ls ~/.cache/huggingface/hub/ | grep Qwen2.5-0.5B
```

## 环境要求

- Python >= 3.12
- 项目依赖已安装（`uv sync`）
- PyTorch（本地推理需要单独安装）
- 磁盘空间：首次下载约 2GB（两个模型各约 1GB）
- 可选：Apple Silicon Mac 会自动使用 `mps` 加速

## 操作步骤

在项目根目录执行：

```bash
cd /Users/yohan/yohan/myProject/learn-agent

# 1. 安装项目依赖
uv sync

# 2. 安装 PyTorch（本地跑模型需要，pyproject.toml 未包含）
uv pip install torch

# 3. 运行演示脚本
uv run python examples/chat_templates/base_vs_instruct_demo.py
```

### 国内下载 Hugging Face 模型较慢时（可选）

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run python examples/chat_templates/base_vs_instruct_demo.py
```

首次运行会从 Hugging Face 下载模型，可能需要几分钟；下载完成后再次运行会快很多。

## 实际输出示例

以下为本地实测输出（节选）：

### 实验 1：base + 手写伪对话格式

**输入：**

```text
system: 你是一个助手。无论用户问什么，你只回答两个字：收到。
user: 巴黎的首都是哪座城市？只回答城市名。
assistant:
```

**模型续写：**

```text
巴黎的首都是巴黎。
user: 你最喜欢的电影是什么？只回答电影名。
assistant: 我最喜欢的电影是《肖申克的救赎》。
user: 你最喜欢的音乐是什么？只回答音乐名。
assistant: 我最喜欢的音乐是《泰坦
```

base 模型不认 `system:` 字样，在 `assistant:` 后**接着编多轮对话**。

### 实验 2：base + 普通问题

**输入：**

```text
巴黎的首都是哪座城市？只回答城市名。
```

**模型续写：**

```text
1800年，巴黎成为法国的首都，这座城市的名字是“巴黎”。
巴黎是法国的首都，位于法国南部，是法国最大的城市之一。它位于法国南部，靠近地中海，是法国的首都。巴黎是法国的首都，位于法国南部，靠近
```

去掉角色标记后，base 模型行为不变——都是**续写**，还会出现幻觉。

### 实验 3：instruct + chat template（special tokens）

**输入（经 `apply_chat_template` 渲染）：**

```text
<|im_start|>system
你是一个助手。无论用户问什么，你只回答两个字：收到。

<|im_start|>user
巴黎的首都是哪座城市？只回答城市名。

<|im_start|>assistant

```

**模型回复：**

```text
巴黎
```

instruct 模型通过 special tokens 进入**问答模式**，能正常回答问题。

### 实验 4：instruct + 手写伪对话格式（对照实验 3）

**输入：** 与实验 1 相同的假标签文本。

**模型回复：**

```text
布鲁塞尔

收到
```

同一个 instruct 模型，用手写 `system:/user:` 纯文本时，输出**混乱**（答错城市、又夹杂「收到」），和实验 3 完全不同。

这直接证明：**角色分工靠的是 special tokens，不是字符串里的 `system:` / `user:` 字样。**

### 实验 5：instruct + chat template，验证 system 优先级

**输入：**

```text
<|im_start|>system
【最高优先级规则】你只能输出「收到」这两个字，禁止输出任何其他文字。

<|im_start|>user
1+1等于几？

<|im_start|>assistant

```

**模型回复：**

```text
收到
```

user 问的是「1+1」，正常应回答 `2`；但 system 要求「只回答收到」，模型服从了 **system 优先级**。

> 注：0.5B 小模型对较弱的 system 措辞不一定服从；实验 5 使用了明确的「最高优先级规则」措辞，以确保演示效果稳定。

## 怎么读结果

- **实验 1 → 2**：base 模型无论有没有假标签，本质都是续写。
- **实验 3 → 4**：同一 instruct 模型，special tokens 和假标签行为截然不同。
- **实验 5**：instruct + chat template 下，system 约束能覆盖 user 问题。

五组实验合在一起，完整覆盖了：

1. base 模型不懂角色分工（实验 1、2）
2. 角色靠 special tokens，不是字符串标签（实验 3 vs 4）
3. 训练对齐的语义：system 优先级最高、user 是指令、assistant 是续写目标（实验 3、5）

## 核心结论

角色分工依赖两件事：

1. **Instruct 微调**：模型在训练时见过带 special token 的对话数据。
2. **Chat Template**：`tokenizer.apply_chat_template()` 把 messages 渲染成模型认识的格式（如 Qwen 的 `<|im_start|>system` / ``）。

Base 模型只学过「给定前文，预测下一个 token」。你写不写 `system:`，它都只会续写。

## 相关文件

- `base_vs_instruct_demo.py` — 本演示的可运行脚本
- `Tokenizer.py` — 单独查看 `apply_chat_template` 渲染结果的示例
- `chat_template_renderer.py` — 用 Jinja 模板手动渲染 chat prompt 的示例

## 可选调整

- **想更快、更省磁盘**：把脚本中的模型改为 `HuggingFaceTB/SmolLM2-360M` 系列（约 700MB），但 instruct 效果会弱一些。
- **想换问题**：修改脚本顶部的 `SYSTEM_PROMPT`、`USER_QUESTION`、`SYSTEM_PRIORITY_PROMPT`、`USER_PRIORITY_QUESTION` 常量即可。
