"""
演示：使用 HuggingFace transformers 的 Tokenizer，把「消息列表」渲染成模型真正消费的 Prompt 字符串。

核心 API：tokenizer.apply_chat_template(messages, ...)
- 每个模型在 HuggingFace 上都自带一份 chat template（jinja 模板），
  定义了如何把 [{role, content}, ...] 拼接成带特殊标记（special tokens）的最终文本。
- tokenize=False 时返回渲染后的字符串（便于观察），为 True 时直接返回 token id 列表。

运行：
    uv run python examples/chat_templates/Tokenizer.py
（首次运行会从 Hugging Face 下载对应模型的 tokenizer，体积很小，不含模型权重）
"""

from transformers import AutoTokenizer

# ---------- 示例 1：Qwen3 的 chat template 渲染 ----------
# 加载 Qwen3-8B 配套的 tokenizer（只下载 tokenizer 与 chat template，不含模型权重）
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 构造一段对话：system 设定角色 + user 提问
msgs = [
  {"role":"system","content":"你是 RCA 助手"},
  {"role":"user","content":"查根因"},
]
# 套用 Qwen3 自带的 chat template，把消息列表转成最终 Prompt 字符串
print(tok.apply_chat_template(msgs, tokenize=False))
# 输出真实送进模型的字符串，含所有特殊标记（如 im_start / im_end 等 role 分隔符）


print("-------------------------------------")


# ---------- 示例 2：带多轮对话 + assistant 续写占位的渲染 ----------
# 1) 加载带有 chat template 的 tokenizer（SmolLM2 指令微调模型）
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
)

# 一段包含 system / user / assistant 多轮交替的完整对话历史
conversation = [
    {"role": "system", "content": "你是电商客服，需要先确认订单号。"},
    {"role": "user", "content": "你好，我的包裹一直没到"},
    {"role": "assistant", "content": "请先提供你的订单号"},
    {"role": "user", "content": "订单号是 ORDER-123"},
]

# 2) 把消息列表转成“模型能吃”的 Prompt 字符串
rendered_prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,             # 如果为 True 就直接返回 token id
    add_generation_prompt=True  # 预留 assistant 回复的“开头”（在末尾追加 assistant 起始标记）
)

# 打印渲染结果：可看到每轮 role 都被特殊标记包裹，末尾留好了 assistant 续写位置
print(rendered_prompt)
