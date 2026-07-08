from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

msgs = [
  {"role":"system","content":"你是 RCA 助手"},
  {"role":"user","content":"查根因"},
]
print(tok.apply_chat_template(msgs, tokenize=False))
# 输出真实送进模型的字符串，含所有 special tokens


print("-------------------------------------")


# 1) 加载带有 chat template 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
)

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
    add_generation_prompt=True  # 预留 assistant 回复的“开头”
)

print(rendered_prompt)
