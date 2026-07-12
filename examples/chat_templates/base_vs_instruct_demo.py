"""
演示：system / user / assistant 是 special tokens，不是字符串标签。

运行前（在项目根目录）：
    uv sync
    uv pip install torch
    uv run python examples/chat_templates/base_vs_instruct_demo.py

首次运行会从 Hugging Face 下载两个 Qwen2.5-0.5B 模型（base + instruct，各约 1GB）。
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 同一系列的 base / instruct，对比最直观（各约 1GB）
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# 实验 1~4：对比「假标签」vs「真 chat template」
SYSTEM_PROMPT = "你是一个助手。无论用户问什么，你只回答两个字：收到。"
USER_QUESTION = "巴黎的首都是哪座城市？只回答城市名。"

FAKE_CHAT_PROMPT = f"""system: {SYSTEM_PROMPT}
user: {USER_QUESTION}
assistant:"""

MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_QUESTION},
]

# 实验 5：专门验证 system 优先级（user 问题很简单，若不服从 system 会直接答 "2"）
SYSTEM_PRIORITY_PROMPT = (
    "【最高优先级规则】你只能输出「收到」这两个字，禁止输出任何其他文字。"
)
USER_PRIORITY_QUESTION = "1+1等于几？"

PRIORITY_MESSAGES = [
    {"role": "system", "content": SYSTEM_PRIORITY_PROMPT},
    {"role": "user", "content": USER_PRIORITY_QUESTION},
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return tokenizer, model


def render_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(
    tokenizer,
    model,
    prompt: str,
    device: str,
    max_new_tokens: int = 60,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_section(title: str, prompt: str, completion: str) -> None:
    print("=" * 72)
    print(title)
    print("-" * 72)
    print("【输入 prompt】")
    print(prompt)
    print("-" * 72)
    print("【模型续写 / 回复】")
    print(completion or "(空)")
    print()


def main() -> None:
    device = pick_device()
    print(f"使用设备: {device}\n")

    print("加载 base 模型...")
    base_tokenizer, base_model = load_model(BASE_MODEL, device)

    print("加载 instruct 模型...")
    instruct_tokenizer, instruct_model = load_model(INSTRUCT_MODEL, device)
    print()

    # 实验 1：base + 手写假标签 → 只会续写
    print_section(
        "实验 1：base 模型 + 手写 'system:/user:/assistant:' 纯文本",
        FAKE_CHAT_PROMPT,
        generate(base_tokenizer, base_model, FAKE_CHAT_PROMPT, device),
    )

    # 实验 2：base + 普通问题 → 也是续写
    print_section(
        "实验 2：base 模型 + 普通问题（无角色标记）",
        USER_QUESTION,
        generate(base_tokenizer, base_model, USER_QUESTION, device),
    )

    # 实验 3：instruct + chat template（special tokens）→ 进入问答模式
    chat_prompt = render_chat_prompt(instruct_tokenizer, MESSAGES)
    print_section(
        "实验 3：instruct 模型 + apply_chat_template（special tokens，正确做法）",
        chat_prompt,
        generate(instruct_tokenizer, instruct_model, chat_prompt, device),
    )

    # 实验 4：同一 instruct 模型 + 手写假标签 → 和实验 3 对比，证明 special tokens 才是关键
    print_section(
        "实验 4：instruct 模型 + 手写 'system:/user:/assistant:' 纯文本（对照实验 3）",
        FAKE_CHAT_PROMPT,
        generate(instruct_tokenizer, instruct_model, FAKE_CHAT_PROMPT, device),
    )

    # 实验 5：instruct + chat template + 简单算术题 → 验证 system 优先级
    priority_prompt = render_chat_prompt(instruct_tokenizer, PRIORITY_MESSAGES)
    print_section(
        "实验 5：instruct 模型 + chat template，验证 system 优先级最高",
        priority_prompt,
        generate(instruct_tokenizer, instruct_model, priority_prompt, device, max_new_tokens=20),
    )

    print("=" * 72)
    print("怎么读结果：")
    print("- 实验 1：base 不认字符串标签，在 assistant: 后乱续写多轮对话")
    print("- 实验 2：base 把问题当普通文本续写，容易幻觉")
    print("- 实验 3 vs 4：同一 instruct 模型，special tokens 和假标签行为不同")
    print("- 实验 5：system 要求只答「收到」时，模型应覆盖 user 里的「1+1」问题")
    print("=" * 72)


if __name__ == "__main__":
    main()
