"""
演示：system / user / assistant 是 special tokens（模型训练时学到的角色标记），而不是普通字符串标签。

通过 5 组对照实验说明一个核心结论：
- base（预训练）模型只会「续写文本」，不认识手写的 system:/user: 这类纯文本标签；
- instruct（指令微调）模型认得 chat template 渲染出的 special tokens，会进入问答模式；
- 只有走 apply_chat_template 生成正确的角色标记，system 指令才会生效（并拥有最高优先级）。

运行前（在项目根目录）：
    uv sync
    uv pip install torch
    uv run python examples/chat_templates/base_vs_instruct_demo.py

首次运行会从 Hugging Face 下载两个 Qwen2.5-0.5B 模型（base + instruct，各约 1GB）。

规模选择：默认 0.5B。经 0.5B / 1.5B 实测对比，0.5B 更贴合本示例的教学目标
（实验 3 正确答出「巴黎」，而 1.5B 会答成「罗马」反而误导）。如需对比其它规模：
    MODEL_SIZE=1.5 uv run python examples/chat_templates/base_vs_instruct_demo.py
"""

from __future__ import annotations

import os

# 必须在 import transformers / huggingface_hub 之前设置：huggingface_hub 在导入时
# 就会读取这些环境变量，之后再改不生效。
# - HF_ENDPOINT：直连 huggingface.co 在国内会连接超时，改走国内镜像；
#   用 setdefault 是为了不覆盖用户在 shell / PyCharm 里手动指定的值。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# - HF_HUB_DISABLE_XET：镜像站不代理 Xet 存储协议，不禁用会导致大文件下载失败。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 同一系列的 base / instruct，对比最直观
# 可通过环境变量 MODEL_SIZE 在 0.5 / 1.5 间切换，默认 0.5
MODEL_SIZE = os.environ.get("MODEL_SIZE", "0.5")
BASE_MODEL = f"Qwen/Qwen2.5-{MODEL_SIZE}B"
INSTRUCT_MODEL = f"Qwen/Qwen2.5-{MODEL_SIZE}B-Instruct"

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
    """自动选择推理设备：优先 GPU(cuda) > Mac 加速(mps) > CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str):
    """加载指定模型的 tokenizer 与权重，并把模型搬到目标设备、切到评估模式。"""
    print(f"  → 正在加载: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # M1 上 1.5B 双模型同时加载时，float32 会 OOM；显式用 float16 走硬件原生加速、减半内存
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    model.to(device)
    model.eval()
    return tokenizer, model


def render_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """用模型自带的 chat template 把消息列表渲染成正确的 Prompt（含 special tokens）。

    add_generation_prompt=True 会在末尾补上 assistant 起始标记，引导模型续写回复。
    """
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
    """对给定 prompt 做一次贪心解码（do_sample=False 保证结果可复现），返回新增的回复文本。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 切掉输入部分，只保留模型新生成的 token，再解码成可读文本
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_section(title: str, prompt: str, completion: str) -> None:
    """统一格式化输出：分隔线 + 标题 + 输入 prompt + 模型回复，便于逐实验对比。"""
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
    print(f"使用设备: {device}")
    print(f"本次模型规模: Qwen2.5-{MODEL_SIZE}B（由环境变量 MODEL_SIZE 控制，默认 0.5）")
    print(f"  base    模型: {BASE_MODEL}")
    print(f"  instruct模型: {INSTRUCT_MODEL}\n")

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
