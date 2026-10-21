"""
演示：同一条用户问题，通过 2 组实验对比不同 prompt 策略对模型行为的影响。

场景是一个「通勤建议助手」：
- 用户只问了一句信息不足的话：「今天适合骑车上班吗？」
  （缺城市、缺通勤时间/距离、缺天气，且模型没有实时工具）

两组实验设计：
- 实验 1（无 system / 基线）：模型大概率凭空假设并直接下结论；
- 实验 2（谨慎追问 system）：要求模型识别缺失信息并主动追问。

> 说明：本脚本默认使用 Qwen2.5-1.5B-Instruct（本地约 3GB，已缓存）。
> 实验 2 的「克制不答 + 识别缺信息 + 主动追问 + 声明边界」对 1.5B 偏难，可能不服从。
> 可通过 MODEL_ID 环境变量切到更大模型对比，如：
>   MODEL_ID=Qwen/Qwen2.5-3B-Instruct（实测可稳定追问），或 0.5B（大概率不服从）。

启动命令（均在项目根目录执行）：
    # 1) 首次准备：同步项目依赖 + 安装 torch（仅一次）
    uv sync
    uv pip install torch

    # 2) 默认运行（模型=1.5B，文件头已 setdefault 镜像与禁 Xet，首次会自动走镜像下载）
    uv run python examples/chat_templates/commute_prompt_demo.py

    # 3) 离线复跑：模型已缓存于 ~/.cache/huggingface 时加 HF_HUB_OFFLINE=1，
    #    不碰网络（代理开着也不会失败），加载最快
    HF_HUB_OFFLINE=1 .venv/bin/python examples/chat_templates/commute_prompt_demo.py

    # 4) 规模对比：MODEL_ID 切换三档已缓存的模型（启动时会打印当前模型名）
    #    0.5B：大概率不服从，演示“能力不够时 prompt 救不回来”
    MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct HF_HUB_OFFLINE=1 .venv/bin/python examples/chat_templates/commute_prompt_demo.py
    #    1.5B：默认值，临界档，实验 2 可能不服从甚至空输出
    MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct HF_HUB_OFFLINE=1 .venv/bin/python examples/chat_templates/commute_prompt_demo.py
    #    3B：实测可稳定执行实验 2 的追问约束（本机 M1 16GB 的上限档位）
    MODEL_ID=Qwen/Qwen2.5-3B-Instruct HF_HUB_OFFLINE=1 .venv/bin/python examples/chat_templates/commute_prompt_demo.py

环境变量小抄：
    MODEL_ID          选择模型，缺省 1.5B；三档均已在本地缓存
    HF_HUB_OFFLINE=1  纯用本地缓存，不联网校验（仅对已下载过的模型可用）
    HF_ENDPOINT       镜像地址，脚本内已 setdefault，手动指定可覆盖
    HF_HUB_DISABLE_XET=1  禁用 Xet 协议（镜像不代理它，不禁用下大文件会 401）
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

# 本地演示模型：指令微调版，具备基本的角色理解能力
# 支持环境变量切换模型做规模对比，如：MODEL_ID=Qwen/Qwen2.5-3B-Instruct
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

# 用户问题：一句话，但隐含大量缺失信息（城市、通勤时段、距离、实时天气都未知）
USER_QUESTION = "今天适合骑车上班吗？"

# system 提示：要求模型谨慎追问、识别缺失信息、声明实时能力边界，而不是直接下结论
SYSTEM_PROMPT = (
    "你是一个通勤建议助手。\n"
    "用户问今天适合出行吗之类的问题时，不要直接下结论。\n"
    "请先判断是否缺少城市、通勤时间、通勤距离、天气信息。\n"
    "如果缺少关键信息，请先追问。\n"
    "如果没有实时工具，请明确说明无法判断实时情况。"
)

# 实验 1：只有 user 问题，不带 system（作为基线）
MESSAGES_WITHOUT_SYSTEM = [
    {"role": "user", "content": USER_QUESTION},
]

# 实验 2：system + user，验证 system 约束能否改变模型行为
MESSAGES_WITH_SYSTEM = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_QUESTION},
]


def pick_device() -> str:
    """自动选择推理设备：优先 GPU(cuda) > Mac 加速(mps) > CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str):
    """加载指定模型的 tokenizer 与权重，并把模型搬到目标设备、切到评估模式。

    显式指定 dtype=torch.float16：Apple M1（16GB 统一内存）对 bfloat16 仅为软件模拟，
    float16 能走硬件原生加速并省下约一半内存；若不指定则默认 fp32 加载，较大模型
    （如 3B 约 12GB 权重）会直接触发 OOM。transformers 5.x 已将 torch_dtype 重命名为 dtype。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    max_new_tokens: int = 200,
) -> str:
    """对给定 prompt 做一次贪心解码（do_sample=False 保证结果可复现），返回新增的回复文本。

    max_new_tokens 设得比原演示更大（默认 200），因为追问类回复通常比两字结论更长。
    """
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
    print(f"当前模型: {MODEL_ID}\n")

    print("加载模型（已缓存则离线加载；首次需下载，1.5B 约 3GB）...")
    tokenizer, model = load_model(MODEL_ID, device)
    print()

    # 实验 1：无 system → 基线行为（大概率直接下结论或凭空假设）
    prompt_no_system = render_chat_prompt(tokenizer, MESSAGES_WITHOUT_SYSTEM)
    print_section(
        "实验 1：instruct 模型 + 只有 user 问题（不带 system，基线对照）",
        prompt_no_system,
        generate(tokenizer, model, prompt_no_system, device),
    )

    # 实验 2：有 system → 期望模型追问缺失信息 / 声明无法获取实时天气
    prompt_with_system = render_chat_prompt(tokenizer, MESSAGES_WITH_SYSTEM)
    print_section(
        "实验 2：instruct 模型 + system（谨慎追问约束）+ 同一 user 问题",
        prompt_with_system,
        generate(tokenizer, model, prompt_with_system, device),
    )

    print("=" * 72)
    print("怎么读结果：")
    print("- 实验 1（无 system）：模型容易凭空假设城市/天气，直接给「适合/不适合」结论")
    print("- 实验 2（谨慎 system）：理想情况下模型应反问缺失信息，或说明无法判断实时天气")
    print("  → 对比 1&2 可见：同一模型、同一问题，仅 system prompt 不同就能明显改变行为")
    print("- 若实验 2 仍直接下结论或空输出，说明 1.5B 对复杂多重约束服从度有限（正常）")
    print("- 切到 3B 可稳定复现追问：MODEL_ID=Qwen/Qwen2.5-3B-Instruct HF_HUB_OFFLINE=1 重跑")
    print("=" * 72)


if __name__ == "__main__":
    main()
