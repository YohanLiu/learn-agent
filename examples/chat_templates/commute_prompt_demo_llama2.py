"""
演示：同一条用户问题，通过 2 组实验对比不同 prompt 策略对 llama-2-7b 行为的影响。

本脚本是 commute_prompt_demo.py（Qwen2.5 instruct 本地版）的 API 对照版：
- 模型换成通过 YUNWU 中转调用的 llama-2-7b；
- 场景不变：用户只问了一句信息不足的话「今天适合骑车上班吗？」
  （缺城市、缺通勤时间/距离、缺天气，且模型没有实时工具）。

两组实验设计：
- 实验 1（无 system / 基线）：观察模型的默认行为；
- 实验 2（谨慎追问 system）：要求模型识别缺失信息并主动追问；

> ⚠️ 实测结论（本文件是「选型踩坑」负面教材，不适合做对照演示）：
> - YUNWU 的 llama-2-7b:stable 端点实际表现为对话式行为（并非预想的基座续写），
>   实验 2 的 system 约束它执行得很好——精准追问了用户在哪个城市；
> - 但实验 1（无 system 基线）它自己也主动追问了：「我需要更多的信息才能给出
>   一个准确的答案」，随后给出泛化免责声明——这类经过 RLHF 对齐的模型被训练得
>   「不确定就澄清」，对照组不再犯错，『加 system 才追问』的反差直接失效；
> - 结论：演示「无 system 直接下结论 vs 加 system 追问」应选自信不留余地的
>   中小 instruct 模型（如 Qwen2.5-7B-Instruct，见 commute_prompt_demo_qwen7b.py），
>   强对齐模型（llama-2 / GPT / Claude 类）做这个对照都会死在基线组上。
> ⚠️ 若报 `'str' object has no attribute 'model_dump'`，说明网关返回了非 JSON 响应
> （实测：旧域名 yunwuai.cc 已不再提供 API，会对所有请求兜底返回官网首页 HTML），
> 需在 YUNWU 控制台确认当前有效的 base_url（官方示例为 https://www.yw123.cc/v1）
> 并重新获取对应站点的 token（新旧站 token 不通用）。

启动命令（均在项目根目录执行，需 .env 中配置 YUNWU_API_KEY / YUNWU_BASE_URL）：
    # 实测两组实验都会追问，对照不成立，仅用于观察强对齐模型的基线行为
    .venv/bin/python examples/chat_templates/commute_prompt_demo_llama2.py

环境变量小抄：
    MODEL_ID   选择模型名，缺省 llama-2-7b:stable；档位后缀(:stable/:floor 等)按 YUNWU 控制台为准
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

from models.model_factory import ModelFactory

# 模型名支持环境变量切换，便于 base / chat 版对照实验（后缀取自 YUNWU 官方示例）
MODEL_ID = os.environ.get("MODEL_ID", "llama-2-7b:stable")

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
    HumanMessage(content=USER_QUESTION),
]

# 实验 2：system + user，验证 system 约束能否改变模型行为
MESSAGES_WITH_SYSTEM = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=USER_QUESTION),
]


def print_section(title: str, messages: list, completion: str) -> None:
    """统一格式化输出：分隔线 + 标题 + 输入消息 + 模型回复，便于逐实验对比。"""
    print("=" * 72)
    print(title)
    print("-" * 72)
    print("【输入消息】")
    for msg in messages:
        print(f"  [{msg.type}] {msg.content}")
    print("-" * 72)
    print("【模型回复】")
    print(completion or "(空)")
    print()


def main() -> None:
    print(f"当前模型: {MODEL_ID}（经 YUNWU 中转调用）\n")

    # 与官方示例保持一致的最小参数集：中转网关对 temperature 等额外参数
    # 兼容性不确定，且 llama-2 基座模型本身输出就不稳定，不再强求可复现
    llm = ModelFactory().create_yunwu_chat_model(model=MODEL_ID)

    # 实验 1：无 system → 基线行为
    reply_1 = llm.invoke(MESSAGES_WITHOUT_SYSTEM)
    print_section(
        "实验 1：llama-2-7b + 只有 user 问题（不带 system，基线对照）",
        MESSAGES_WITHOUT_SYSTEM,
        reply_1.content,
    )

    # 实验 2：有 system → 实测模型确实精准追问城市（服从良好），
    # 但因实验 1 基线也追问，本组对比看不出 system 带来的行为差异
    reply_2 = llm.invoke(MESSAGES_WITH_SYSTEM)
    print_section(
        "实验 2：llama-2-7b + system（谨慎追问约束）+ 同一 user 问题",
        MESSAGES_WITH_SYSTEM,
        reply_2.content,
    )

    print("=" * 72)
    print("怎么读结果（实测记录）：")
    print("- 实验 1（无 system）：模型没有如预期凭空下结论，而是自己先追问 + 免责声明")
    print("- 实验 2（有 system）：模型精准追问城市，说明 system 执行没问题")
    print("- 问题在于对照组失效：两组行为几乎一样，演示不出『加 system 才追问』的反差")
    print("- 根因：llama-2 这类经过 RLHF 对齐的模型被训练得『不确定就澄清』，")
    print("  不用 system 提醒它也很谨慎——演示提示词威力要选『敢答』的中小 instruct 模型")
    print("- 正面示范见同目录 commute_prompt_demo_qwen7b.py（DashScope 调用 qwen2.5-7b-instruct）")
    print("=" * 72)


if __name__ == "__main__":
    main()
