"""
演示：不依赖 transformers，直接用 jinja2 手动渲染本地的 chat template（.jinja 文件）。

作用：
- HuggingFace 的 chat template 本质是一段 jinja2 模板，模型官方会把它导出为 xxx_chat_template.jinja。
- 本脚本把这段模板 + 一份 messages 数据交给 jinja2 引擎，直接渲染出最终 Prompt 字符串，
  方便在不加载 tokenizer / 模型的情况下，直观对比不同模型（Qwen vs GLM）模板的差异。

关键点：HF 模板里用到了一些非 jinja2 内置的自定义函数/过滤器，渲染前必须先注册：
- raise_exception：模板内做参数校验时调用（如 role 非法时抛错）。
- tojson：把 tools 等结构序列化成 JSON 字符串（ensure_ascii=False 保留中文）。

运行：
    uv run python examples/chat_templates/chat_template_renderer.py
"""

import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined


def raise_exception(message: str) -> None:
    """模板中可通过 raise_exception(...) 主动抛错，用于校验非法输入（如未知 role）。"""
    raise ValueError(message)


# 待渲染的对话数据：所有模板共享同一份 messages，便于横向对比输出差异
messages = [
    {"role": "system", "content": "You are an experienced software engineer"},
    {"role": "user", "content": "help me fix the following bugs: xxx"},
]

# 构建 jinja2 环境：
# - StrictUndefined：模板引用未传入的变量时直接报错，避免静默渲染出空值
# - trim_blocks / lstrip_blocks：让 {% %} 控制标签产生的多余空行被清理，输出更干净
env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
# 注册 HF 模板依赖的全局函数与自定义过滤器
env.globals["raise_exception"] = raise_exception
env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)

# ---------- 渲染 Qwen3.5 的 chat template ----------
tpl_path = Path(__file__).parent / "Qwen3.5-397B-A17B_chat_template.jinja"
tpl = env.from_string(tpl_path.read_text(encoding="utf-8"))
print(
    tpl.render(
        messages=messages,          # 对话内容
        tools=None,                 # 无工具调用（传工具 schema 列表可渲染工具分支）
        add_generation_prompt=True, # 末尾追加 assistant 起始标记，预留续写位置
        add_vision_id=False,        # 是否为多模态图片分配 id
        enable_thinking=True,       # 是否开启思考（thinking）模式标记
    )
)

print("-----------------------------------")

# ---------- 渲染 GLM-4.7 的 chat template（同一份数据，对比模板差异）----------
tpl_path = Path(__file__).parent / "GLM-4.7_chat_template.jinja"
tpl = env.from_string(tpl_path.read_text(encoding="utf-8"))
print(
    tpl.render(
        messages=messages,
        tools=None,
        add_generation_prompt=True,
        add_vision_id=False,
        enable_thinking=True,
    )
)
