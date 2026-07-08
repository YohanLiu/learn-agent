import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined

def raise_exception(message: str) -> None:
    raise ValueError(message)

messages = [
    {"role": "system", "content": "You are an experienced software engineer"},
    {"role": "user", "content": "help me fix the following bugs: xxx"},
]

env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = raise_exception
env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)

tpl_path = Path(__file__).parent / "Qwen3.5-397B-A17B_chat_template.jinja"
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

print("-----------------------------------")
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