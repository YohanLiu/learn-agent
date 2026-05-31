import os
import subprocess
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

# 从 .zshrc 加载环境变量
def load_env_from_zshrc():
    """从 .zshrc 文件加载环境变量到当前进程"""
    env_vars = {}
    
    # 读取 DEEPSEEK_API_KEY
    result = subprocess.run(
        ["zsh", "-c", "source ~/.zshrc && echo $DEEPSEEK_API_KEY"],
        capture_output=True,
        text=True
    )
    api_key = result.stdout.strip()
    if api_key:
        env_vars["DEEPSEEK_API_KEY"] = api_key
        os.environ["DEEPSEEK_API_KEY"] = api_key
        print(f"✓ 已加载 DEEPSEEK_API_KEY")
    
    # 读取 DEEPSEEK_BASE_URL
    result = subprocess.run(
        ["zsh", "-c", "source ~/.zshrc && echo $DEEPSEEK_BASE_URL"],
        capture_output=True,
        text=True
    )
    base_url = result.stdout.strip()
    if base_url:
        env_vars["DEEPSEEK_BASE_URL"] = base_url
        os.environ["DEEPSEEK_BASE_URL"] = base_url
        print(f"✓ 已加载 DEEPSEEK_BASE_URL: {base_url}")
    
    return env_vars

# 加载环境变量
env_config = load_env_from_zshrc()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 使用 langchain_deepseek 创建模型
model = ChatDeepSeek(
    model="deepseek-ai/DeepSeek-V4-Flash",
    api_key=env_config.get("DEEPSEEK_API_KEY"),
    api_base=env_config.get("DEEPSEEK_BASE_URL"),
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)