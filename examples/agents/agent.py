from langchain.agents import create_agent
from models.model_factory import ModelFactory
from tools.weather import get_weather

# 创建模型
factory = ModelFactory()
model = factory.create_modelscope_chat_model()

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个乐于助人的智能助手",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "皇后镇今天天气怎么样？"}]}
)
print(result["messages"][-1].content_blocks)
