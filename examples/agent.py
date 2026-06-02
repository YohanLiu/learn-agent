from langchain.agents import create_agent
from models.model_factory import ModelFactory
from tools.weather import get_weather

# 创建模型
factory = ModelFactory()
model = factory.create_chat_model()

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
