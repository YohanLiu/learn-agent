# Learn Agent

LangChain / LangGraph 学习项目，使用 DeepSeek 大模型进行 Agent 和 Tool Calling 的实践练习。

## 技术栈

- **Python** >= 3.12
- **LangChain** >= 1.3.2
- **LangChain DeepSeek** >= 1.0.1
- **uv** 包管理器

## 项目结构

```
learn-agent/
├── config/                # 配置模块
│   └── env_loader.py      # 从 .zshrc 加载环境变量
├── models/                # 模型模块
│   └── model_factory.py   # DeepSeek 模型工厂
├── tools/                 # 工具模块
│   └── weather.py         # 天气查询工具示例
├── examples/              # 练习示例
│   ├── agent.py           # 使用 create_agent 的 Agent 示例
│   └── tool.py            # 手动 Tool Calling 示例
└── pyproject.toml         # 项目配置
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 开发模式安装（使项目模块全局可导入）

```bash
uv pip install -e .
```

### 3. 配置环境变量

在 `~/.zshrc` 中添加：

```bash
export DEEPSEEK_API_KEY="your-api-key"
export DEEPSEEK_BASE_URL="your-base-url"
```

### 4. 运行示例

```bash
# Agent 示例（使用 create_agent）
python examples/agent.py

# Tool Calling 示例（手动调用工具）
python examples/tool.py
```

## 模块说明

### EnvLoader

从 `~/.zshrc` 动态加载环境变量，避免在代码中硬编码敏感信息。

```python
from config.env_loader import EnvLoader

loader = EnvLoader()
loader.load_all()  # 加载 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL
```

### ModelFactory

封装 DeepSeek 模型的创建逻辑，自动处理环境变量加载。

```python
from models.model_factory import ModelFactory

factory = ModelFactory()
model = factory.create_modelscope_chat_model()  # 默认使用 DeepSeek-V4-Flash
```

### Tools

使用 `@tool` 装饰器定义 LangChain 工具：

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."
```
