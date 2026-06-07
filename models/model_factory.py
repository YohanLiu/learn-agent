from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from config.env_loader import EnvLoader


class ModelFactory:
    """模型工厂类，封装 DeepSeek 与 YUNWU（OpenAI 兼容）模型的创建逻辑。"""

    DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
    DEFAULT_YUNWU_MODEL = "gpt-5.4-nano"

    def __init__(self, env_loader: EnvLoader | None = None):
        """
        Args:
            env_loader: EnvLoader 实例，为 None 时自动创建并加载默认环境变量。
        """
        if env_loader is None:
            self._env_loader = EnvLoader()
            self._env_loader.load_all()
        else:
            self._env_loader = env_loader

    def create_chat_model(
        self,
        model: str = DEFAULT_MODEL,
        **kwargs,
    ) -> ChatDeepSeek:
        """
        创建 ChatDeepSeek 实例。

        Args:
            model: 模型名称，默认使用 DeepSeek-V4-Flash。
            **kwargs: 传递给 ChatDeepSeek 的额外参数，会覆盖默认配置。

        Returns:
            ChatDeepSeek 实例
        """
        return ChatDeepSeek(
            model=model,
            api_key=self._env_loader.get("DEEPSEEK_API_KEY"),
            api_base=self._env_loader.get("DEEPSEEK_BASE_URL"),
            **kwargs,
        )

    # https://www.yunwuai.cc/console
    def create_yunwu_chat_model(
        self,
        model: str = DEFAULT_YUNWU_MODEL,
        **kwargs,
    ) -> ChatOpenAI:
        """
        创建通过 YUNWU 中转的 ChatOpenAI 实例。

        Args:
            model: 模型名称，默认使用 gpt-5.4-nano。
            **kwargs: 传递给 ChatOpenAI 的额外参数，会覆盖默认配置。

        Returns:
            ChatOpenAI 实例
        """
        self._env_loader.load("YUNWU_API_KEY", "YUNWU_BASE_URL")
        return ChatOpenAI(
            model=model,
            api_key=self._env_loader.get("YUNWU_API_KEY"),
            base_url=self._env_loader.get("YUNWU_BASE_URL"),
            **kwargs,
        )
