from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from config.env_loader import EnvLoader


class ModelFactory:
    """模型工厂类，封装 DeepSeek 与 YUNWU（OpenAI 兼容）模型的创建逻辑。"""

    DEFAULT_MODELSCOPE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
    DEFAULT_YUNWU_MODEL = "gpt-5.4-nano:floor"
    DEFAULT_DASHSCOPE_MODEL = "qwen3.7-plus"

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

    # https://modelscope.cn/models?filter=inference_type&page=1&tabKey=task
    def create_modelscope_chat_model(
        self,
        model: str = DEFAULT_MODELSCOPE_MODEL,
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
            api_key=self._env_loader.get("MODELSCOPE_API_KEY"),
            api_base=self._env_loader.get("MODELSCOPE_BASE_URL"),
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
        return ChatOpenAI(
            model=model,
            api_key=self._env_loader.get("YUNWU_API_KEY"),
            base_url=self._env_loader.get("YUNWU_BASE_URL"),
            **kwargs,
        )

    # https://bailian.console.aliyun.com/cn-beijing?tab=model#/model-usage/free-quota
    # qwen 的模型结构化输出参考文档:https://help.aliyun.com/zh/model-studio/qwen-structured-output?spm=a2ty02.42736056.0.0.379774a1qUlvcT&scm=20140722.S_help@@%E6%96%87%E6%A1%A3@@2862209@@61.S_llmOS0.ID_95211-RL_%E7%BB%93%E6%9E%84%E5%8C%96%E8%BE%93%E5%87%BA%E5%B9%B6%E6%B2%A1%E6%9C%89%E6%8C%89%E7%85%A7%E6%88%91%E7%9A%84%E9%A2%84%E6%9C%9F%E8%BF%9B%E8%A1%8C%E8%BE%93%E5%87%BA-LOC_chat~DAS~llm-OR_ser-PAR1_0be37d3f17810010489491053d0a6c-V_4-P0_0-P1_0
    # 找客服说还要关闭深度思考才可以,麻烦,不如用gpt的模型省事了
    def create_dashscope_chat_model(
        self,
        model: str = DEFAULT_DASHSCOPE_MODEL,
        enable_thinking: bool = False,
        **kwargs,
    ) -> ChatOpenAI:
        """
        创建通过 Dashscope 的 ChatOpenAI 实例。

        Args:
            model: 模型名称，默认使用 qwen3.7-max。
            enable_thinking: 是否启用 Qwen3 深度思考，默认关闭。
            **kwargs: 传递给 ChatOpenAI 的额外参数，会覆盖默认配置。

        Returns:
            ChatOpenAI 实例
        """
        return ChatOpenAI(
            model=model,
            api_key=self._env_loader.get("DASHSCOPE_API_KEY"),
            base_url=self._env_loader.get("DASHSCOPE_BASE_URL"),
            extra_body={"enable_thinking": enable_thinking},
            **kwargs,
        )