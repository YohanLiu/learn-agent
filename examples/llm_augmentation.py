"""LLM 增强示例：结构化输出 + 工具调用。

对应文档章节：LLMs and augmentations
https://docs.langchain.com/oss/python/langgraph/workflows-agents#llms-and-augmentations
"""

from pydantic import BaseModel, Field
from models.model_factory import ModelFactory


class LLMAugmentation:
    """演示 LLM 的两种增强方式：结构化输出（structured output）和工具调用（tool calling）。"""

    def __init__(self, model_factory: ModelFactory | None = None):
        self._factory = model_factory or ModelFactory()
        self.llm = self._factory.create_yunwu_chat_model()

    # ── 结构化输出 ──────────────────────────────────────────────

    class SearchQuery(BaseModel):
        search_query: str = Field(None, description="针对网络搜索优化的查询。")
        justification: str = Field(
            None, description="为什么此查询与用户的请求相关。"
        )

    def run_structured_output(self, question: str) -> "LLMAugmentation.SearchQuery":
        """使用结构化输出让 LLM 返回符合 Schema 的结果。

        Args:
            question: 用户问题。

        Returns:
            SearchQuery 实例。
        """
        structured_llm = self.llm.with_structured_output(self.SearchQuery)
        output = structured_llm.invoke(question)
        print("结构化输出：")
        print(output)
        return output

    # ── 工具调用 ────────────────────────────────────────────────

    @staticmethod
    def multiply(a: int, b: int) -> int:
        return a * b

    def run_tool_calling(self, question: str = "2乘以3等于多少？") -> list[dict]:
        """将工具绑定到 LLM 并触发工具调用。

        Args:
            question: 触发工具调用的问题。

        Returns:
            工具调用列表（tool_calls）。
        """
        llm_with_tools = self.llm.bind_tools([self.multiply])
        msg = llm_with_tools.invoke(question)
        print("工具调用：")
        print(msg.tool_calls)
        return msg.tool_calls


if __name__ == "__main__":
    demo = LLMAugmentation()
    demo.run_structured_output("钙化CT评分与高胆固醇有什么关系？")
    print("\n--- --- ---\n")
    demo.run_tool_calling("2乘以3等于多少？")
