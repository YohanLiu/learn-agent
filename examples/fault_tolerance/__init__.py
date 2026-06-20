"""LangGraph 容错（Fault Tolerance）示例。

对应文档：https://docs.langchain.com/oss/python/langgraph/fault-tolerance

包含三大机制：
- Retries：自动重试失败的节点
- Timeouts：限制单次节点执行时长
- Error handling：重试耗尽后运行恢复函数

以及：
- Graph defaults：图级统一配置
- Graceful shutdown：协作式优雅关闭
"""
