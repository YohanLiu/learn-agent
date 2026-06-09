"""LangGraph 工作流可视化工具。

提供统一的 show_graph 方法，供所有示例类复用。
IPython 环境内联显示，否则保存为 PNG 文件。
"""

from pathlib import Path

GRAPH_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "graphs"


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


class GraphVisualizer:
    """LangGraph 编译产物的可视化工具。

    用法::

        viz = GraphVisualizer()
        viz.show(compiled_workflow, "MyWorkflow")
        viz.show(compiled_workflow, "MyWorkflow", xray=True)
    """

    def __init__(self, output_dir: Path = GRAPH_OUTPUT_DIR):
        self._output_dir = output_dir

    def show(self, compiled_workflow, name: str, *, xray: bool = False) -> Path | None:
        """显示或保存工作流图。

        Args:
            compiled_workflow: LangGraph compile() 返回的 CompiledGraph。
            name: 图片文件名（不含 .png 后缀）。
            xray: 是否以 xray 模式渲染（显示内部状态结构）。

        Returns:
            IPython 环境返回 None，否则返回保存的文件路径。
        """
        png = compiled_workflow.get_graph(xray=xray).draw_mermaid_png()

        if _in_ipython():
            from IPython.display import Image, display
            display(Image(png))
            return None

        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{name}.png"
        path.write_bytes(png)
        print(f"Graph saved to {path}")
        return path
