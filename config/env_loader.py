import os
import subprocess


class EnvLoader:
    """从 .zshrc 文件加载环境变量到当前进程，支持按需加载和缓存。"""

    def __init__(self, zshrc_path: str = "~/.zshrc"):
        self._zshrc_path = zshrc_path
        self._cache: dict[str, str] = {}

    def load(self, *var_names: str) -> dict[str, str]:
        """
        加载指定的环境变量。

        Args:
            *var_names: 要加载的环境变量名，如 "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"

        Returns:
            包含已成功加载的环境变量字典
        """
        result = {}
        for name in var_names:
            value = self._read_var(name)
            if value:
                self._cache[name] = value
                os.environ[name] = value
                result[name] = value
                print(f"✓ 已加载 {name}" + (f": {value}" if name.endswith("URL") else ""))
        return result

    def load_all(self, var_names: list[str] | None = None) -> dict[str, str]:
        """
        批量加载环境变量。如果不传 var_names，则加载默认的 DeepSeek 相关变量。

        Args:
            var_names: 要加载的环境变量名列表，为 None 时使用默认列表

        Returns:
            包含已成功加载的环境变量字典
        """
        if var_names is None:
            var_names = ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"]
        return self.load(*var_names)

    def get(self, name: str) -> str | None:
        """获取已缓存的环境变量值，未找到则尝试从 os.environ 获取。"""
        return self._cache.get(name) or os.environ.get(name)

    def _read_var(self, name: str) -> str:
        """通过 source .zshrc 并 echo 指定变量来读取其值。"""
        result = subprocess.run(
            ["zsh", "-c", f"source {self._zshrc_path} && echo ${name}"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
