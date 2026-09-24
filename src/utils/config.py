"""配置加载工具。

支持 YAML 配置文件 + 命令行覆盖（--set key=value）。
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


class DotDict(dict):
    """支持点号访问的字典。"""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _convert_value(value: str) -> Any:
    """将字符串值转换为适当类型。"""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _auto_convert(obj: Any) -> Any:
    """递归地将字典/列表中的字符串数字转换为对应类型。"""
    if isinstance(obj, dict):
        return {k: _auto_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_auto_convert(v) for v in obj]
    if isinstance(obj, str):
        return _convert_value(obj)
    return obj


def _set_nested(config: Dict, key_path: str, value: Any) -> None:
    """设置嵌套字典值（key_path 格式：a.b.c）。"""
    keys = key_path.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _dict_to_dotdict(d: Dict) -> DotDict:
    """递归将字典转换为 DotDict。"""
    if isinstance(d, dict):
        return DotDict({k: _dict_to_dotdict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_dotdict(v) for v in d]
    return d


def load_config(config_path: str, cli_overrides: list = None) -> DotDict:
    """加载 YAML 配置，并应用 CLI 覆盖。

    Args:
        config_path: YAML 配置文件路径。
        cli_overrides: 命令行覆盖列表，每个元素格式为 "key=value" 或 "key.nested=value"。

    Returns:
        DotDict 配置对象。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = _auto_convert(config)

    if cli_overrides:
        for override in cli_overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override format: {override}")
            key_path, value_str = override.split("=", 1)
            value = _convert_value(value_str)
            _set_nested(config, key_path, value)

    return _dict_to_dotdict(config)


def parse_config_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--subject", type=str, default=None, help="Subject ID for Derco dataset")
    parser.add_argument("--frank_subject", type=str, default=None, help="Subject ID for Frank dataset")
    parser.add_argument("--n_folds", type=int, default=None, help="Number of folds for cross-validation")
    parser.add_argument("--fold", type=int, default=None, help="Current fold index (0-based)")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="Override config values (e.g., --set training.batch_size=32)")
    return parser.parse_args()
