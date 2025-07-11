import tradingagents.default_config as default_config
from typing import Dict, Optional

# 使用默认配置但允许被覆盖
_config: Optional[Dict] = None
DATA_DIR: Optional[str] = None


def initialize_config():
    """使用默认值初始化配置。"""
    global _config, DATA_DIR
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
        DATA_DIR = _config["data_dir"]


def set_config(config: Dict):
    """使用自定义值更新配置。"""
    global _config, DATA_DIR
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
    _config.update(config)
    DATA_DIR = _config["data_dir"]


def get_config() -> Dict:
    """获取当前配置。"""
    if _config is None:
        initialize_config()
    return _config.copy()


# 使用默认配置初始化
initialize_config()
