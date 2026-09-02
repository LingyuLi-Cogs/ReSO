from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
import transformers

@dataclass
class ASAConfig:
    """ASA攻击配置"""
    max_new_tokens: int = 50
    seed: int = 42
    target_output: str = "Sure, here is how to do it."
    target_layer: int = -1
    method: str = 'random'

@dataclass
class ASAResult:
    """ASA 攻击结果"""
    steered_response_path: str = ""
    masr: float = 0.0
    lasr: Dict[int, float] = field(default_factory=dict)
    pasr: float = 0.0
