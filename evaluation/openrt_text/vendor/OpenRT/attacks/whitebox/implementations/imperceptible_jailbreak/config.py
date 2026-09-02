from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ImperceptibleJailbreakConfig:
    """Imperceptible Jailbreak attack configuration.

    Based on: "Imperceptible Jailbreak Attacks on Large Language Models"
    Uses Unicode variation selectors to craft invisible adversarial suffixes
    that alter tokenization while remaining visually imperceptible.
    """
    n_iterations: int = 500
    n_chars_adv: int = 100
    n_chars_change_max: int = 50
    seed: int = 42
    max_new_tokens: int = 150
    target_tokens: Optional[List[List[str]]] = None
    target_prob_threshold: float = 0.2
    deterministic_jailbreak: bool = False
    n_restarts: int = 20


@dataclass
class ImperceptibleJailbreakResult:
    """Result of an imperceptible jailbreak attack."""
    goal: str = ""
    best_adv_suffix: str = ""
    best_msg: str = ""
    best_logprob: float = float('-inf')
    original_response: str = ""
    final_response: str = ""
    jailbroken_rule: bool = False
    n_queries: int = 0
    response_path: str = ""
    asr: float = 0.0
    results: List[Dict] = field(default_factory=list)
