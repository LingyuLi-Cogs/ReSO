#!/usr/bin/env python3
"""Dependency-free catalog for the 26 project OpenRT core attacks."""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class AttackSpec:
    name: str
    display_name: str
    family: str
    backend: str
    module: str
    class_name: str
    needs_attacker: bool = False
    needs_embedding: bool = False


def _spec(name, display, family, backend, module, class_name, **kwargs):
    return AttackSpec(
        name=name,
        display_name=display,
        family=family,
        backend=backend,
        module=module,
        class_name=class_name,
        **kwargs,
    )


ATTACK_SPECS: Tuple[AttackSpec, ...] = (
    # White-box text attacks (the visual white-box method is intentionally absent).
    _spec("gcg", "GCG (NanoGCG)", "whitebox", "whitebox",
          "OpenRT.attacks.whitebox.implementations.nanogcg.attack", "NanoGCGAttack"),

    # Black-box optimization/fuzzing.
    _spec("autodan", "AutoDAN", "optimization", "blackbox",
          "OpenRT.attacks.blackbox.implementations.autodan", "AutoDAN_Attack",
          needs_attacker=True),
    _spec("gptfuzzer", "GPTFuzzer", "optimization", "blackbox",
          "OpenRT.attacks.blackbox.implementations.gptfuzzer.core", "GPTFuzzerAttack",
          needs_attacker=True),
    _spec("treeattack", "TreeAttack", "optimization", "blackbox",
          "OpenRT.attacks.blackbox.implementations.tree_attack", "TreeAttack",
          needs_attacker=True),
    _spec("seqar", "SeqAR", "optimization", "blackbox",
          "OpenRT.attacks.blackbox.implementations.seqar_attack", "SeqARAttack"),
    _spec("race", "RACE", "optimization", "blackbox",
          "OpenRT.attacks.blackbox.implementations.race.race_attack", "RACEAttack",
          needs_attacker=True),
    _spec("autodan_r", "AutoDAN-R", "optimization", "blackbox",
          "OpenRT.attacks.blackbox.implementations.autodan_turbo_r.autodan_turbo_r",
          "AutoDANTurboR", needs_attacker=True, needs_embedding=True),
    _spec("laa", "LAA (Adaptive Attack)", "optimization", "blackbox",
          "OpenRT.attacks.blackbox.implementations.adaptive_attack", "AdaptiveAttack"),

    # LLM refinement.
    _spec("pair", "PAIR", "llm_refinement", "blackbox",
          "OpenRT.attacks.blackbox.implementations.pair_attack", "PAIRAttack",
          needs_attacker=True),
    _spec("renellm", "ReNeLLM", "llm_refinement", "blackbox",
          "OpenRT.attacks.blackbox.implementations.renellm_attack", "ReNeLLMAttack",
          needs_attacker=True),
    _spec("drattack", "DrAttack", "llm_refinement", "blackbox",
          "OpenRT.attacks.blackbox.implementations.DrAttack.attack", "DrAttack",
          needs_attacker=True, needs_embedding=True),
    # Linguistic and encoding.
    _spec("cipherchat", "CipherChat", "linguistic", "blackbox",
          "OpenRT.attacks.blackbox.implementations.cipherchat.attack", "CipherChatAttack"),
    _spec("codeattack", "CodeAttack", "linguistic", "blackbox",
          "OpenRT.attacks.blackbox.implementations.CodeAttack.attack", "CodeAttack"),
    _spec("multilingual", "Multilingual", "linguistic", "blackbox",
          "OpenRT.attacks.blackbox.implementations.multilingual_attack",
          "MultilingualAttack", needs_attacker=True),
    _spec("jailbroken", "Jailbroken", "linguistic", "blackbox",
          "OpenRT.attacks.blackbox.implementations.jailbroken_attack",
          "JailBrokenAttack", needs_attacker=True),
    _spec("ica", "ICA", "linguistic", "blackbox",
          "OpenRT.attacks.blackbox.implementations.ica_attack", "ICAAttack"),
    _spec("flipattack", "FlipAttack", "linguistic", "blackbox",
          "OpenRT.attacks.blackbox.implementations.flipattack", "FlipAttack"),
    _spec("prefill", "Prefill", "linguistic", "blackbox",
          "OpenRT.attacks.blackbox.implementations.prefill_attack", "PrefillAttack"),
    _spec("pasttense", "Past Tense", "linguistic", "blackbox",
          "OpenRT.attacks.blackbox.implementations.past_tense_attack", "PastTenseAttack",
          needs_attacker=True),
    _spec("artprompt", "ArtPrompt", "linguistic", "blackbox",
          "OpenRT.attacks.blackbox.implementations.ArtPrompt.artprompt_attack",
          "ArtPromptAttack", needs_attacker=True),

    # Contextual.
    _spec("deepinception", "DeepInception", "contextual", "blackbox",
          "OpenRT.attacks.blackbox.implementations.deepinception_attack",
          "DeepInceptionAttack"),
    _spec("crescendo", "Crescendo", "contextual", "blackbox",
          "OpenRT.attacks.blackbox.implementations.crescendo_attack", "CrescendoAttack",
          needs_attacker=True),
    _spec("redqueen", "RedQueen", "contextual", "blackbox",
          "OpenRT.attacks.blackbox.implementations.redqueen_attack", "RedQueenAttack"),
    _spec("coa", "CoA", "contextual", "blackbox",
          "OpenRT.attacks.blackbox.implementations.coa.coa_attack", "CoAAttack",
          needs_attacker=True),

    # Multi-agent text attacks.
    _spec("actorattack", "ActorAttack", "multi_agent", "blackbox",
          "OpenRT.attacks.blackbox.implementations.actor_attack", "ActorAttack",
          needs_attacker=True),
    _spec("xteaming", "X-Teaming", "multi_agent", "blackbox",
          "OpenRT.attacks.blackbox.implementations.xteaming_attack", "XTeamingAttack",
          needs_attacker=True),
)


BY_NAME: Dict[str, AttackSpec] = {spec.name: spec for spec in ATTACK_SPECS}
BLACKBOX_ATTACKS: Tuple[str, ...] = tuple(
    spec.name for spec in ATTACK_SPECS
    if spec.backend == "blackbox"
)
PAPER_CORE_ATTACKS: Tuple[str, ...] = ("gcg",) + BLACKBOX_ATTACKS
DEFAULT_ATTACKS: Tuple[str, ...] = tuple(spec.name for spec in ATTACK_SPECS)


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


_ALIASES = {_key(spec.name): spec.name for spec in ATTACK_SPECS}
_ALIASES.update({_key(spec.display_name): spec.name for spec in ATTACK_SPECS})
_ALIASES.update({
    "nanogcg": "gcg",
    "autodanturbor": "autodan_r",
    "adaptiveattack": "laa",
    "pasttenseattack": "pasttense",
})


def normalize_attacks(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    raw = []
    for value in values or ("paper",):
        raw.extend(part.strip() for part in str(value).split(",") if part.strip())
    if any(_key(value) == "all" for value in raw):
        return DEFAULT_ATTACKS
    if any(_key(value) == "default" for value in raw):
        return PAPER_CORE_ATTACKS
    if any(_key(value) in {"blackbox", "allblackbox"} for value in raw):
        if len(raw) != 1:
            raise ValueError("blackbox cannot be combined with individual attacks")
        return BLACKBOX_ATTACKS
    if any(_key(value) in {"paper", "project"} for value in raw):
        if len(raw) != 1:
            raise ValueError("paper cannot be combined with individual attacks")
        return PAPER_CORE_ATTACKS

    selected = []
    for value in raw:
        key = _key(value)
        if key not in _ALIASES:
            raise ValueError(
                f"unknown attack {value!r}; choose from "
                + ", ".join(spec.name for spec in ATTACK_SPECS)
            )
        canonical = _ALIASES[key]
        if canonical not in selected:
            selected.append(canonical)
    return tuple(selected)


def select_by_backend(names: Iterable[str], backend: str) -> Tuple[str, ...]:
    return tuple(name for name in names if BY_NAME[name].backend == backend)


assert len(ATTACK_SPECS) == 26
assert len(BY_NAME) == 26
assert len(BLACKBOX_ATTACKS) == 25
assert len(PAPER_CORE_ATTACKS) == 26
assert "autodan_r" in BLACKBOX_ATTACKS
assert len(DEFAULT_ATTACKS) == 26
assert DEFAULT_ATTACKS[0] == "gcg"
assert DEFAULT_ATTACKS[-1] == "xteaming"
