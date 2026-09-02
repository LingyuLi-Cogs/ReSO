"""Core attack interfaces.

The upstream package eagerly imported every attack here.  That makes a
text-only offline run import image and cloud-only optional dependencies before
an attack is selected.  The local runner imports the requested implementations
explicitly, so keep this package initializer deliberately lazy.
"""

from .base_attack import BaseAttack, AttackResult

__all__ = ["BaseAttack", "AttackResult"]
