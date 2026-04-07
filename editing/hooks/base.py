from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class AttentionHook(ABC):
    """Base class for attention hooks.

    Attention hooks allow modifying attention behavior during forward pass
    without permanently changing the model.
    """

    def __init__(self):
        self._restore_stack: List[Tuple[object, str, object]] = []

    @abstractmethod
    def patch_model(self, model: "torch.nn.Module") -> None:
        """Apply patches to the model.

        Args:
            model: Model to patch
        """
        pass

    def restore(self) -> None:
        """Restore all patched attributes."""
        while self._restore_stack:
            obj, attr, original = self._restore_stack.pop()
            setattr(obj, attr, original)

    def _save_original(self, obj: object, attr: str, original: Any) -> None:
        """Save original attribute for later restoration.

        Args:
            obj: Object being patched
            attr: Attribute name
            original: Original value
        """
        self._restore_stack.append((obj, attr, original))
