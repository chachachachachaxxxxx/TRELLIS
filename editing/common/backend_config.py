from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from typing import Mapping


SUPPORTED_ATTENTION_BACKENDS = ("flash_attn", "xformers")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def choose_attention_backend(requested_backend: str = "") -> str:
    requested = requested_backend.strip().lower()
    if requested:
        if requested not in SUPPORTED_ATTENTION_BACKENDS:
            raise RuntimeError(
                f"Unsupported attention backend '{requested}'. "
                f"Expected one of: {', '.join(SUPPORTED_ATTENTION_BACKENDS)}."
            )
        if not module_available(requested):
            raise RuntimeError(
                f"Requested attention backend '{requested}' is not installed in the current environment."
            )
        return requested

    for backend in SUPPORTED_ATTENTION_BACKENDS:
        if module_available(backend):
            return backend

    raise RuntimeError(
        "Neither flash_attn nor xformers is installed, but TRELLIS image editing needs one of them."
    )


@dataclass(frozen=True)
class BackendConfig:
    attn_backend: str
    sparse_attn_backend: str
    spconv_algo: str

    @property
    def env(self) -> dict[str, str]:
        return {
            "ATTN_BACKEND": self.attn_backend,
            "SPARSE_ATTN_BACKEND": self.sparse_attn_backend,
            "SPCONV_ALGO": self.spconv_algo,
        }


def build_backend_config(
    attn_backend: str = "",
    sparse_attn_backend: str = "",
    spconv_algo: str = "native",
) -> BackendConfig:
    resolved_attn = choose_attention_backend(attn_backend)

    sparse_requested = sparse_attn_backend.strip().lower()
    if sparse_requested:
        if sparse_requested not in SUPPORTED_ATTENTION_BACKENDS:
            raise RuntimeError(
                f"Unsupported sparse attention backend '{sparse_requested}'. "
                f"Expected one of: {', '.join(SUPPORTED_ATTENTION_BACKENDS)}."
            )
        if not module_available(sparse_requested):
            raise RuntimeError(
                f"Requested sparse attention backend '{sparse_requested}' is not installed in the current environment."
            )
        resolved_sparse = sparse_requested
    else:
        resolved_sparse = resolved_attn

    resolved_spconv = spconv_algo.strip() or "native"
    return BackendConfig(
        attn_backend=resolved_attn,
        sparse_attn_backend=resolved_sparse,
        spconv_algo=resolved_spconv,
    )


def apply_backend_env(
    config: BackendConfig,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.update(config.env)
    return env
