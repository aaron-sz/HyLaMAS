"""Theorem 3.1 bookkeeping: text tokens a latent thought stands in for.

The ratio is a property of the checkpoint (hidden size against the entropy of
one sampled token) and is configured once, then used to annotate traces and the
`text_equivalent_tokens` column. Returns None when it cannot be determined.
"""

import math
from typing import Optional

# 2.0 = information-theoretic (bits per token). 3.0 = the paper's figures.
LOG_BASE = 2.0

_RATIO: Optional[float] = None
_INFO: Optional[dict] = None


def efficiency_ratio(hidden_size: int, vocab_size: int,
                     log_base: float = LOG_BASE) -> Optional[float]:
    """R = d_h / log_b|V|, text tokens needed to express one latent thought."""
    if not hidden_size or not vocab_size or vocab_size <= 1:
        return None
    return hidden_size / (math.log(vocab_size) / math.log(log_base))


def configure(hidden_size: Optional[int], vocab_size: Optional[int],
              log_base: float = LOG_BASE) -> None:
    """Record the ratio for this run. Called once, after the model loads."""
    global _RATIO, _INFO
    _RATIO = efficiency_ratio(hidden_size, vocab_size, log_base) if hidden_size else None
    if _RATIO is None:
        _INFO = None
        return
    _INFO = {
        "hidden_size": int(hidden_size),
        "vocab_size": int(vocab_size),
        "log_base": log_base,
        "text_tokens_per_latent_step": round(_RATIO, 4),
        "source": "Theorem 3.1, arXiv:2511.20639 (lower bound)",
    }


def ratio() -> Optional[float]:
    """Text tokens per latent thought, or None if not configured."""
    return _RATIO


def info() -> Optional[dict]:
    """The full record for manifest.json, or None if not configured."""
    return _INFO


def text_equivalent(latent_steps: int) -> Optional[float]:
    """Lower bound on text tokens needed to express `latent_steps` thoughts."""
    if _RATIO is None or not latent_steps:
        return None
    return latent_steps * _RATIO


def compact(n: float) -> str:
    """Short human-readable count: 934, 4.5k, 23.8k, 1.2M."""
    n = float(n)
    if n < 1000:
        return f"{n:.0f}"
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"
