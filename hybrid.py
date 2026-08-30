"""Qwen3.5 hybrid cache: attention KV plus the Gated DeltaNet state.

    conv_states       [B, 2*key_dim + value_dim, linear_conv_kernel_dim]
    recurrent_states  [B, linear_num_value_heads, key_head_dim, value_head_dim]
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

import debug

try:  # transformers >= 5.x, the only API Qwen3.5 ships on
    from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin
except ImportError as exc:  # pragma: no cover - guarded again in models.py
    raise ImportError(
        "This fork needs transformers >= 5.2 for its unified Cache API "
        "(Qwen3.5 does not exist on older releases). See README.md."
    ) from exc


# ---------------------------------------------------------------------------
# Config / architecture introspection
# ---------------------------------------------------------------------------

def text_config(config):
    """The text-model config, unwrapping Qwen3.5's multimodal `Qwen3_5Config`.
    The text-model config, unwrapping Qwen3.5's multimodal `Qwen3_5Config`.

    Qwen3.5 checkpoints, including the text-only sizes, carry
    `model_type: qwen3_5` with the language model nested under `text_config`.
    `get_text_config()` is the supported way to reach it and works whether the
    config handed in is already the text one or the wrapper.
    """
    getter = getattr(config, "get_text_config", None)
    if getter is not None:
        try:
            return getter(decoder=True)
        except TypeError:
            return getter()
    return getattr(config, "text_config", None) or config


def layer_types(config) -> List[str]:
    return list(getattr(text_config(config), "layer_types", None) or [])


def is_hybrid(config) -> bool:
    return "linear_attention" in layer_types(config)


def linear_layer_indices(config) -> List[int]:
    return [i for i, t in enumerate(layer_types(config)) if t == "linear_attention"]


def attention_layer_indices(config) -> List[int]:
    return [i for i, t in enumerate(layer_types(config)) if t != "linear_attention"]


def architecture_summary(config) -> str:
    types = layer_types(config)
    cfg = text_config(config)
    lin = len(linear_layer_indices(config))
    state = (f"[B, {getattr(cfg, 'linear_num_value_heads', '?')}, "
             f"{getattr(cfg, 'linear_key_head_dim', '?')}, "
             f"{getattr(cfg, 'linear_value_head_dim', '?')}]")
    return (f"{lin}/{len(types)} layers are Gated DeltaNet "
            f"(recurrent state {state} per layer, conv kernel "
            f"{getattr(cfg, 'linear_conv_kernel_dim', '?')})")


# ---------------------------------------------------------------------------
# Cache accessors
# ---------------------------------------------------------------------------

def kv_length(cache) -> int:
    """Positions held by the attention layers (0 for None/empty).
    Positions held by the attention layers (0 for None/empty).

    `Cache.get_seq_length()` already skips past leading DeltaNet layers to the
    first attention layer, so this is the token length of the workspace.
    """
    if cache is None:
        return 0
    return int(cache.get_seq_length())


def linear_layers(cache) -> List[Tuple[int, object]]:
    if cache is None:
        return []
    return [
        (i, layer)
        for i, layer in enumerate(getattr(cache, "layers", []) or [])
        if isinstance(layer, LinearAttentionCacheLayerMixin)
    ]


def attention_layers(cache) -> List[Tuple[int, object]]:
    if cache is None:
        return []
    return [
        (i, layer)
        for i, layer in enumerate(getattr(cache, "layers", []) or [])
        if isinstance(layer, CacheLayerMixin)
    ]


def get_state(cache, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(conv_state, recurrent_state) of one DeltaNet layer, or (None, None).
    (conv_state, recurrent_state) of one DeltaNet layer, or (None, None).

    Both live in `{state_idx: tensor}` dicts; state 0 is the one the model
    reads (Qwen3.5 declares a single state per layer).
    """
    if cache is None:
        return None, None
    layer = cache.layers[layer_idx]
    return layer.conv_states.get(0), layer.recurrent_states.get(0)


def carries_state(cache) -> bool:
    """Whether any DeltaNet layer is holding live state.
    Whether any DeltaNet layer is holding live state.

    Reads the layers' own `has_previous_state` flags rather than testing the
    tensors: after `reset()` the buffers still exist but are zeroed and the
    flags are down, and it is the flags the model gates on.
    """
    for _, layer in linear_layers(cache):
        flags = layer.has_previous_state
        if any(bool(v) for v in (flags.values() if isinstance(flags, dict) else [flags])):
            return True
    return False


def state_norm(cache) -> Optional[float]:
    """RMS over every recurrent state, a one-number fingerprint of the mamba
    RMS over every recurrent state, a one-number fingerprint of the mamba
    memory, recorded per agent so a trace shows the state really crossing the
    boundary (and really resetting where a transfer mode says it must).
    None means no state is being carried at all."""
    if not carries_state(cache):
        return None
    total, count = 0.0, 0
    for idx, _ in linear_layers(cache):
        _, rec = get_state(cache, idx)
        if rec is None:
            continue
        total += float(rec.detach().float().pow(2).sum())
        count += rec.numel()
    if count == 0:
        return None
    return round((total / count) ** 0.5, 6)


def state_bytes(cache) -> int:
    """Bytes of DeltaNet state that are actually live. Constant in the number
    of agents, which is what the `state_only` transfer arm measures.
    """
    if not carries_state(cache):
        return 0
    total = 0
    for idx, _ in linear_layers(cache):
        conv, rec = get_state(cache, idx)
        for t in (conv, rec):
            if t is not None:
                total += t.numel() * t.element_size()
    return total


def channel_dtypes(cache) -> dict:
    """The dtype each cache channel is actually stored in.

    Worth recording separately: on Qwen3.5 the weights, the attention KV and
    the conv window are bf16 while the recurrent state is kept in fp32, so a
    single "dtype: bfloat16" line in a manifest would misdescribe three
    quarters of the model's memory.
    """
    out = {"kv": None, "recurrent": None, "conv": None}
    for _, layer in attention_layers(cache):
        if getattr(layer, "is_initialized", False) and layer.keys is not None:
            out["kv"] = str(layer.keys.dtype)
            break
    for idx, _layer in linear_layers(cache):
        conv, rec = get_state(cache, idx)
        if conv is not None and out["conv"] is None:
            out["conv"] = str(conv.dtype)
        if rec is not None and out["recurrent"] is None:
            out["recurrent"] = str(rec.dtype)
        if out["conv"] and out["recurrent"]:
            break
    return out


def kv_bytes(cache) -> int:
    total = 0
    for _, layer in attention_layers(cache):
        if not getattr(layer, "is_initialized", False):
            continue
        for t in (layer.keys, layer.values):
            if t is not None:
                total += t.numel() * t.element_size()
    return total


# ---------------------------------------------------------------------------
# State snapshot / restore (the padded-append machinery)
# ---------------------------------------------------------------------------

def snapshot_state(cache, layer_indices: Sequence[int]):
    """Clones of each DeltaNet layer's (conv, recurrent) state.
    Clones of each DeltaNet layer's (conv, recurrent) state.

    Cloning is required: `LinearAttentionLayer.update_*_state`
    writes with `copy_` into the existing buffer to keep a static address for
    cudagraphs, so an un-cloned reference would mutate underneath us during
    the very forward we mean to roll back.
    """
    snaps = []
    for i in layer_indices:
        conv, rec = get_state(cache, i)
        snaps.append((i,
                      None if conv is None else conv.clone(),
                      None if rec is None else rec.clone()))
    return snaps


def restore_state_rows(cache, snaps, rows: torch.Tensor) -> None:
    """Put `rows` of every DeltaNet state back to a snapshot.
    Put `rows` of every DeltaNet state back to a snapshot.

    A None snapshot means the layer had no state at snapshot time, whose
    faithful restore is zeros: a zero recurrent state is exactly
    `initial_state=None` to the delta-rule kernels, and a zero conv window is
    exactly the left zero-pad a fresh prefill writes.
    """
    for i, conv_snap, rec_snap in snaps:
        conv, rec = get_state(cache, i)
        if conv is not None:
            conv[rows] = 0 if conv_snap is None else conv_snap[rows].to(conv.dtype)
        if rec is not None:
            rec[rows] = 0 if rec_snap is None else rec_snap[rows].to(rec.dtype)


# ---------------------------------------------------------------------------
# Transfer policy, what survives an agent boundary
# ---------------------------------------------------------------------------

def strip_kv(cache) -> None:
    """Drop the attention half, keep the DeltaNet state.
    Drop the attention half, keep the DeltaNet state.

    Assigns a zero-width slice rather than None: the layer stays
    `is_initialized`, `get_seq_length()` reads `keys.numel()` and reports 0,
    and the next append's `torch.cat` still has the dtype/device/head dims it
    needs. Setting None instead raises on the next forward.
    """
    if cache is None:
        return
    for _, layer in attention_layers(cache):
        if not getattr(layer, "is_initialized", False) or layer.keys is None:
            continue
        layer.keys = layer.keys[..., :0, :]
        layer.values = layer.values[..., :0, :]
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = 0


def truncate_kv(cache, keep: int) -> None:
    """Crop the attention half back to `keep` positions, keeping the DeltaNet state.
    """
    if cache is None:
        return
    for _, layer in attention_layers(cache):
        if not getattr(layer, "is_initialized", False) or layer.keys is None:
            continue
        if layer.keys.shape[-2] <= keep:
            continue
        layer.keys = layer.keys[..., :keep, :]
        layer.values = layer.values[..., :keep, :]
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = keep


def restore_state(cache, snaps) -> None:
    """Put *every* row of every DeltaNet state back to a snapshot.
    Put *every* row of every DeltaNet state back to a snapshot.

    The row-wise form exists for the padded-append path, which rolls back only
    the rows still sitting on padding. Rewinding a whole agent turn rolls back
    all of them.
    """
    if cache is None or not snaps:
        return
    conv, _ = get_state(cache, snaps[0][0])
    rows = torch.arange(conv.shape[0], device=conv.device) if conv is not None else None
    if rows is None:
        return
    restore_state_rows(cache, snaps, rows)


def strip_state(cache) -> None:
    """Drop the DeltaNet half, keep the attention KV.
    Drop the DeltaNet half, keep the attention KV.

    `reset()` is this API's own "forget the state": it zeroes the buffers *and*
    clears `has_previous_state`. Nulling the tensors instead leaves the flags
    up, and the next append then tries to `torch.cat` a None conv state.
    """
    if cache is None:
        return
    for _, layer in linear_layers(cache):
        layer.reset()


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def chunk_positions(next_pos: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
    """Padding-aware absolute RoPE positions for a chunk.

    `next_pos` is [B, 1]: the position the next *real* token of each row takes.
    Real tokens get consecutive positions from there; padding reuses the first
    slot, which is arbitrary but in range (those positions are masked out of
    attention and zeroed out of the DeltaNet input anyway).

    every downstream hidden state, measured at 9.9e-02 on the last hidden
    """
    steps = (chunk_mask.long().cumsum(dim=1) - 1).clamp(min=0)
    return next_pos.to(steps.device) + steps


# ---------------------------------------------------------------------------
# The workspace: cache + honest mask over it + per-row position counter
# ---------------------------------------------------------------------------

@dataclass
class Workspace:
    """One batch's shared latent memory as it passes between agents.
    """

    cache: object = None
    mask: Optional[torch.Tensor] = None

    @property
    def kv_len(self) -> int:
        return kv_length(self.cache)

    def next_pos(self, batch: int, device) -> torch.Tensor:
        """[B, 1] absolute position of each row's next real token."""
        if self.mask is None:
            return torch.zeros(batch, 1, dtype=torch.long, device=device)
        return self.mask.long().sum(dim=1, keepdim=True).to(device)

    def extend_mask(self, chunk_mask: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            return chunk_mask
        return torch.cat([self.mask.to(chunk_mask.device), chunk_mask], dim=-1)

    def state_norm(self) -> Optional[float]:
        return state_norm(self.cache)

    def describe(self) -> str:
        return (f"kv_len={self.kv_len} "
                f"kv={kv_bytes(self.cache) / 2**20:.1f}MiB "
                f"state={state_bytes(self.cache) / 2**20:.1f}MiB "
                f"state_norm={state_norm(self.cache)}")


# ---------------------------------------------------------------------------
# Append planning
# ---------------------------------------------------------------------------

@dataclass
class AppendPlan:
    """How a chunk has to be fed in, given padding and the cache's state.
    How a chunk has to be fed in, given padding and the cache's state.

    `stepwise_cols` is how many leading columns must go one token at a time
    with per-row state restore; the rest goes in a single forward. Whether the
    remainder is contaminated is decided entirely by whether padding remains
    inside it, so `stepwise_cols = width - min_real_length` is the exact
    boundary: past that column every row is on real tokens.
    """

    stepwise_cols: int
    chunk_cols: int
    reason: str

    @property
    def forwards(self) -> int:
        return self.stepwise_cols + (1 if self.chunk_cols else 0)


def plan_append(chunk_mask: torch.Tensor, live_state: bool, mode: str) -> AppendPlan:
    width = chunk_mask.shape[1]
    padded = not bool(chunk_mask.all())

    if mode == "chunk":
        return AppendPlan(0, width, "forced chunk")
    if mode == "stepwise":
        return AppendPlan(width, 0, "forced stepwise")
    if mode not in ("auto", "split"):
        raise ValueError(f"unknown append mode {mode!r}")

    if not padded:
        # Nothing to protect: every row is on a real token at every column.
        return AppendPlan(0, width, "unpadded")
    if mode == "auto" and not live_state:
        # A zero recurrent state decays to zero, and a zero conv window is
        # exactly the left zero-pad a fresh prefill writes, so a padded
        # prefill onto an empty state is already exact (measured 8e-10). Only
        # `auto` takes this shortcut; `split` protects padding unconditionally.
        return AppendPlan(0, width, "padded prefill on a fresh state")

    pad_prefix = int(width - chunk_mask.long().sum(dim=1).min())
    where = "over live state" if live_state else "(forced split)"
    return AppendPlan(pad_prefix, width - pad_prefix,
                      f"padded {where} (pad prefix {pad_prefix})")


# ---------------------------------------------------------------------------
# Startup probe, the append contract, measured rather than assumed
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_state_continuation(model, margin: float = 0.1) -> dict:
    """Measure whether a multi-token append over a live cache continues each
    DeltaNet channel: the recurrent state and the causal-convolution window.

    Three forwards per channel: a one-shot reference over AB, the same split
    as A then B over the live cache, and A then B with the state dropped at the
    boundary. A correct append should land near the reference and far from the
    dropped case.

    The comparison is relative, not absolute. The states are stored in the
    model dtype, so in bf16 a correct append lands about 1% of the state's
    magnitude from the one-shot reference while a genuine reset lands about
    58% away. An absolute tolerance would read the first as a failure and
    demote the run to stepwise appends for nothing.

    Returns per-channel {"carries", "split", "reset", "ratio"} plus a top-level
    "carries" that is True only when both channels continue.
    """
    out = {"recurrent": None, "conv": None, "carries": False, "margin": margin}
    idx = linear_layer_indices(model.config)
    if not idx:
        return out
    try:
        device = next(model.parameters()).device
        cfg = text_config(model.config)
        vocab = int(getattr(cfg, "vocab_size", 1000))
        ids = torch.randint(1, max(2, min(100, vocab - 1)), (1, 8), device=device)

        def states(cache, which):
            got = []
            for i in idx:
                conv, rec = get_state(cache, i)
                t = conv if which == "conv" else rec
                got.append(None if t is None else t.detach().float().clone())
            return got

        def spread(a, b):
            worst = 0.0
            for x, y in zip(a, b):
                if x is None or y is None or x.shape != y.shape:
                    return None
                worst = max(worst, float((x - y).abs().max()))
            return worst

        one_shot = model(input_ids=ids, use_cache=True, return_dict=True).past_key_values

        first = model(input_ids=ids[:, :5], use_cache=True, return_dict=True)
        split_cache = model(input_ids=ids[:, 5:], past_key_values=first.past_key_values,
                            use_cache=True, return_dict=True).past_key_values

        dropped = model(input_ids=ids[:, :5], use_cache=True, return_dict=True)
        strip_state(dropped.past_key_values)
        reset_cache = model(input_ids=ids[:, 5:], past_key_values=dropped.past_key_values,
                            use_cache=True, return_dict=True).past_key_values

        for which in ("recurrent", "conv"):
            ref = states(one_shot, which)
            d_split = spread(ref, states(split_cache, which))
            d_reset = spread(ref, states(reset_cache, which))
            if d_split is None or d_reset is None:
                debug.dbg(f"hybrid: append probe could not read the {which} states")
                continue
            scale = max((float(r.abs().max()) for r in ref if r is not None), default=0.0)
            if d_reset <= max(1e-12, 1e-3 * scale):
                # Dropping the state changed nothing, so the probe cannot tell
                # the two cases apart on this channel.
                debug.dbg(f"hybrid: {which} probe is not discriminating "
                          f"(dropping the state moved it by only {d_reset:.2e})")
                continue
            ratio = d_split / d_reset
            out[which] = {"carries": bool(ratio < margin),
                          "split": float(d_split), "reset": float(d_reset),
                          "ratio": float(ratio), "scale": float(scale)}
            debug.dbg(f"hybrid: {which} continuation, split {d_split:.2e} vs reset "
                      f"{d_reset:.2e} ({ratio:.1%} of a full reset) -> "
                      f"{'carries' if ratio < margin else 'DOES NOT carry'}")

        out["carries"] = bool(out["recurrent"] and out["recurrent"]["carries"]
                              and out["conv"] and out["conv"]["carries"])
        out["dtype"] = str(next(model.parameters()).dtype)
        return out
    except Exception as exc:  # a probe must never be why a run dies
        debug.dbg(f"hybrid: append probe failed ({type(exc).__name__}: {exc}), "
                  f"assuming appends do NOT carry state")
        return out


def probe_append_carries_state(model, margin: float = 0.1) -> bool:
    """True when a multi-token append continues both DeltaNet channels."""
    return bool(probe_state_continuation(model, margin)["carries"])


__all__ = [
    "Workspace", "AppendPlan", "plan_append", "chunk_positions",
    "text_config", "layer_types", "is_hybrid", "linear_layer_indices",
    "attention_layer_indices", "architecture_summary",
    "kv_length", "linear_layers", "attention_layers", "get_state",
    "carries_state", "state_norm", "state_bytes", "kv_bytes",
    "channel_dtypes",
    "snapshot_state", "restore_state_rows", "restore_state", "strip_kv",
    "strip_state", "truncate_kv",
    "probe_append_carries_state", "probe_state_continuation",
]
