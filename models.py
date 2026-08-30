"""Model wrapper: chat rendering, batched append, latent rollout, decoding.
Model wrapper: chat rendering, batched append, latent rollout, decoding.

Derived from Gen-Verse/LatentMAS `models.py` (Apache-2.0). The differences are
the hybrid cache (see hybrid.py), padding-aware appends over a live recurrent
state, and the removal of the vLLM branch.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import backends
import debug
import hybrid
import metrics
import theory
from hybrid import Workspace

# The recurrent state of a DeltaNet layer is not per-token, so nothing about
# it degrades gracefully on a transformers that predates the unified Cache
# API. Fail here with a pointer rather than deep inside a forward.
try:
    import transformers

    _TF_VERSION = tuple(int(p) for p in transformers.__version__.split(".")[:2])
except Exception:  # pragma: no cover
    _TF_VERSION = (0, 0)
if _TF_VERSION < (5, 2):
    raise ImportError(
        f"transformers {transformers.__version__} is too old for Qwen3.5 "
        "(model type `qwen3_5` landed in 5.2). Install the pin in "
        "requirements.txt: pip install -r requirements.txt"
    )


# Packages transformers imports at module scope while resolving a model class,
# each behind an availability check that only asks whether the package can be
# *located* (`find_spec`), never whether it imports. A companion built against
# a different torch therefore passes the check and then dies with an ABI error
# deep in an import chain that has nothing to do with the model being loaded.
_TORCH_COMPANIONS = ("torchvision", "torchaudio", "torchcodec", "torchao")


def _diagnose_load_failure(exc: BaseException) -> Optional[str]:
    """Turn a known environment breakage into an actionable message.
    Turn a known environment breakage into an actionable message.

    Only called once a load has already failed, so it costs nothing normally.
    """
    chain, seen, cur = [], set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    joined = " | ".join(chain)

    culprits = [p for p in _TORCH_COMPANIONS if p in joined]
    if not culprits:
        return None
    named = ", ".join(culprits)
    try:
        import torch as _t

        have = f"torch {_t.__version__}"
    except Exception:
        have = "your torch"
    return (
        f"The model failed to load inside {named}, not inside Qwen3.5.\n\n"
        f"transformers imports {named} while resolving any model class, and "
        "gates that on `find_spec` - which only checks the package can be "
        "*located*, never that it imports. A build compiled against a "
        f"different torch than {have} passes the gate and then fails here "
        "with an ABI error (undefined symbol / missing operator).\n\n"
        f"Nothing in this repo uses {named}. Remove it:\n"
        f"  pip uninstall -y {' '.join(culprits)}\n\n"
        "That is version-independent and always correct for a text-only run. "
        "If another tool on this machine needs the package, reinstall the "
        "build matching your torch instead, with --no-deps so pip cannot swap "
        "torch out from under you.\n\n"
        "Companions go stale in groups, so find the rest in one pass:\n"
        "  python -m models --check-env\n\n"
        f"Original failure: {chain[-1] if chain else exc}"
    )


def check_environment() -> int:
    """Report which torch companion packages are installed but unimportable.
    Report which torch companion packages are installed but unimportable.

    `python -m models --check-env`. Exists because these break in groups after
    a torch upgrade, and each one only surfaces when the previous is removed.
    """
    import importlib
    import importlib.util

    try:
        import torch as _t

        print(f"torch {_t.__version__}")
    except Exception as exc:
        print(f"torch is not importable: {type(exc).__name__}: {exc}")
        return 1
    broken = []
    for pkg in _TORCH_COMPANIONS:
        if importlib.util.find_spec(pkg) is None:
            print(f"  absent   {pkg}")
            continue
        try:
            importlib.import_module(pkg)
            print(f"  ok       {pkg}")
        except Exception as exc:
            broken.append(pkg)
            print(f"  BROKEN   {pkg}: {type(exc).__name__}: {str(exc)[:100]}")
    if broken:
        print(f"\nThese were built against a different torch. Nothing here uses "
              f"them:\n  pip uninstall -y {' '.join(broken)}")
        return 1
    print("\nAll torch companion packages import cleanly.")
    return 0


# Sentinel for render_chat: "use the run-wide --chat_thinking setting". Distinct
# from None, which means "pass no enable_thinking kwarg at all" and is what the
# tail-discovery probe needs in order to read the template's own default.
_INHERIT = object()


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def resolve_dtype(args=None) -> torch.dtype:
    """The dtype to load weights in.
    The dtype to load weights in.

    Upstream hardcodes bfloat16 whenever CUDA is present, which is wrong on
    pre-Ampere cards: a T4 (Turing) has no native bf16, so those runs fall back
    to emulated paths that are far slower and in places unsupported. `auto`
    asks the device instead of assuming, and is byte-identical to the old
    behavior on any GPU that really does support bf16.
    """
    choice = (getattr(args, "dtype", None) or "auto") if args is not None else "auto"
    explicit = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if choice in explicit:
        return explicit[choice]
    if not torch.cuda.is_available():
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    debug.dbg("dtype: this GPU has no native bfloat16, loading in float16")
    return torch.float16


class ModelWrapper:
    """Loads Qwen3.5 and exposes the primitives the agent chain runs on."""

    def __init__(self, model_name: str, device: torch.device, args=None) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.args = args
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.append_mode = getattr(args, "mamba_append", "auto") if args else "auto"
        self.chat_thinking = getattr(args, "chat_thinking", "auto") if args else "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)

        # A remote backend answers the text-only controls, so the weights are
        # never needed. Skipping the load is not an optimisation here: the
        # point of --backend is that no local accelerator is available.
        self.backend = backends.build_backend(args)
        if self.backend is not None:
            # No local accelerator is in play; --device would otherwise send
            # the tokenised prompts to a GPU that is the reason for using a
            # remote backend in the first place.
            self.device = torch.device("cpu")
            self.model = None
            self.config = None
            self.linear_idx = []
            self.append_carries_state = None
            self.state_probe = None
            self.attn_implementation = None
            self._configure_theory_safe()
            debug.dbg(f"backend: {self.backend.describe()}")
            return

        dtype = resolve_dtype(args)
        debug.dbg(f"dtype: loading {model_name} in {dtype}")
        with torch.no_grad(), debug.phase(f"load model {model_name} (HF, {dtype})"):
            # Qwen3.5 checkpoints declare `Qwen3_5ForConditionalGeneration` and
            # nest the language model under `text_config`. AutoModelForCausalLM
            # maps `qwen3_5` to the text-only `Qwen3_5ForCausalLM`, which loads
            # those same weights and leaves the vision tower behind, verified
            # against a round-tripped checkpoint in verify_hybrid.py.
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            except Exception as exc:
                hint = _diagnose_load_failure(exc)
                if hint is None:
                    raise
                raise RuntimeError(hint) from exc
        with debug.phase(f"move model to {self.device}"):
            self.model.to(self.device)
        self.model.eval()
        self._finish_setup()

    @classmethod
    def from_loaded(cls, model, tokenizer, args=None, device=None) -> "ModelWrapper":
        """Wrap an already-constructed model instead of loading a checkpoint.
        Wrap an already-constructed model instead of loading a checkpoint.

        `verify_hybrid.py` uses this to run the shipped append / rollout /
        decode paths against a tiny Qwen3.5 built in memory, so the checks
        exercise this file rather than a re-implementation of it.
        """
        self = cls.__new__(cls)
        self.model_name = getattr(model.config, "name_or_path", "<in-memory>")
        self.device = torch.device(device or next(model.parameters()).device)
        self.args = args
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices = {}
        self.append_mode = getattr(args, "mamba_append", "auto") if args else "auto"
        self.chat_thinking = getattr(args, "chat_thinking", "auto") if args else "auto"
        self.tokenizer = tokenizer
        self.backend = None
        self.model = model.eval()
        self._finish_setup()
        return self

    def _finish_setup(self) -> None:
        """Architecture checks, the append probe, and reporting config."""
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        self.config = hybrid.text_config(self.model.config)
        if not hybrid.is_hybrid(self.model.config):
            raise ValueError(
                f"{self.model_name} has no linear-attention layers, so there is no "
                "mamba state to carry and this fork has nothing to add. It is "
                "for Qwen3.5 (hybrid attention + Gated DeltaNet); for a pure "
                "attention model use the original LatentMAS."
            )
        self.linear_idx = hybrid.linear_layer_indices(self.model.config)
        debug.dbg(f"model: {hybrid.architecture_summary(self.model.config)}")

        self.state_probe = hybrid.probe_state_continuation(self.model)
        self.append_carries_state = bool(self.state_probe.get("carries"))
        if not self.append_carries_state and self.append_mode in ("auto", "chunk", "split"):
            # Every mode but stepwise assumes a multi-token append continues the
            # recurrent state. If the model says otherwise, the only honest
            # option is one token at a time (a 1-token input always takes the
            # state-consuming path).
            if self.append_mode == "auto":
                debug.dbg("hybrid: appends do NOT carry state on this build, "
                          "falling back to --mamba_append stepwise")
                self.append_mode = "stepwise"
            else:
                raise ValueError(
                    f"--mamba_append {self.append_mode} needs multi-token appends "
                    "to continue the DeltaNet state, and the startup probe says "
                    "this transformers/model does not. Use --mamba_append stepwise."
                )
        debug.dbg(f"hybrid: append mode = {self.append_mode}")
        self.attn_implementation = self._attn_implementation()
        debug.dbg(f"model: attn_implementation = {self.attn_implementation}, "
                  f"dtype = {self.dtype}")
        self._configure_theory()

    def _attn_implementation(self) -> Optional[str]:
        """Which attention kernel transformers selected, for the record.
        """
        for obj in (getattr(self.model, "config", None), self.config):
            impl = getattr(obj, "_attn_implementation", None) or getattr(
                obj, "attn_implementation", None)
            if impl:
                return str(impl)
        return None

    @property
    def dtype(self):
        if self.model is None:
            return None
        return next(self.model.parameters()).dtype

    # ------------------------------------------------------------------
    # Chat rendering / tokenization
    # ------------------------------------------------------------------

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True,
                    enable_thinking=_INHERIT) -> str:
        """Render a chat prompt.
        """
        if enable_thinking is _INHERIT:
            # Run-wide default. This is what makes --chat_thinking reach the
            # baseline and text_mas methods too: they call render_chat without
            # an opinion, and on a thinking-by-default template that otherwise
            # leaves them decoding unbounded reasoning while latent_mas's judger
            # does not -- which would make an Acc/Token comparison across the
            # three methods meaningless.
            enable_thinking = {"auto": None, "on": True, "off": False}[self.chat_thinking]
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt, **kwargs
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def tokenize_batch(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-padded ids + mask.
        Left-padded ids + mask.

        Left padding is required here for two reasons:
        the rollout reads the hidden state at each row's last position (a right
        pad would hand it a pad's state), and the padded-append machinery in
        hybrid.py assumes the pads lead so that after `width - min_len` columns
        every row is on real tokens.
        """
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        return (encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device))

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts = [
            self.render_chat(messages, add_generation_prompt=add_generation_prompt)
            for messages in batch_messages
        ]
        input_ids, attention_mask = self.tokenize_batch(prompts)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt",
        )["input_ids"].to(self.device)

    # ------------------------------------------------------------------
    # Latent realignment (unchanged in behavior from upstream)
    # ------------------------------------------------------------------

    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        """(realignment matrix, target norm) for turning h into an input embedding.
        """
        input_embeds = model.get_input_embeddings()
        output_embeds = model.get_output_embeddings() or getattr(model, "lm_head", None)
        if input_embeds is None or not hasattr(input_embeds, "weight"):
            raise RuntimeError("Cannot build latent realignment: no input embeddings.")

        input_weight = input_embeds.weight
        block = 16384

        # target_norm: mean over per-row L2 norms of the input embedding table.
        # Row norms are independent, so blocking is bit-identical here.
        norms = torch.cat([
            input_weight[i:i + block].detach().to(device=device, dtype=torch.float32).norm(dim=1)
            for i in range(0, input_weight.shape[0], block)
        ])
        target_norm = norms.mean()

        hidden = input_weight.shape[1]
        if not getattr(args, "latent_space_realign", False):
            return torch.eye(hidden, device=device, dtype=torch.float32), target_norm

        if output_embeds is None or not hasattr(output_embeds, "weight"):
            raise RuntimeError("Cannot build latent realignment: no output embeddings.")
        output_weight = output_embeds.weight
        gram = torch.zeros(hidden, hidden, device=device, dtype=torch.float32)
        rhs = torch.zeros(hidden, hidden, device=device, dtype=torch.float32)
        for i in range(0, output_weight.shape[0], block):
            out_block = output_weight[i:i + block].detach().to(device=device, dtype=torch.float32)
            in_block = input_weight[i:i + block].detach().to(device=device, dtype=torch.float32)
            gram += out_block.T @ out_block
            rhs += out_block.T @ in_block
        gram += 1e-5 * torch.eye(hidden, device=device, dtype=torch.float32)
        return torch.linalg.solve(gram, rhs), target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)
        if info is None:
            with debug.phase("build latent realignment matrix"):
                matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)
        if isinstance(target_norm, torch.Tensor):
            target_norm = target_norm.to(device=target_device, dtype=matrix.dtype)
        else:
            target_norm = torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)
        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor) -> torch.Tensor:
        """h -> the next input embedding. Upstream's rollout step, unchanged."""
        matrix, target_norm = self._ensure_latent_realign_matrix(
            self.model, hidden.device, self.args
        )
        aligned = hidden.to(torch.float32)
        if self.latent_space_realign:
            aligned = torch.matmul(aligned, matrix)
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward(self, **kwargs):
        """A forward that never materialises the full logits tensor.
        A forward that never materialises the full logits tensor.

        Appends and latent rollouts never read logits, but the lm_head still
        runs over every position by default. Qwen3.5's vocabulary is 248,320,
        so a batch of 8 x 400 tokens is ~1.6 GB of logits produced and thrown
        away per forward. `logits_to_keep=1` drops that to one row.
        """
        kwargs.setdefault("use_cache", True)
        kwargs.setdefault("return_dict", True)
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("logits_to_keep", 1)
        return self.model(**kwargs)

    def _last_hidden(self, outputs) -> torch.Tensor:
        return outputs.hidden_states[-1][:, -1, :]

    # ------------------------------------------------------------------
    # Appending a prompt onto the shared workspace
    # ------------------------------------------------------------------

    @torch.no_grad()
    def append(
        self,
        input_ids: torch.Tensor,
        chunk_mask: torch.Tensor,
        workspace: Optional[Workspace] = None,
    ) -> Tuple[Workspace, torch.Tensor, Dict]:
        """Feed a prompt chunk into the workspace, both memory channels intact.
        Feed a prompt chunk into the workspace, both memory channels intact.

        Returns (workspace, last hidden state per row, stats). Which route the
        chunk takes is decided by `hybrid.plan_append`; see hybrid.py for why a
        padded chunk over a live recurrent state cannot go through in one pass.
        """
        ws = workspace or Workspace()
        batch, width = input_ids.shape
        chunk_mask = chunk_mask.to(input_ids.device)
        plan = hybrid.plan_append(chunk_mask, hybrid.carries_state(ws.cache), self.append_mode)
        if plan.stepwise_cols:
            debug.dbg(f"append: {plan.reason}, {plan.stepwise_cols} column(s) stepwise "
                      f"with per-row state restore, then {plan.chunk_cols} in one forward")

        metrics.cuda_sync(self.device)
        _t0 = time.perf_counter()
        next_pos = ws.next_pos(batch, input_ids.device)
        positions = hybrid.chunk_positions(next_pos, chunk_mask)
        mask = ws.mask
        last_hidden = None

        meter = debug.StepMeter("append", plan.stepwise_cols, batch_size=batch)
        for t in range(plan.stepwise_cols):
            col_mask = chunk_mask[:, t:t + 1]
            pad_rows = (col_mask.squeeze(1) == 0).nonzero(as_tuple=True)[0]
            snaps = (hybrid.snapshot_state(ws.cache, self.linear_idx)
                     if pad_rows.numel() and ws.cache is not None else None)
            mask = col_mask if mask is None else torch.cat([mask, col_mask], dim=-1)
            outputs = self._forward(
                input_ids=input_ids[:, t:t + 1],
                attention_mask=mask,
                position_ids=positions[:, t:t + 1],
                past_key_values=ws.cache,
            )
            ws.cache = outputs.past_key_values
            if pad_rows.numel():
                if snaps is None:
                    # nothing existed to snapshot; a faithful restore is zeros
                    snaps = [(i, None, None) for i in self.linear_idx]
                hybrid.restore_state_rows(ws.cache, snaps, pad_rows)
            last_hidden = self._last_hidden(outputs)
            meter.step()
        meter.end()

        if plan.chunk_cols:
            start = plan.stepwise_cols
            rest_mask = chunk_mask[:, start:]
            mask = rest_mask if mask is None else torch.cat([mask, rest_mask], dim=-1)
            outputs = self._forward(
                input_ids=input_ids[:, start:],
                attention_mask=mask,
                position_ids=positions[:, start:],
                past_key_values=ws.cache,
            )
            ws.cache = outputs.past_key_values
            last_hidden = self._last_hidden(outputs)

        ws.mask = mask
        metrics.cuda_sync(self.device)
        stats = {
            "append_forwards": plan.forwards,
            "append_reason": plan.reason,
            "append_sec": round(time.perf_counter() - _t0, 4),
            # Positions actually pushed through the model on this append,
            # summed over the batch. The per-forward count alone understates a
            # stepwise route and overstates a one-shot one; this is the unit
            # the cost of --mamba_append is really denominated in.
            "append_positions": int(chunk_mask.sum().item()),
        }
        return ws, last_hidden, stats

    # ------------------------------------------------------------------
    # The latent rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def latent_rollout(
        self,
        workspace: Workspace,
        last_hidden: torch.Tensor,
        latent_steps: int,
        label: str = "latent",
    ) -> Tuple[Workspace, torch.Tensor, List[torch.Tensor]]:
        """Upstream's rollout: realign h into embedding space and feed it back.
        """
        batch = last_hidden.shape[0]
        base_pos = workspace.next_pos(batch, last_hidden.device)
        # The vectors are returned as well as absorbed. `latent_only` visibility
        # replays them into a workspace that never saw this agent's prompt, so
        # it needs the thoughts themselves rather than only their effect here.
        produced: List[torch.Tensor] = []
        meter = debug.StepMeter(label, latent_steps, batch_size=batch)
        for step in range(latent_steps):
            vec = self._apply_latent_realignment(last_hidden)
            produced.append(vec.detach())
            workspace.mask = torch.cat(
                [workspace.mask,
                 torch.ones(batch, 1, dtype=workspace.mask.dtype, device=workspace.mask.device)],
                dim=-1,
            )
            outputs = self._forward(
                inputs_embeds=vec.unsqueeze(1),
                attention_mask=workspace.mask,
                position_ids=base_pos + step,
                past_key_values=workspace.cache,
            )
            workspace.cache = outputs.past_key_values
            last_hidden = self._last_hidden(outputs)
            meter.step()
        meter.end()
        return workspace, last_hidden, produced

    @torch.no_grad()
    def append_embeds(
        self,
        workspace: Workspace,
        vectors: List[torch.Tensor],
        *,
        prefix_ids: Optional[torch.Tensor] = None,
        suffix_ids: Optional[torch.Tensor] = None,
    ) -> Workspace:
        """Write already-computed latent vectors into a workspace, prompt-free.
        """
        batch = vectors[0].shape[0] if vectors else (
            prefix_ids.shape[0] if prefix_ids is not None else 1)
        device = self.device

        def _tokens(ids):
            mask = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.long, device=ids.device)
            return mask

        if prefix_ids is not None and prefix_ids.shape[1]:
            workspace, _, _ = self.append(prefix_ids, _tokens(prefix_ids), workspace)
        if workspace.mask is None:
            workspace.mask = torch.zeros(batch, 0, dtype=torch.long, device=device)

        base_pos = workspace.next_pos(batch, device)
        meter = debug.StepMeter("republish", len(vectors), batch_size=batch)
        for step, vec in enumerate(vectors):
            workspace.mask = torch.cat(
                [workspace.mask,
                 torch.ones(batch, 1, dtype=workspace.mask.dtype, device=workspace.mask.device)],
                dim=-1,
            )
            outputs = self._forward(
                inputs_embeds=vec.unsqueeze(1).to(self.dtype),
                attention_mask=workspace.mask,
                position_ids=base_pos + step,
                past_key_values=workspace.cache,
            )
            workspace.cache = outputs.past_key_values
            meter.step()
        meter.end()

        if suffix_ids is not None and suffix_ids.shape[1]:
            workspace, _, _ = self.append(suffix_ids, _tokens(suffix_ids), workspace)
        return workspace

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        latent_steps: int,
        workspace: Optional[Workspace] = None,
    ) -> Tuple[Workspace, Dict]:
        """One agent's turn: append its prompt, then think `latent_steps` times.
        One agent's turn: append its prompt, then think `latent_steps` times.

        Same contract as upstream's `generate_latent_batch`, except that the
        thing handed on is a `Workspace` (cache + mask + positions) rather than
        a bare KV tuple, because on Qwen3.5 the cache alone no longer describes
        the state of the conversation.
        """
        workspace, last_hidden, stats = self.append(input_ids, attention_mask, workspace)

        metrics.cuda_sync(self.device)
        _t0 = time.perf_counter()
        latents: List[torch.Tensor] = []
        if latent_steps > 0:
            workspace, _, latents = self.latent_rollout(workspace, last_hidden, latent_steps)
        metrics.cuda_sync(self.device)
        stats["latent_sec"] = round(time.perf_counter() - _t0, 4)

        # The size of each half of the workspace as this agent leaves it. This
        # is the measurement the `state_only` arm exists to make: `kv_bytes`
        # grows with every token and latent thought the agents have spent,
        # `state_bytes` does not move at all. A trace that shows the second
        # flat while accuracy holds is the O(1) inter-agent channel result;
        # without the bytes recorded it is only an assertion.
        stats["kv_bytes"] = hybrid.kv_bytes(workspace.cache)
        stats["state_bytes"] = hybrid.state_bytes(workspace.cache)
        stats["latents"] = latents
        return workspace, stats

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_text_with_workspace(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        workspace: Optional[Workspace],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[List[str], List[int], Dict]:
        """Decode an answer with the shared workspace as context.
        """
        stats: Dict = {"append_sec": 0.0, "append_forwards": 0,
                       "append_positions": 0, "append_reason": None}
        if workspace is not None and workspace.cache is not None and input_ids.shape[1] > 1:
            workspace, _, append_stats = self.append(
                input_ids[:, :-1], attention_mask[:, :-1], workspace)
            stats.update(append_stats)
            gen_ids = input_ids[:, -1:]
            full_mask = torch.cat([workspace.mask, attention_mask[:, -1:]], dim=-1)
            past = workspace.cache
        else:
            gen_ids = input_ids
            full_mask = attention_mask
            past = None

        meter = (debug.TokMeter("decode", batch_size=input_ids.shape[0], cap=max_new_tokens)
                 if debug.enabled() else None)
        metrics.cuda_sync(self.device)
        _t0 = time.perf_counter()
        outputs = self.model.generate(
            input_ids=gen_ids,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past,
            streamer=meter,
            **self._sampling_kwargs(temperature, top_p),
        )
        metrics.cuda_sync(self.device)
        stats["decode_sec"] = round(time.perf_counter() - _t0, 4)
        if meter is not None:
            meter.end()
        texts, counts, truncated = self._decode_sequences(
            outputs.sequences[:, gen_ids.shape[1]:])
        stats["truncated"] = truncated
        stats["max_new_tokens"] = max_new_tokens
        return texts, counts, stats

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        prompts: Optional[List[str]] = None,
        messages: Optional[List[List[Dict]]] = None,
    ) -> Tuple[List[str], Dict, List[int]]:
        """Plain batched generation with no shared workspace (baseline/text_mas).
        Plain batched generation with no shared workspace (baseline/text_mas).

        `prompts` / `messages` are ignored on the local path and required on a
        remote one: an OpenAI-compatible server is sent the rendered prompt so
        the chat template stays under our control, while Bedrock's Converse API
        takes the structured messages and applies its own.
        """
        if self.backend is not None:
            if prompts is None:
                raise ValueError(
                    "a remote backend needs the rendered prompts; call "
                    "generate_text_batch(..., prompts=prompts, messages=batch_messages)")
            texts, counts, _, stats = self.backend.generate(
                prompts, messages or [[] for _ in prompts],
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            return texts, stats, counts

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        meter = (debug.TokMeter("decode", batch_size=input_ids.shape[0], cap=max_new_tokens)
                 if debug.enabled() else None)
        metrics.cuda_sync(self.device)
        _t0 = time.perf_counter()
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            streamer=meter,
            **self._sampling_kwargs(temperature, top_p),
        )
        metrics.cuda_sync(self.device)
        decode_sec = round(time.perf_counter() - _t0, 4)
        if meter is not None:
            meter.end()
        texts, counts, truncated = self._decode_sequences(
            outputs.sequences[:, input_ids.shape[1]:])
        stats = {"decode_sec": decode_sec, "truncated": truncated,
                 "max_new_tokens": max_new_tokens, "append_sec": 0.0,
                 "append_forwards": 0, "append_positions": 0}
        return texts, stats, counts

    def _sampling_kwargs(self, temperature: float, top_p: float) -> Dict:
        """Sampling config for `generate`, with a real greedy path.
        """
        if temperature is not None and temperature > 0:
            return {"do_sample": True, "temperature": temperature, "top_p": top_p}
        return {"do_sample": False}

    def _stop_token_ids(self) -> List[int]:
        """Every id that ends a generation, from the tokenizer and gen config."""
        ids = set()
        sources = [getattr(self.tokenizer, "eos_token_id", None)]
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None:
            sources.append(getattr(gen_cfg, "eos_token_id", None))
        for src in sources:
            if src is None:
                continue
            if isinstance(src, (list, tuple, set)):
                ids.update(int(i) for i in src if i is not None)
            else:
                ids.add(int(src))
        return sorted(ids)

    def _decode_sequences(
        self, generated: torch.Tensor
    ) -> Tuple[List[str], List[int], List[Optional[bool]]]:
        """Texts, exact generated-token counts, and per-row truncation flags.
        """
        pad_id = self.tokenizer.pad_token_id
        stop_ids = self._stop_token_ids()
        stop_tensor = (torch.tensor(stop_ids, device=generated.device)
                       if stop_ids else None)
        texts, counts, truncated = [], [], []
        for row in generated:
            if pad_id is None:
                counts.append(int(row.shape[-1]))
            else:
                counts.append(int((row != pad_id).sum().item()))
            if stop_tensor is None:
                truncated.append(None)
            else:
                truncated.append(not bool(torch.isin(row, stop_tensor).any().item()))
            texts.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        return texts, counts, truncated

    # ------------------------------------------------------------------

    def _configure_theory_safe(self) -> None:
        """Theorem 3.1 reporting needs d_h, which a remote backend never sees."""
        try:
            theory.configure(None, len(self.tokenizer) if self.tokenizer else None)
        except Exception:
            pass

    def _configure_theory(self) -> None:
        """Record d_h and |V| for the Theorem 3.1 reporting metrics."""
        try:
            hidden = int(getattr(self.config, "hidden_size"))
            vocab = len(self.tokenizer) if self.tokenizer is not None else None
            theory.configure(hidden, vocab)
            ratio = theory.ratio()
            if ratio is not None:
                debug.dbg(f"theorem 3.1: d_h={hidden} |V|={vocab}, "
                          f"1 latent thought >= {ratio:.1f} text tokens")
        except Exception:  # reporting only; never break a run over it
            pass


if __name__ == "__main__":
    import sys

    if "--check-env" in sys.argv:
        sys.exit(check_environment())
    print("usage: python -m models --check-env")
    sys.exit(2)
