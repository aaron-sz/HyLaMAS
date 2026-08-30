"""Enhanced run output (opt-in via --print2).
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from typing import Dict, List, Optional

import metrics
import theory
from debug import enable_windows_ansi

# ANSI codes (used only when stdout is an interactive terminal)
_RESET = "\033[0m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"

MIB = 2 ** 20


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    """True if the working tree has uncommitted changes (commit alone would lie)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return None


def _environment() -> Dict:
    env: Dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    # importlib.metadata reads installed versions without importing the
    # packages (importing vllm alone costs seconds).
    # causal-conv1d and flash-linear-attention decide which Gated DeltaNet
    # kernel path runs, so they change results, not just speed.
    for pkg in ("torch", "transformers", "vllm", "accelerate", "numpy",
                "datasets", "causal-conv1d", "flash-linear-attention",
                "fla-core", "flash-attn", "triton"):
        try:
            env[pkg] = metadata.version(pkg)
        except Exception:
            pass
    try:
        from transformers.utils import is_causal_conv1d_available
        env["causal_conv1d_available"] = bool(is_causal_conv1d_available())
    except Exception:
        pass
    try:
        from transformers.utils import is_flash_linear_attention_available
        env["flash_linear_attention_available"] = bool(
            is_flash_linear_attention_available())
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 2**30, 1
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    # Compute capability, driver, power cap, TF32 flags, the facts that make
    # the same command run at two different speeds on two different cards.
    details = metrics.device_details()
    if details:
        env["device_details"] = details
    return env


def _short(value, n: int = 14) -> str:
    s = "None" if value is None else str(value).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_dur(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _display_path(path: str) -> str:
    # Forward slashes read cleaner and work in every shell, Windows included.
    return path.replace(os.sep, "/")


def _mib(n: Optional[int]) -> Optional[float]:
    return None if n is None else round(n / MIB, 3)


def _chat_fingerprint(model) -> dict:
    """Pin the chat template, because it silently decides how a run behaves.
    """
    tok = getattr(model, "tokenizer", None)
    out = {"chat_thinking": getattr(model, "chat_thinking", None)}
    tpl = getattr(tok, "chat_template", None) if tok is not None else None
    if isinstance(tpl, str):
        out["template_sha256"] = hashlib.sha256(tpl.encode("utf-8")).hexdigest()[:16]
        out["template_chars"] = len(tpl)
    for obj in (getattr(model, "model", None), model):
        cfg = getattr(obj, "config", None)
        rev = getattr(cfg, "_commit_hash", None) if cfg is not None else None
        if rev:
            out["revision"] = rev
            break
    if "revision" not in out:
        # A checkpoint loaded from the hub cache carries its commit in the
        # snapshot path; a local directory has none and stays absent.
        try:
            src_path = getattr(getattr(model, "model", None), "name_or_path", None) \
                or getattr(tok, "name_or_path", "")
            parts = str(src_path).replace("\\", "/").split("/")
            if "snapshots" in parts:
                out["revision"] = parts[parts.index("snapshots") + 1]
        except Exception:
            pass
    probe = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    marker = '<|im_start|>assistant\n'
    for key, flag in (("auto", None), ("on", True), ("off", False)):
        try:
            txt = model.render_chat(probe, add_generation_prompt=True,
                                    enable_thinking=flag)
            cut = txt.rfind(marker)
            out["generation_tail_" + key] = (txt[cut + len(marker):] if cut >= 0
                                             else txt[-40:])
        except Exception:
            out["generation_tail_" + key] = None
    return out


class RunWriter:
    """Writes per-sample and per-batch records + a manifest, and prints progress."""

    def __init__(self, args) -> None:
        # Make unicode marks safe on any console (Windows included).
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

        # Color the ✓/✗ marks only for an interactive terminal (honors NO_COLOR).
        try:
            self._color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        except Exception:
            self._color = False
        if self._color:
            self._color = enable_windows_ansi(-11)  # -11 = STD_OUTPUT_HANDLE

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}_{args.method}_{args.task}"
        self.dir = os.path.join("runs", self.run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.trace_path = os.path.join(self.dir, "trace.jsonl")
        self.batch_path = os.path.join(self.dir, "batches.jsonl")
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        self.telemetry_path = os.path.join(self.dir, "telemetry.jsonl")

        self.args = args
        self.started = datetime.now(timezone.utc).isoformat()
        # Resolved once. `write_manifest` runs after every batch, and shelling
        # out to git twice per batch is both wasted time inside the timed run
        # and, in a non-repo checkout, two guaranteed-failing subprocesses.
        self.git = {"commit": _git_commit(), "dirty": _git_dirty()}
        self.env = _environment()
        self.model_info: Dict = {}
        self._t0 = time.time()
        self._n = 0
        self._correct = 0
        self._batch_index = -1
        self._batch_size = 0
        self._prompt_tokens = 0
        self._generated_tokens = 0
        self._latent_steps = 0
        self._text_equivalent = 0.0

        # Truncation: a sample whose final answer ran into --max_new_tokens
        # was cut off, which inflates its token count and (usually) loses its
        # accuracy. Counted rather than averaged away.
        self._truncated_final = 0
        self._truncated_any = 0

        # Forward-pass economics of the hybrid append (see hybrid.py).
        self._append_forwards = 0
        self._append_positions = 0

        # Batch-level phase wall time. These are recorded once per batch, not
        # once per sample: the underlying figures are measured over the whole
        # batch and written identically to every row in it, so summing them
        # per sample would multiply by the batch size.
        self._append_sec = 0.0
        self._latent_sec = 0.0
        self._decode_sec = 0.0
        self._phase_counted = False

        # Inter-agent channel size, in bytes, at the widest agent boundary of
        # each sample. `transfer_*` is post-`--state_transfer`: what actually
        # crossed. The `state_only` claim lives in these numbers.
        self._kv_bytes_max = 0
        self._state_bytes_max = 0
        self._transfer_kv_max = 0
        self._transfer_state_max = 0

        self._batch_secs: List[float] = []
        self._batch_rows: List[Dict] = []
        self._batch_tokens = {"prompt": 0, "generated": 0, "latent": 0}
        self._warmup_batches = 0
        self._cold_batches = 1  # until note_warmup says otherwise

        open(self.trace_path, "w", encoding="utf-8").close()  # start fresh
        open(self.batch_path, "w", encoding="utf-8").close()

        self.telemetry: Optional[metrics.TelemetrySampler] = None
        if getattr(args, "gpu_telemetry", False):
            sampler = metrics.TelemetrySampler(
                self.telemetry_path, interval=getattr(args, "telemetry_interval", 5.0))
            if sampler.start():
                self.telemetry = sampler
                print(f"[print2] gpu telemetry -> {_display_path(self.telemetry_path)}")
            else:
                print("[print2] gpu telemetry requested but nvidia-smi is unavailable",
                      file=sys.stderr)

        metrics.reset_peak_memory()
        print(f"[print2] writing run to {_display_path(self.dir)}/")

    # ------------------------------------------------------------------

    def attach_model(self, model) -> None:
        """Record what was actually loaded, once the wrapper exists.
        Record what was actually loaded, once the wrapper exists.

        `--dtype auto` and the attention backend both resolve at load time
        from what the machine offers, so the config in `args` does not say
        what ran. An L4 with no native bf16 silently becomes a float16 run;
        that belongs in the manifest, not in the operator's memory.
        """
        try:
            self.model_info = {
                "name": getattr(model, "model_name", None),
                "dtype": str(getattr(model, "dtype", None)),
                "attn_implementation": getattr(model, "attn_implementation", None),
                "append_mode": getattr(model, "append_mode", None),
                "append_carries_state": getattr(model, "append_carries_state", None),
                "state_continuation": getattr(model, "state_probe", None),
                "hidden_size": int(getattr(model.config, "hidden_size", 0)) or None,
                "vocab_size": len(model.tokenizer) if model.tokenizer else None,
                "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0)) or None,
                "linear_attention_layers": len(getattr(model, "linear_idx", []) or []),
            }
            backend = getattr(model, "backend", None)
            if backend is not None:
                # The controls answered remotely. Record what actually served
                # them: a Speed column mixing backends is meaningless, and
                # Acc/Token only transfer if the checkpoint is the same one.
                self.model_info["backend"] = backend.describe()
                self.model_info["weights_mib"] = None
                # describe() is a snapshot; the request counter only becomes
                # meaningful once the run has issued some. Refreshed at write.
                self._backend_ref = backend
            else:
                self.model_info["weights_mib"] = round(
                    sum(p.numel() * p.element_size() for p in model.model.parameters()) / MIB, 1)
            self.model_info["chat"] = _chat_fingerprint(model)
        except Exception:
            pass
        # The model is resident but nothing has run: this is the floor the
        # per-batch watermarks should be read against.
        self.model_info["memory_after_load"] = metrics.memory_snapshot()
        metrics.reset_peak_memory()

    def note_warmup(self, batches_run: int) -> None:
        """How many discarded warmup batches ran before the clock started.
        How many discarded warmup batches ran before the clock started.

        If none did, the first *measured* batch is the cold one. It stays in
        `total_time_sec`, that number has to remain honest end to end, but
        it is flagged in batches.jsonl and left out of the steady-state
        summary, so a short run's rate is not a measurement of CUDA startup.
        """
        self._warmup_batches = batches_run
        self._cold_batches = 0 if batches_run > 0 else 1

    def begin_batch(self, batch_size: int) -> None:
        """Called once per generation batch so trace rows carry batch context."""
        self._batch_index += 1
        self._batch_size = batch_size
        self._phase_counted = False
        self._batch_tokens = {"prompt": 0, "generated": 0, "latent": 0}

    def end_batch(self, wall_sec: float) -> None:
        """Close the batch out with its measured wall time.
        Close the batch out with its measured wall time.

        Appends one row to batches.jsonl. The row is what lets a reader drop
        the first (cold) batch from the Speed number and see whether the rest
        of the run held a steady rate, neither of which a total can show.
        """
        self._batch_secs.append(wall_sec)
        peak = metrics.peak_reserved_mib()
        row = {
            "batch_index": self._batch_index,
            "batch_size": self._batch_size,
            "wall_sec": round(wall_sec, 4),
            "sec_per_sample": round(wall_sec / max(1, self._batch_size), 4),
            "samples_done": self._n,
            "is_warmup": self._batch_index < self._cold_batches,
            "prompt_tokens": self._batch_tokens["prompt"],
            "generated_tokens": self._batch_tokens["generated"],
            "latent_steps": self._batch_tokens["latent"],
            "peak_reserved_mib": peak,
        }
        self._batch_rows.append(row)
        with open(self.batch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def record(self, idx: int, res: Dict, latency_sec: float) -> bool:
        """Append one sample to the JSONL trace. Returns whether it was correct."""
        self._n += 1
        ok = bool(res.get("correct"))
        self._correct += int(ok)

        agents = []
        prompt_tokens = 0
        generated_tokens = 0
        latent_steps = 0
        append_forwards = 0
        append_positions = 0
        sample_trunc_any = False
        sample_trunc_final = False
        kv_max = state_max = t_kv_max = t_state_max = 0

        raw_agents = res.get("agents", [])
        for a in raw_agents:
            in_tok = a.get("input_token_count")
            if in_tok is None:  # older/other methods: fall back to the token list
                in_tok = len(a.get("input_tokens") or [])
            out_tok = int(a.get("output_token_count") or 0)
            steps = int(a.get("latent_steps") or 0)
            prompt_tokens += in_tok
            generated_tokens += out_tok
            latent_steps += steps
            append_forwards += int(a.get("append_forwards") or 0)
            append_positions += int(a.get("append_positions") or 0)
            if a.get("truncated"):
                sample_trunc_any = True
            kv_max = max(kv_max, int(a.get("kv_bytes") or 0))
            state_max = max(state_max, int(a.get("state_bytes") or 0))
            t_kv_max = max(t_kv_max, int(a.get("transfer_kv_bytes") or 0))
            t_state_max = max(t_state_max, int(a.get("transfer_state_bytes") or 0))
            if a.get("channel_dtypes") and not getattr(self, "channel_dtypes", None):
                self.channel_dtypes = a["channel_dtypes"]
            record = {
                "name": a.get("name"),
                "role": a.get("role"),
                "latent_steps": steps,
                "input_token_count": in_tok,
                "output_token_count": out_tok,
                # wall time of this agent's stage over the whole batch;
                # divide by batch_size for a rough per-sample figure
                "agent_time_sec": a.get("agent_time_sec"),
                "input": a.get("input", ""),
                "output": a.get("output", ""),
            }
            # latent_mas on Qwen3.5 annotates each agent with the state of the
            # hybrid cache at its boundary: the length of the KV half, an RMS
            # fingerprint of the DeltaNet half, how the prompt had to be
            for extra in ("cache_len", "state_norm", "append_path",
                          "append_forwards", "append_positions",
                          "append_sec", "latent_sec", "decode_sec",
                          "kv_bytes", "state_bytes",
                          "transfer_kv_bytes", "transfer_state_bytes",
                          "truncated"):
                if a.get(extra) is not None:
                    record[extra] = a[extra]
            agents.append(record)

        # The answer that gets graded comes from the last agent, so its
        # truncation is the one that can turn a correct run into a wrong one.
        if raw_agents and raw_agents[-1].get("truncated"):
            sample_trunc_final = True
        self._truncated_final += int(sample_trunc_final)
        self._truncated_any += int(sample_trunc_any)

        # Phase timings are batch-level figures replicated onto every row, so
        # take them from the first sample of the batch only.
        if not self._phase_counted and raw_agents:
            for a in raw_agents:
                self._append_sec += float(a.get("append_sec") or 0.0)
                self._latent_sec += float(a.get("latent_sec") or 0.0)
                self._decode_sec += float(a.get("decode_sec") or 0.0)
            self._phase_counted = True

        # Theorem 3.1: text tokens this sample's latent thoughts stand in for
        # (a lower bound; None when the ratio could not be determined).
        text_equivalent = theory.text_equivalent(latent_steps)

        self._prompt_tokens += prompt_tokens
        self._generated_tokens += generated_tokens
        self._latent_steps += latent_steps
        self._text_equivalent += text_equivalent or 0.0
        self._append_forwards += append_forwards
        self._append_positions += append_positions
        self._kv_bytes_max = max(self._kv_bytes_max, kv_max)
        self._state_bytes_max = max(self._state_bytes_max, state_max)
        self._transfer_kv_max = max(self._transfer_kv_max, t_kv_max)
        self._transfer_state_max = max(self._transfer_state_max, t_state_max)
        if hasattr(self, "_batch_tokens"):
            self._batch_tokens["prompt"] += prompt_tokens
            self._batch_tokens["generated"] += generated_tokens
            self._batch_tokens["latent"] += latent_steps

        question = res.get("question") or ""
        row = {
            "idx": idx,
            # stable across runs as long as the question text is identical
            # join key for cross-run/per-sample comparisons
            "sample_id": hashlib.sha1(question.encode("utf-8")).hexdigest()[:12],
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "batch_index": self._batch_index,
            "batch_size": self._batch_size,
            "correct": ok,
            "prediction": res.get("prediction"),
            "gold": res.get("gold"),
            "error_msg": res.get("error_msg"),
            "question": question,
            "raw_prediction": res.get("raw_prediction"),
            "solution": res.get("solution"),
            "latency_sec": round(latency_sec, 4),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "latent_steps_total": latent_steps,
            "append_forwards_total": append_forwards,
            "append_positions_total": append_positions,
            "truncated_final": sample_trunc_final,
            "truncated_any": sample_trunc_any,
            "text_equivalent_tokens": round(text_equivalent, 2) if text_equivalent else None,
            "agents": agents,
        }
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return ok

    def print_sample(self, idx: int, total: int, res: Dict, latency_sec: float, ok: bool) -> None:
        if self._color:
            mark = f"{_GREEN}✓{_RESET}" if ok else f"{_RED}✗{_RESET}"
        else:
            mark = "✓" if ok else "✗"
        # Latent thoughts summed over the sample's agents, and (Theorem 3.1)
        # the text they stand in for. Both totals, so latent x ratio = equiv
        # reads consistently on screen.
        agents = res.get("agents", [])
        latent = sum(int(a.get("latent_steps") or 0) for a in agents)
        equiv = theory.text_equivalent(latent)
        equiv_str = f" ≳{theory.compact(equiv):>6} txt" if equiv else ""
        # latent_mas only: the hybrid cache at the last agent boundary, how
        # many token positions the attention half held, and the RMS of the
        # DeltaNet half. `s=None` means no mamba state crossed (expected under
        # --state_transfer kv_only/none, a bug anywhere else).
        norms = [a.get("state_norm") for a in agents if "state_norm" in a]
        lens = [a.get("cache_len") for a in agents if a.get("cache_len") is not None]
        adaptive = ""
        if lens:
            adaptive += f" kv={lens[-1]}"
        if norms:
            last = norms[-1]
            adaptive += f" s={last:.4f}" if isinstance(last, (int, float)) else " s=None"
        # A clipped final answer is scored as a wrong answer, so it must be
        # visible while the run is going, not only in the post-hoc rate.
        if agents and agents[-1].get("truncated"):
            adaptive += f"{_YELLOW} CUT{_RESET}" if self._color else " CUT"
        print(
            f"[{idx:>4}/{total}] {mark}  "
            f"pred={_short(res.get('prediction')):<14} "
            f"gold={_short(res.get('gold')):<14} "
            f"latent={latent:<4}{equiv_str} ~{latency_sec:.2f}s"  # ~ : batch-averaged
            f"{adaptive}"
        )
        if self._n % 10 == 0 and self._n:
            acc = self._correct / self._n
            elapsed = time.time() - self._t0
            eta = (total - self._n) * (elapsed / self._n)
            peak = metrics.peak_reserved_mib()
            mem = f" · vram {peak / 1024:.1f}GiB" if peak else ""
            cut = f" · cut {self._truncated_final}" if self._truncated_final else ""
            line = (f"           acc {acc:.3f} ({self._correct}/{self._n}) · "
                    f"elapsed {_fmt_dur(elapsed)} · eta {_fmt_dur(eta)}{mem}{cut}")
            print(f"{_DIM}{line}{_RESET}" if self._color else line)

    # ------------------------------------------------------------------

    def _theorem_block(self, n: int) -> Optional[Dict]:
        """Theorem 3.1 expressiveness accounting for the whole run.
        """
        info = theory.info()
        if info is None:
            return None
        block = dict(info)
        block.update(
            {
                "text_equivalent_tokens_total": round(self._text_equivalent, 2),
                "text_equivalent_tokens_per_sample": round(self._text_equivalent / n, 2),
            }
        )
        alt = theory.efficiency_ratio(info["hidden_size"], info["vocab_size"], 3.0)
        if alt is not None:
            block["text_tokens_per_latent_step_log3"] = round(alt, 4)
            block["text_equivalent_tokens_total_log3"] = round(self._latent_steps * alt, 2)
        if self._generated_tokens > 0:
            block["latent_vs_generated_ratio"] = round(
                self._text_equivalent / self._generated_tokens, 3
            )
        return block

    def _timing_block(self, elapsed: float, processed: int) -> Dict:
        """Where the wall clock went, and how steady it was.
        Where the wall clock went, and how steady it was.

        `steady_*` excludes the warmup batches. The first batch on a cold
        context pays CUDA module load, kernel autotuning and allocator growth;
        on a short run that single batch can dominate the mean and make the
        Speed column a measurement of startup.
        """
        warm = [r for r in self._batch_rows if not r["is_warmup"]]
        steady = [r["sec_per_sample"] for r in warm]
        block = {
            "total_sec": round(elapsed, 4),
            "sec_per_sample": round(elapsed / processed, 4) if processed else 0.0,
            "samples_per_sec": round(processed / elapsed, 4) if elapsed > 0 else 0.0,
            # Discarded before the clock started (--warmup), vs. measured
            # batches flagged cold and left out of the steady summary.
            "warmup_batches_discarded": self._warmup_batches,
            "cold_batches_excluded_from_steady": self._cold_batches,
            "batches": len(self._batch_rows),
            "batch_sec": metrics.summarize([r["wall_sec"] for r in self._batch_rows]),
            "steady_sec_per_sample": metrics.summarize(steady),
            # Sum of the measured stages. Falls short of total_sec by the
            # tokenizing, grading, trace writing and dataset iteration between
            # them, that gap is itself worth watching, since sandboxed code
            # execution on mbppplus/humanevalplus lands entirely inside it.
            "phase_sec": {
                "append": round(self._append_sec, 3),
                "latent": round(self._latent_sec, 3),
                "decode": round(self._decode_sec, 3),
                "measured_total": round(
                    self._append_sec + self._latent_sec + self._decode_sec, 3),
            },
        }
        if warm:
            secs = sum(r["wall_sec"] for r in warm)
            n = sum(r["batch_size"] for r in warm)
            block["steady_total_sec"] = round(secs, 4)
            block["steady_samples"] = n
            block["steady_sec_per_sample_mean"] = round(secs / n, 4) if n else None
        return block

    def _tokens_block(self, n: int, elapsed: float) -> Dict:
        """The paper's Token column, plus the denominators it needs.

        `generated_total` is the figure the paper's 70.8-83.7% output-token
        reduction is computed from. `prompt_total` is reported beside it and
        is *not* comparable across methods in the same way: text_mas re-reads
        a context that grows with every agent, while latent_mas re-reads only
        its own prompt because the collaboration arrives through the cache.
        """
        block = {
            "prompt_total": self._prompt_tokens,
            "generated_total": self._generated_tokens,
            "latent_steps_total": self._latent_steps,
            "prompt_per_sample": round(self._prompt_tokens / n, 2),
            "generated_per_sample": round(self._generated_tokens / n, 2),
            "latent_steps_per_sample": round(self._latent_steps / n, 2),
            # Throughput against total elapsed. For latent_mas most of the
            # clock is spent on appends and latent rollout, which emit no
            # tokens, so this understates decode speed by design, read it as
            # "text produced per second of run", not as decode throughput.
            "generated_tok_per_sec": round(self._generated_tokens / elapsed, 2) if elapsed > 0 else 0.0,
            # Forwards and positions pushed through the model by the hybrid
            # append path. The cost of --mamba_append and of ragged batching
            # is denominated here, not in the token counts.
            "append_forwards_total": self._append_forwards,
            "append_positions_total": self._append_positions,
            "append_forwards_per_sample": round(self._append_forwards / n, 2),
        }
        if self._decode_sec > 0:
            block["decode_tok_per_sec"] = round(self._generated_tokens / self._decode_sec, 2)
        return block

    def _channel_block(self) -> Dict:
        """Size of the memory that crosses an agent boundary, in bytes.

        make "the inter-agent channel is O(1)" a measured claim instead of an
        architectural observation.
        """
        return {
            "workspace_kv_mib_max": _mib(self._kv_bytes_max),
            "workspace_state_mib_max": _mib(self._state_bytes_max),
            "transferred_kv_mib_max": _mib(self._transfer_kv_max),
            "transferred_state_mib_max": _mib(self._transfer_state_max),
            "transferred_total_mib_max": _mib(self._transfer_kv_max + self._transfer_state_max),
            "dtypes": getattr(self, "channel_dtypes", None),
            "state_transfer": getattr(self.args, "state_transfer", None),
        }

    def write_manifest(self, *, accuracy: float, correct: int, processed: int,
                       total: int, elapsed: float, status: str) -> None:
        """(Over)write manifest.json with the current run state.
        (Over)write manifest.json with the current run state.

        Called after every batch with status="running" so the file always
        reflects the last completed batch, and once at the end with
        status="completed" or "interrupted". Cheap enough to rewrite each time.
        """
        n = max(1, self._n)
        backend = getattr(self, "_backend_ref", None)
        if backend is not None:
            self.model_info["backend"] = backend.describe()
        manifest = {
            "run_id": self.run_id,
            "command": " ".join(sys.argv),
            "git": self.git,
            "environment": self.env,
            "model": self.model_info,
            "started": self.started,
            "updated": datetime.now(timezone.utc).isoformat(),
            "status": status,  # running | completed | interrupted
            "args": vars(self.args),
            "accuracy": accuracy,
            # A finite test set makes accuracy an estimate. The comparison this
            # fork is for is a difference between two of these, so the interval
            # is reported alongside: a gap smaller than the half-widths is not
            # a result, it is a seed.
            "accuracy_ci95": metrics.wilson_interval(correct, processed),
            "correct": correct,
            "processed": processed,
            "total": total,
            "total_time_sec": round(elapsed, 4),
            "time_per_sample_sec": round(elapsed / processed, 4) if processed else 0.0,
            "timing": self._timing_block(elapsed, processed),
            "tokens": self._tokens_block(n, elapsed),
            "generation": {
                # Answers clipped at --max_new_tokens. A nonzero rate means the
                # accuracy below is partly a measurement of the token budget,
                # and the token totals are partly a measurement of the cap.
                "truncated_final": self._truncated_final,
                "truncated_final_rate": round(self._truncated_final / n, 4),
                "truncated_any_agent": self._truncated_any,
                "max_new_tokens": getattr(self.args, "max_new_tokens", None),
                "temperature": getattr(self.args, "temperature", None),
                "top_p": getattr(self.args, "top_p", None),
                "greedy": (getattr(self.args, "temperature", 1.0) or 0) <= 0,
                "seed": getattr(self.args, "seed", None),
            },
            "memory": {
                "peak": metrics.memory_snapshot(),
                "after_load": self.model_info.get("memory_after_load"),
            },
            "channel": self._channel_block(),
            "theorem_3_1": self._theorem_block(n),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        if status != "running":  # avoid spamming the console every batch
            print(f"[print2] {_display_path(self.trace_path)}  ({self._n} samples, {status})")
            print(f"[print2] {_display_path(self.batch_path)}  ({len(self._batch_rows)} batches)")
            print(f"[print2] {_display_path(self.manifest_path)}  acc={accuracy:.4f} ({correct}/{processed})")
            if self._truncated_final:
                print(f"[print2] WARNING: {self._truncated_final}/{self._n} answers hit "
                      f"--max_new_tokens and were cut off. Accuracy and token totals "
                      f"both reflect the cap, not the method.", file=sys.stderr)

    def close(self) -> None:
        if self.telemetry is not None:
            self.telemetry.stop()
