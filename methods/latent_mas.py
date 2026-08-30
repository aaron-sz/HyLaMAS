"""LatentMAS on Qwen3.5: the 4-agent latent chain over a hybrid cache.

Derived from Gen-Verse/LatentMAS `methods/latent_mas.py` (Apache-2.0).

    state_only  drop the KV; the channel is then O(1) in how much was thought
    none        drop both (no-collaboration control)
"""

import argparse
import sys
import time
from typing import Dict, List

import torch

import debug
import hybrid
from hybrid import Workspace
from models import ModelWrapper
from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)
from utils import (
    extract_gsm8k_answer,
    extract_markdown_python_block,
    normalize_answer,
    run_with_timeout,
)

from . import default_agents

TRANSFER_MODES = ("both", "kv_only", "state_only", "none")
TURN_MODES = ("raw", "content", "think")
VISIBILITY_MODES = ("full", "latent_only")

# ChatML fragments used to frame a latent block as a well-formed chat turn.
# Qwen3.5 is ChatML (prompts.py already asserts a Qwen model), but these are
# checked against the live template at startup rather than trusted -- the same
# policy as the append probe in hybrid.py, because a template change would
# otherwise degrade the workspace silently instead of loudly.
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
ASSISTANT_OPEN = IM_START + "assistant\n"
THINK_OPEN = "<think>\n"
THINK_EMPTY = "<think>\n\n</think>\n\n"
THINK_CLOSE = "\n</think>\n\n"

# What the workspace says an agent contributed when its prompt is withheld
# (--latent_visibility latent_only). The instructions stay private; a short
# role line keeps the conversation well formed and tells the judger whose
# thoughts it is looking at.
ROLE_LABELS = {
    "planner": "Planner's plan (latent).",
    "critic": "Critic's feedback on the plan (latent).",
    "refiner": "Refiner's revised plan (latent).",
}


class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = "latent_mas"
        self.task = args.task
        self.state_transfer = getattr(args, "state_transfer", "both")
        if self.state_transfer not in TRANSFER_MODES:
            raise ValueError(f"unknown --state_transfer {self.state_transfer!r}")

        self.latent_turn = getattr(args, "latent_turn", "raw")
        if self.latent_turn not in TURN_MODES:
            raise ValueError(f"unknown --latent_turn {self.latent_turn!r}")
        self.latent_visibility = getattr(args, "latent_visibility", "full")
        if self.latent_visibility not in VISIBILITY_MODES:
            raise ValueError(f"unknown --latent_visibility {self.latent_visibility!r}")
        self.chat_thinking = getattr(args, "chat_thinking", "auto")
        if self.chat_thinking not in ("auto", "on", "off"):
            raise ValueError(f"unknown --chat_thinking {self.chat_thinking!r}")
        self.judger_thinking = getattr(args, "judger_thinking", "inherit")
        if self.judger_thinking == "inherit":
            self.judger_thinking = self.chat_thinking
        if self.judger_thinking not in ("auto", "on", "off"):
            raise ValueError(f"unknown --judger_thinking {self.judger_thinking!r}")
        self._tails = None  # discovered lazily from the live chat template
        if self.latent_turn != "raw" or self.latent_visibility != "full":
            self._check_chat_fragments()
        debug.dbg(f"latent_mas: latent_turn={self.latent_turn} "
                  f"latent_visibility={self.latent_visibility} "
                  f"chat_thinking={self.chat_thinking} "
                  f"judger_thinking={self.judger_thinking}")
        self._warn_if_judger_reasons_unbounded()

        if self.generate_bs > 1:
            # Note that this inverts the usual advice: rows
            # of a batch have different prompt lengths, so they get padded, and
            # a padded chunk over a live recurrent state has to feed its
            debug.dbg(f"latent_mas: generate_bs={self.generate_bs} > 1, ragged "
                      f"batches pay extra forwards on every append after the "
                      f"first; compare against --generate_bs 1 on your hardware")

    # ------------------------------------------------------------------

    def _batch_messages(self, role: str, items: List[Dict]) -> List[List[Dict]]:
        if self.args.prompt == "hierarchical":
            builder = build_agent_message_hierarchical_latent_mas
        else:
            builder = build_agent_message_sequential_latent_mas
        return [
            builder(role=role, question=item["question"], context="",
                    method=self.method_name, args=self.args)
            for item in items
        ]

    # ------------------------------------------------------------------
    # Chat framing
    # ------------------------------------------------------------------

    def _discover_tails(self) -> dict:
        """Learn this template's assistant-turn tails rather than assume them.
        """
        if self._tails is not None:
            return self._tails
        probe = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        tails = {}
        for key, flag in (("auto", None), ("on", True), ("off", False)):
            try:
                txt = self.model.render_chat(probe, add_generation_prompt=True,
                                             enable_thinking=flag)
            except Exception:
                txt = self.model.render_chat(probe, add_generation_prompt=True)
            cut = txt.rfind(ASSISTANT_OPEN)
            if cut < 0:
                raise ValueError(
                    "--latent_turn / --latent_visibility need a ChatML template "
                    f"whose generation prompt contains {ASSISTANT_OPEN!r}; this "
                    f"model's ends with {txt[-60:]!r}. Use --latent_turn raw "
                    "--latent_visibility full for upstream behavior.")
            tails[key] = txt[cut + len(ASSISTANT_OPEN):]
        self._tails = tails
        debug.dbg("latent_mas: template tails " + repr(tails))
        return tails

    def _check_chat_fragments(self) -> None:
        """Verify the hardcoded ChatML pieces against the live template.
        """
        tails = self._discover_tails()
        if tails["on"] == tails["off"] and self.latent_turn == "think":
            debug.dbg("latent_mas: this template ignores enable_thinking; "
                      "--latent_turn think will frame latents with its default tail")
        debug.dbg("latent_mas: chat framing resolved against the live template")

    def _warn_if_judger_reasons_unbounded(self) -> None:
        """Say so, loudly, when the judger will decode inside an open <think>.

        Qwen3.5-9B / GSM8K: 20.5% of samples truncated, and accuracy fell from
        96.2% (on samples that finished) to 76.5% overall.
        """
        try:
            tail = self._discover_tails()[self._tail_key(judger=True)]
        except Exception:
            return
        if tail.count("<think>") <= tail.count("</think>"):
            return
        cap = getattr(self.args, "max_new_tokens", None)
        print(
            f"[warn] the judger will decode inside an OPEN <think> block "
            f"(chat_thinking={self.chat_thinking}, judger_thinking="
            f"{self.judger_thinking}, tail={tail!r}).\n"
            f"       On this template that is unbounded reasoning. Expect a high "
            f"truncated_final_rate at --max_new_tokens {cap}, and note that a\n"
            f"       truncated judger is scored on the last \\boxed{{}} it emitted "
            f"-- often the prompt's own placeholder, not its answer.\n"
            f"       Pass --chat_thinking off (or --judger_thinking off) for a "
            f"direct answer. Check generation.truncated_final_rate afterwards.",
            file=sys.stderr,
        )

    def _system_text(self, messages) -> str:
        for m in messages:
            if m.get("role") == "system":
                return IM_START + "system\n" + m["content"] + IM_END + "\n"
        return ""

    def _user_text(self, content: str) -> str:
        return IM_START + "user\n" + content + IM_END + "\n"

    def _tail_key(self, judger: bool = False) -> str:
        """Which discovered tail this turn should open with.
        """
        if judger:
            return self.judger_thinking          # auto | on | off
        return "on" if self.latent_turn == "think" else self.chat_thinking

    def _assistant_tail(self, judger: bool = False) -> str:
        """How a turn opens, for the turn the model is about to speak in.
        How a turn opens, for the turn the model is about to speak in.

        The judger is decided separately from the agents. This matters because on a thinking-by-default template the judger reasons
        without bound, which both inflates the Token column and -- once it runs
        past --max_new_tokens -- costs the answer, because the grader takes the
        last \boxed{...} and an unterminated ramble tends to quote the prompt's
        own \boxed{YOUR_FINAL_ANSWER} placeholder instead.
        """
        return ASSISTANT_OPEN + self._discover_tails()[self._tail_key(judger)]

    def _closer_text(self) -> str:
        """What is written after a latent block to end that assistant turn.
        What is written after a latent block to end that assistant turn.

        Derived, not fixed: a `</think>` is emitted only when the tail this
        agent opened with actually left a thinking block open.
        """
        if self.latent_turn == "raw":
            return ""
        tail = self._discover_tails()[self._tail_key(False)]
        open_think = tail.count("<think>") > tail.count("</think>")
        return (THINK_CLOSE if open_think else "") + IM_END + "\n"

    def _prompts_for(self, agent, items, agent_index: int):
        """The text each row of the batch is about to have appended.
        """
        batch_messages = self._batch_messages(agent.role, items)
        if self.latent_turn == "raw":
            # `auto` keeps upstream exactly: whatever the template does. `on`/
            # `off` reach into the judger's turn only, which is the one place
            # unbounded reasoning costs both headline columns at once.
            flag = {"auto": None, "on": True, "off": False}[
                self.judger_thinking if agent.role == "judger" else "auto"]
            prompts = [self.model.render_chat(m, add_generation_prompt=True,
                                              enable_thinking=flag)
                       for m in batch_messages]
            if self.args.think:
                prompts = [f"{p}<think>" for p in prompts]
            return prompts, batch_messages

        # Under latent_only every agent's own prompt is private, so each one
        # renders as a standalone prompt with its own system block.
        standalone = (agent_index == 0) or (self.latent_visibility == "latent_only")
        prompts = []
        for m in batch_messages:
            head = self._system_text(m) if standalone else ""
            prompts.append(head + self._user_text(m[-1]["content"])
                           + self._assistant_tail(judger=(agent.role == "judger")))
        return prompts, batch_messages

    # ------------------------------------------------------------------
    # latent_only: think with the prompt, publish only the thoughts
    # ------------------------------------------------------------------

    def _mark(self, ws: Workspace):
        """Everything needed to rewind the workspace to this instant."""
        linear = self.model.linear_idx
        snaps = (hybrid.snapshot_state(ws.cache, linear) if ws.cache is not None
                 else [(i, None, None) for i in linear])
        return (ws.kv_len, snaps, None if ws.mask is None else ws.mask.clone())

    def _publish_prefix(self, agent, first: bool) -> str:
        """The structural tokens written before a withheld agent's thoughts."""
        if self.latent_turn == "raw":
            return ""
        head = (IM_START + "system\nYou are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant." + IM_END + "\n") if first else ""
        label = ROLE_LABELS.get(agent.role, agent.name + " (latent).")
        return head + self._user_text(label) + self._assistant_tail(judger=False)

    def _batch_ids(self, text: str, batch: int):
        ids = self.model.tokenize_text(text)
        if ids.shape[0] != batch:
            ids = ids.expand(batch, -1).contiguous()
        return ids

    def _rewind_and_publish(self, ws: Workspace, mark, latents, agent,
                            batch: int) -> Workspace:
        """Undo this agent's prompt, then write back only its latent thoughts."""
        kv_len, snaps, mask = mark
        hybrid.truncate_kv(ws.cache, kv_len)
        hybrid.restore_state(ws.cache, snaps)
        ws.mask = mask
        prefix = self._publish_prefix(agent, first=(kv_len == 0))
        suffix = self._closer_text()
        return self.model.append_embeds(
            ws, latents,
            prefix_ids=self._batch_ids(prefix, batch) if prefix else None,
            suffix_ids=self._batch_ids(suffix, batch) if suffix else None,
        )

    def _close_turn(self, ws: Workspace, batch: int) -> Workspace:
        """Terminate an agent's assistant turn when its prompt is staying put."""
        text = self._closer_text()
        if not text:
            return ws
        ids = self._batch_ids(text, batch)
        mask = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.long, device=ids.device)
        ws, _, _ = self.model.append(ids, mask, ws)
        return ws

    def _apply_transfer_policy(self, ws: Workspace) -> Workspace:
        """What crosses the agent boundary, per --state_transfer."""
        if ws.cache is None or self.state_transfer == "both":
            return ws
        if self.state_transfer == "kv_only":
            hybrid.strip_state(ws.cache)
            return ws
        if self.state_transfer == "state_only":
            hybrid.strip_kv(ws.cache)
            # The mask described KV positions and none are left; the recurrent
            # state carries no positions, so the next agent starts at 0.
            ws.mask = None
            return ws
        return Workspace()  # "none"

    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        ws = Workspace()
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        for agent_index, agent in enumerate(self.agents):
            debug.dbg(f"agent {agent.name} ({agent.role}), "
                      f"{'text decode' if agent.role == 'judger' else 'latent thinking'}")
            debug.set_context(agent.name)  # labels the append/latent/decode meters
            agent_t0 = time.time()

            prompts, batch_messages = self._prompts_for(agent, items, agent_index)
            input_ids, attention_mask = self.model.tokenize_batch(prompts)
            token_counts = attention_mask.sum(dim=1).tolist()

            if agent.role != "judger":
                # Under latent_only the workspace is rewound to here afterwards,
                # so this agent reads its own prompt but no later agent does.
                mark = self._mark(ws) if self.latent_visibility == "latent_only" else None
                ws, stats = self.model.generate_latent_batch(
                    input_ids,
                    attention_mask,
                    latent_steps=self.latent_steps,
                    workspace=ws,
                )
                if mark is not None:
                    ws = self._rewind_and_publish(
                        ws, mark, stats.get("latents") or [], agent, batch_size)
                else:
                    ws = self._close_turn(ws, batch_size)
                ws = self._apply_transfer_policy(ws)

                agent_time_sec = round(time.time() - agent_t0, 4)
                norm = ws.state_norm()
                cache_len = ws.kv_len
                # Measured *after* the transfer policy, so these are the bytes
                # that genuinely cross into the next agent rather than the
                # bytes this agent happened to produce. Under `state_only`
                # transfer_kv_bytes is 0 and transfer_state_bytes is constant
                # no matter how long the agents thought, that pair of columns
                # is the O(1)-channel claim, stated in bytes.
                transfer_kv = hybrid.kv_bytes(ws.cache)
                transfer_state = hybrid.state_bytes(ws.cache)
                channel_dtypes = hybrid.channel_dtypes(ws.cache)
                debug.dbg(f"workspace after {agent.name}: {ws.describe()}")
                for idx in range(batch_size):
                    active = input_ids[idx][attention_mask[idx].bool()].tolist()
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": active,
                        "input_tokens": self.model.tokenizer.convert_ids_to_tokens(active),
                        "input_token_count": int(token_counts[idx]),
                        "latent_steps": self.latent_steps,
                        "output": "",
                        "output_token_count": 0,
                        "agent_time_sec": agent_time_sec,
                        "append_sec": stats.get("append_sec"),
                        "latent_sec": stats.get("latent_sec"),
                        "decode_sec": 0.0,
                        "cache_len": cache_len,
                        "state_norm": norm,
                        "append_forwards": stats["append_forwards"],
                        "append_positions": stats.get("append_positions"),
                        "append_path": stats["append_reason"],
                        "kv_bytes": stats.get("kv_bytes"),
                        "state_bytes": stats.get("state_bytes"),
                        "transfer_kv_bytes": transfer_kv,
                        "transfer_state_bytes": transfer_state,
                        "channel_dtypes": channel_dtypes,
                    })
            else:
                # Upstream parity: with no latent thinking there is nothing
                # worth handing over, and upstream decodes the judger cold.
                ws_for_decoding = ws if self.latent_steps > 0 else None
                texts, gen_counts, gen_stats = self.model.generate_text_with_workspace(
                    input_ids,
                    attention_mask,
                    ws_for_decoding,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                agent_time_sec = round(time.time() - agent_t0, 4)
                truncated = gen_stats.get("truncated") or [None] * batch_size
                for idx in range(batch_size):
                    final_texts[idx] = texts[idx].strip()
                    active = input_ids[idx][attention_mask[idx].bool()].tolist()
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": active,
                        "input_tokens": self.model.tokenizer.convert_ids_to_tokens(active),
                        "input_token_count": int(token_counts[idx]),
                        "output": final_texts[idx],
                        "output_token_count": gen_counts[idx],
                        "agent_time_sec": agent_time_sec,
                        "append_sec": gen_stats.get("append_sec"),
                        "latent_sec": 0.0,
                        "decode_sec": gen_stats.get("decode_sec"),
                        "append_forwards": gen_stats.get("append_forwards"),
                        "append_positions": gen_stats.get("append_positions"),
                        "append_path": gen_stats.get("append_reason"),
                        "truncated": truncated[idx],
                    })

        debug.set_context()  # no agent active while scoring

        # Grading is upstream's, verbatim, so a number from this fork is
        # comparable to a number from the original.
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "error_msg": error_msg,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
