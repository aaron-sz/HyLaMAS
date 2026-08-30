"""Self-test for the Qwen3.5 hybrid-cache handling. No download, no GPU.
"""

import argparse
import sys
import tempfile
from types import SimpleNamespace
from typing import List

import torch

import hybrid
from hybrid import Workspace
from models import ModelWrapper

TINY = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128,
    num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=1,
    head_dim=32, max_position_embeddings=512,
    linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,
    full_attention_interval=4, tie_word_embeddings=False,
)
TOL = 1e-5          # what "the same" means for a state comparison
PAD_ID = 0


class _StubTokenizer:
    """Just enough tokenizer for the primitives under test."""

    pad_token_id = PAD_ID

    def __len__(self):
        return TINY["vocab_size"]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [str(int(i)) for i in ids]


def build_wrapper(append_mode="auto", dtype=torch.float32):
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(**TINY)
    model = Qwen3_5ForCausalLM(cfg).eval().to(dtype)
    args = SimpleNamespace(latent_space_realign=False, mamba_append=append_mode,
                           dtype=str(dtype).replace("torch.", ""))
    return ModelWrapper.from_loaded(model, _StubTokenizer(), args=args, device="cpu")


def ragged(rows: List[torch.Tensor]):
    """Left-pad a list of [1, n] id tensors into (ids, mask)."""
    width = max(r.shape[1] for r in rows)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, width - r.shape[1]:] = r[0]
        mask[i, width - r.shape[1]:] = 1
    return ids, mask


def recurrent(ws: Workspace, row: int):
    out = []
    for idx, _ in hybrid.linear_layers(ws.cache):
        _, rec = hybrid.get_state(ws.cache, idx)
        out.append(rec[row:row + 1].detach().float().clone())
    return out


def conv(ws: Workspace, row: int):
    out = []
    for idx, _ in hybrid.linear_layers(ws.cache):
        c, _ = hybrid.get_state(ws.cache, idx)
        out.append(c[row:row + 1].detach().float().clone())
    return out


def worst(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append((name, bool(passed), detail))
    mark = "PASS" if passed else "FAIL"
    # ASCII only: this runs on whatever console the reader has.
    print(f"  [{mark}] {name}" + (f"  --  {detail}" if detail else ""))
    return passed


# ---------------------------------------------------------------------------


def two_agent_run(mw, prompts_a, prompts_b, latent_steps=0):
    """Agent A appends+thinks, agent B appends+thinks. Returns the workspace."""
    ids, mask = ragged(prompts_a)
    ws, _ = mw.generate_latent_batch(ids, mask, latent_steps=latent_steps)
    ids, mask = ragged(prompts_b)
    ws, stats = mw.generate_latent_batch(ids, mask, latent_steps=latent_steps, workspace=ws)
    return ws, stats


def main() -> int:
    torch.manual_seed(0)
    mw = build_wrapper()
    print(f"model: {hybrid.architecture_summary(mw.model.config)}")
    print(f"append mode in effect: {mw.append_mode}\n")

    # Two agents' prompts, ragged the way real questions are.
    a_rows = [torch.randint(1, 100, (1, n)) for n in (9, 6)]
    b_rows = [torch.randint(1, 100, (1, n)) for n in (12, 8)]
    STEPS = 4

    print("1. the append contract")
    check("a multi-token append continues the DeltaNet state",
          mw.append_carries_state,
          "otherwise every agent boundary silently resets 3/4 of the model")
    # The probe must survive low precision. The recurrent state is stored in
    # the model dtype, so in bf16 a *correct* append lands ~1% away from a
    # one-shot prefill -- which an absolute fp32-style tolerance reads as a
    # failure and silently demotes every append to stepwise. Real runs are
    # bf16, so test the dtype real runs use.
    for dt in (torch.bfloat16, torch.float16):
        name = str(dt).replace("torch.", "")
        try:
            check(f"the probe still says 'carries' in {name}",
                  build_wrapper(dtype=dt).append_carries_state,
                  "a false negative here costs ~200 forwards per agent append")
        except Exception as exc:
            check(f"the probe still says 'carries' in {name}", False,
                  f"{type(exc).__name__}: {exc}")

    print("\n2. ragged batch vs. one row at a time (the thing that must hold)")
    ref = []
    for i in range(2):
        w, _ = two_agent_run(mw, [a_rows[i]], [b_rows[i]])
        ref.append((recurrent(w, 0), conv(w, 0), w.kv_len))
    ws, stats = two_agent_run(mw, a_rows, b_rows)
    ok = True
    for i in range(2):
        dr = worst(ref[i][0], recurrent(ws, i))
        dc = worst(ref[i][1], conv(ws, i))
        ok &= check(f"row {i} recurrent + conv state after 2 agents",
                    dr < TOL and dc < TOL, f"recurrent {dr:.2e}, conv {dc:.2e}")
    check("the second append took the split route",
          "pad prefix" in stats["append_reason"], stats["append_reason"])

    print("\n3. what the naive one-forward append would have cost")
    naive = build_wrapper(append_mode="chunk")
    ws_n, _ = two_agent_run(naive, a_rows, b_rows)
    errs = [(worst(ref[i][0], recurrent(ws_n, i)), worst(ref[i][1], conv(ws_n, i)))
            for i in range(2)]
    padded_row = 1  # the shorter one, so it carries padding
    dr, dc = errs[padded_row]
    check("--mamba_append chunk is measurably wrong on a padded row",
          dr > TOL or dc > TOL,
          f"recurrent {dr:.2e}, conv {dc:.2e} (this is the error 'auto' removes)")

    print("\n4. latent rollout on a ragged batch")
    ref_l = []
    for i in range(2):
        w, _ = two_agent_run(mw, [a_rows[i]], [b_rows[i]], latent_steps=STEPS)
        ref_l.append((recurrent(w, 0), conv(w, 0)))
    ws_l, _ = two_agent_run(mw, a_rows, b_rows, latent_steps=STEPS)
    for i in range(2):
        dr = worst(ref_l[i][0], recurrent(ws_l, i))
        dc = worst(ref_l[i][1], conv(ws_l, i))
        check(f"row {i} state after {STEPS} latent thoughts per agent",
              dr < TOL and dc < TOL, f"recurrent {dr:.2e}, conv {dc:.2e}")
    check("latent thoughts moved the state", ws_l.state_norm() is not None,
          f"state RMS {ws_l.state_norm()}")

    print("\n5. decoding over the workspace")
    prompt = torch.randint(1, 100, (2, 7))
    pmask = torch.ones(2, 7, dtype=torch.long)
    ws_d, _, _ = mw.append(prompt[:, :-1], pmask[:, :-1], Workspace())
    with torch.no_grad():
        cached = mw._forward(input_ids=prompt[:, -1:],
                             attention_mask=torch.cat([ws_d.mask, pmask[:, -1:]], dim=-1),
                             position_ids=ws_d.next_pos(2, prompt.device),
                             past_key_values=ws_d.cache)
        fresh = mw._forward(input_ids=prompt, attention_mask=pmask)
    d = float((cached.logits[:, -1, :] - fresh.logits[:, -1, :]).abs().max())
    check("cached prefill gives the same next-token logits as an uncached one",
          d < 1e-3, f"max logit diff {d:.2e}")
    texts, counts, gen_stats = mw.generate_text_with_workspace(
        prompt, pmask, ws_d, max_new_tokens=5, temperature=1.0, top_p=1.0)
    check("generate_text_with_workspace produces output",
          len(texts) == 2 and all(c > 0 for c in counts), f"token counts {counts}")
    # The judger's row carries the fields the run record depends on: a decode
    # timing and one truncation verdict per row. A None verdict means the
    # tokenizer declares no stop token, which would silently disable
    # truncation reporting for the whole run.
    check("decode stats carry timing and per-row truncation",
          isinstance(gen_stats.get("decode_sec"), float)
          and len(gen_stats.get("truncated") or []) == 2,
          f"decode_sec={gen_stats.get('decode_sec')} truncated={gen_stats.get('truncated')}")

    # The judger's real situation: a ragged batch, decoded on top of a
    # workspace two agents deep. This is where generate() has to derive
    # padding-aware positions itself from the mask we hand it.
    ws_r, _ = two_agent_run(mw, a_rows, b_rows, latent_steps=2)
    j_rows = [torch.randint(1, 100, (1, n)) for n in (11, 7)]
    j_ids, j_mask = ragged(j_rows)
    ws_r, _, _ = mw.append(j_ids[:, :-1], j_mask[:, :-1], ws_r)
    with torch.no_grad():
        got = mw.model.generate(
            input_ids=j_ids[:, -1:],
            attention_mask=torch.cat([ws_r.mask, j_mask[:, -1:]], dim=-1),
            past_key_values=ws_r.cache, max_new_tokens=3, do_sample=False,
            pad_token_id=PAD_ID, return_dict_in_generate=True, output_scores=True)
    ok = True
    for i in range(2):
        w, _ = two_agent_run(mw, [a_rows[i]], [b_rows[i]], latent_steps=2)
        w, _, _ = mw.append(j_rows[i][:, :-1], torch.ones(1, j_rows[i].shape[1] - 1,
                                                          dtype=torch.long), w)
        with torch.no_grad():
            want = mw.model.generate(
                input_ids=j_rows[i][:, -1:],
                attention_mask=torch.cat([w.mask, torch.ones(1, 1, dtype=torch.long)], dim=-1),
                past_key_values=w.cache, max_new_tokens=3, do_sample=False,
                pad_token_id=PAD_ID, return_dict_in_generate=True, output_scores=True)
        d = max(float((got.scores[s][i:i + 1] - want.scores[s]).abs().max()) for s in range(3))
        ok &= check(f"row {i} decodes the same in a ragged batch as alone",
                    d < 1e-3, f"max logit diff over 3 steps {d:.2e}")

    print("\n6. --state_transfer switches")
    ws_t, _ = two_agent_run(mw, a_rows, b_rows, latent_steps=2)
    kv_before, norm_before = ws_t.kv_len, ws_t.state_norm()
    check("both: KV and state are present", kv_before > 0 and norm_before is not None,
          f"kv_len={kv_before} state RMS={norm_before}")

    for mode, want_kv, want_state in (("kv_only", True, False),
                                      ("state_only", False, True),
                                      ("none", False, False)):
        w, _ = two_agent_run(mw, a_rows, b_rows, latent_steps=2)
        if mode == "kv_only":
            hybrid.strip_state(w.cache)
        elif mode == "state_only":
            hybrid.strip_kv(w.cache)
            w.mask = None
        else:
            w = Workspace()
        has_kv, has_state = w.kv_len > 0, w.state_norm() is not None
        check(f"{mode}: kv={'kept' if want_kv else 'dropped'}, "
              f"state={'kept' if want_state else 'dropped'}",
              has_kv == want_kv and has_state == want_state,
              f"kv_len={w.kv_len} state RMS={w.state_norm()}")
        ids, m = ragged(b_rows)
        before = w.kv_len
        try:
            w2, _, _ = mw.append(ids, m, w)
            check(f"{mode}: the next agent can still append", w2.kv_len > before,
                  f"kv_len {before} -> {w2.kv_len}")
        except Exception as exc:
            check(f"{mode}: the next agent can still append", False,
                  f"{type(exc).__name__}: {exc}")

    print("\n7. rewind + republish (--latent_visibility latent_only)")
    # (a) A rewind must put the workspace back exactly. Snapshot, run a whole
    #     agent turn over it, rewind, and compare against the snapshot: if the
    #     DeltaNet state does not come back, `latent_only` is not withholding
    #     the prompt, it is corrupting the workspace.
    ws_r, _ = two_agent_run(mw, a_rows, b_rows, latent_steps=STEPS)
    before_rec = [recurrent(ws_r, i) for i in range(2)]
    before_conv = [conv(ws_r, i) for i in range(2)]
    before_len = ws_r.kv_len
    snaps = hybrid.snapshot_state(ws_r.cache, mw.linear_idx)
    ids, m = ragged([torch.randint(1, 100, (1, n)) for n in (10, 6)])
    ws_r, stats = mw.generate_latent_batch(ids, m, latent_steps=STEPS, workspace=ws_r)
    grew = ws_r.kv_len > before_len
    hybrid.truncate_kv(ws_r.cache, before_len)
    hybrid.restore_state(ws_r.cache, snaps)
    dr = max(worst(before_rec[i], recurrent(ws_r, i)) for i in range(2))
    dc = max(worst(before_conv[i], conv(ws_r, i)) for i in range(2))
    check("a turn can be rewound out of the workspace exactly",
          grew and ws_r.kv_len == before_len and dr < TOL and dc < TOL,
          f"kv_len {before_len} -> grew -> {ws_r.kv_len}, recurrent {dr:.2e}, conv {dc:.2e}")

    # (b) The real claim: after latent_only publishing, the workspace must be
    #     indistinguishable from one in which the prompt NEVER existed. The
    #     reference appends only the latent vectors, so if any part of the
    #     prompt had leaked into the recurrent state these would diverge.
    torch.manual_seed(1)
    ws_a, _ = two_agent_run(mw, a_rows, b_rows, latent_steps=0)   # shared history
    mark_len = ws_a.kv_len
    mark_snaps = hybrid.snapshot_state(ws_a.cache, mw.linear_idx)
    mark_mask = ws_a.mask.clone()
    p_ids, p_mask = ragged([torch.randint(1, 100, (1, n)) for n in (11, 7)])
    ws_a, st = mw.generate_latent_batch(p_ids, p_mask, latent_steps=STEPS, workspace=ws_a)
    vecs = st["latents"]
    hybrid.truncate_kv(ws_a.cache, mark_len)
    hybrid.restore_state(ws_a.cache, mark_snaps)
    ws_a.mask = mark_mask
    ws_a = mw.append_embeds(ws_a, vecs)

    torch.manual_seed(1)
    ws_b, _ = two_agent_run(mw, a_rows, b_rows, latent_steps=0)   # same history
    ws_b = mw.append_embeds(ws_b, vecs)                            # thoughts only

    dr = max(worst(recurrent(ws_b, i), recurrent(ws_a, i)) for i in range(2))
    dc = max(worst(conv(ws_b, i), conv(ws_a, i)) for i in range(2))
    check("latent_only leaves a workspace the prompt never touched",
          ws_a.kv_len == ws_b.kv_len and dr < TOL and dc < TOL,
          f"kv_len {ws_a.kv_len} vs {ws_b.kv_len}, recurrent {dr:.2e}, conv {dc:.2e}")
    check("only the latent thoughts crossed the boundary",
          ws_a.kv_len == mark_len + STEPS,
          f"kv_len {mark_len} + {STEPS} thoughts = {ws_a.kv_len}")

    print("\n8. loading a Qwen3.5 checkpoint as a text-only causal LM")
    check(*_check_checkpoint_load())

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


def _check_checkpoint_load():
    """Real Qwen3.5 checkpoints are multimodal; we want only the text stack."""
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5 import (
        Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5VisionConfig,
    )
    try:
        vis = Qwen3_5VisionConfig(depth=2, hidden_size=32, intermediate_size=64,
                                  num_heads=2, out_hidden_size=TINY["hidden_size"],
                                  num_position_embeddings=64)
        cfg = Qwen3_5Config(text_config=dict(TINY), vision_config=vis.to_dict())
        vlm = Qwen3_5ForConditionalGeneration(cfg).eval()
        with tempfile.TemporaryDirectory() as d:
            vlm.save_pretrained(d, safe_serialization=True)
            text = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
            same = torch.equal(vlm.model.language_model.embed_tokens.weight,
                               text.model.embed_tokens.weight)
            no_vision = not hasattr(text.model, "visual")
            return ("AutoModelForCausalLM loads the text stack, drops the vision tower",
                    same and no_vision,
                    f"{type(text).__name__}, weights match={same}, vision dropped={no_vision}")
    except Exception as exc:
        return ("AutoModelForCausalLM loads the text stack, drops the vision tower",
                False, f"{type(exc).__name__}: {exc}")


def save_tiny(directory: str) -> int:
    """Write a tiny random Qwen3.5 + the real tokenizer, for a run.py smoke test.
    Write a tiny random Qwen3.5 + the real tokenizer, for a run.py smoke test.

    The weights are random, so answers are garbage and accuracy is 0 by
    construction. The point is to run the whole pipeline, prompts, chat
    template, four agents, hybrid handoff, grading, --print2 record, in
    seconds on a CPU before spending GPU hours.
    """
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    print("downloading the Qwen3.5 tokenizer (a few MB) ...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    spec = dict(TINY)
    spec["vocab_size"] = len(tok)
    spec["tie_word_embeddings"] = True  # halves the file; irrelevant to the test
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**spec)).eval().to(torch.bfloat16)
    model.save_pretrained(directory, safe_serialization=True)
    tok.save_pretrained(directory)
    print(f"wrote a tiny random Qwen3.5 to {directory}")
    print("smoke test (random weights, accuracy will be 0.00):")
    print(f"  python run.py --method latent_mas --model_name {directory} \\\n"
          f"      --device cpu --task gsm8k --max_samples 4 --latent_steps 8 \\\n"
          f"      --max_new_tokens 32 --generate_bs 2 --print2 --debug")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-tiny", metavar="DIR",
                    help="write a tiny random Qwen3.5 checkpoint there and exit")
    ns = ap.parse_args()
    sys.exit(save_tiny(ns.save_tiny) if ns.save_tiny else main())
