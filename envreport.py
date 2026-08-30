"""Emit the reproducibility table for a run, as markdown or LaTeX.

Reads a run's manifest.json and prints every field a reproducibility appendix
needs, so the paper quotes recorded values rather than transcribed ones.

    python envreport.py results/main/gsm8k_latent_mas
    python envreport.py results/main/gsm8k_latent_mas --latex
    python envreport.py --all            # consistency check across every run
"""

import argparse
import glob
import json
import os
import sys

ROWS = [
    ("Model", "model.name"),
    ("Revision", "model.chat.revision"),
    ("Weights dtype", "model.dtype"),
    ("Attention impl.", "model.attn_implementation"),
    ("Layers (total / linear)", "@layers"),
    ("Hidden size / vocab", "@hidden"),
    ("KV cache dtype", "channel.dtypes.kv"),
    ("Recurrent state dtype", "channel.dtypes.recurrent"),
    ("Conv state dtype", "channel.dtypes.conv"),
    ("Thinking mode", "model.chat.chat_thinking"),
    ("Generation tail (used)", "@tail"),
    ("Chat template sha256", "model.chat.template_sha256"),
    ("State continuation: recurrent", "@cont_rec"),
    ("State continuation: conv", "@cont_conv"),
    ("Append mode", "model.append_mode"),
    ("", None),
    ("Method", "args.method"),
    ("Task / split", "@task"),
    ("Samples (offset, n)", "@samples"),
    ("Batch size", "args.generate_bs"),
    ("Latent steps", "args.latent_steps"),
    ("Latent realignment", "args.latent_space_realign"),
    ("State transfer", "args.state_transfer"),
    ("Latent turn / visibility", "@framing"),
    ("Decoding", "@decoding"),
    ("max_new_tokens", "args.max_new_tokens"),
    ("Seed", "args.seed"),
    ("", None),
    ("Python", "environment.python"),
    ("PyTorch", "environment.torch"),
    ("Transformers", "environment.transformers"),
    ("CUDA", "environment.cuda"),
    ("causal-conv1d", "@ccv"),
    ("flash-linear-attention", "@fla"),
    ("GPU", "@gpu"),
    ("", None),
    ("Warm-up batches discarded", "timing.warmup_batches_discarded"),
    ("Timed batches", "timing.batches"),
    ("Wall clock (s)", "total_time_sec"),
    ("s / sample (steady)", "timing.steady_sec_per_sample"),
    ("Accuracy", "@acc"),
    ("Truncated finals", "@cut"),
]


def dig(man, path):
    cur = man
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def derived(man, key):
    a = man.get("args") or {}
    m = man.get("model") or {}
    e = man.get("environment") or {}
    g = man.get("generation") or {}
    chat = m.get("chat") or {}
    cont = m.get("state_continuation") or {}

    if key == "layers":
        return f"{m.get('num_hidden_layers')} / {m.get('linear_attention_layers')}"
    if key == "hidden":
        return f"{m.get('hidden_size')} / {m.get('vocab_size')}"
    if key == "tail":
        return repr(chat.get("generation_tail_" + str(chat.get("chat_thinking"))))
    if key in ("cont_rec", "cont_conv"):
        ch = cont.get("recurrent" if key == "cont_rec" else "conv")
        if not ch:
            legacy = m.get("append_carries_state")
            return ("(not recorded; append_carries_state="
                    f"{legacy})" if key == "cont_conv" else f"carries={legacy}")
        return (f"carries={ch['carries']}  split/reset={ch['ratio']:.1%} "
                f"(split {ch['split']:.2e}, reset {ch['reset']:.2e})")
    if key == "task":
        return f"{a.get('task')} / {a.get('split')}"
    if key == "samples":
        return f"offset {a.get('sample_offset')}, n={man.get('processed')}"
    if key == "framing":
        return f"{a.get('latent_turn')} / {a.get('latent_visibility')}"
    if key == "decoding":
        t = a.get("temperature")
        if t in (0, 0.0):
            return "do_sample=False (greedy); temperature/top_p not passed"
        return f"do_sample=True, temperature={t}, top_p={a.get('top_p')}"
    if key == "ccv":
        v = e.get("causal-conv1d")
        av = e.get("causal_conv1d_available")
        return f"{v or 'not installed'}  (available={av})"
    if key == "fla":
        v = e.get("flash-linear-attention") or e.get("fla-core")
        av = e.get("flash_linear_attention_available")
        return f"{v or 'not installed'}  (available={av})"
    if key == "gpu":
        gpus = e.get("gpus") or []
        if not gpus:
            return None
        return f"{len(gpus)} x {gpus[0].get('name')} ({gpus[0].get('vram_gb')} GiB)"
    if key == "acc":
        ci = man.get("accuracy_ci95") or {}
        return (f"{man.get('accuracy', 0) * 100:.1f}% "
                f"+/- {ci.get('half_width', 0) * 100:.1f} (Wilson 95%)")
    if key == "cut":
        return f"{g.get('truncated_final')} / {man.get('processed')}"
    return None


def value(man, path):
    if path.startswith("@"):
        return derived(man, path[1:])
    return dig(man, path)


def render(man, latex):
    out = []
    if latex:
        out.append(r"\begin{tabular}{ll}\hline")
    for label, path in ROWS:
        if path is None:
            out.append(r"\hline" if latex else "")
            continue
        v = value(man, path)
        v = "(not recorded)" if v is None else str(v)
        if latex:
            esc = lambda s: (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
                             .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))
            out.append(f"{esc(label)} & \\texttt{{{esc(v)}}} \\\\")
        else:
            out.append(f"| {label} | `{v}` |" if label else "|  |  |")
    if latex:
        out.append(r"\hline\end{tabular}")
    elif out:
        out.insert(0, "| Field | Value |\n|---|---|")
    return "\n".join(out)


def check_all(root):
    """Flag anything that must be identical across runs but is not."""
    mans = []
    for p in sorted(glob.glob(os.path.join(root, "**", "manifest.json"), recursive=True)):
        with open(p, encoding="utf-8") as f:
            mans.append((os.path.relpath(p), json.load(f)))
    if not mans:
        print(f"no manifests under {root}/")
        return
    watch = ["model.name", "model.dtype", "environment.torch", "environment.transformers",
             "environment.cuda", "@gpu", "model.chat.template_sha256",
             "model.chat.chat_thinking", "args.generate_bs", "args.temperature",
             "args.seed"]
    print(f"{len(mans)} manifests\n")
    for path in watch:
        vals = {}
        for name, m in mans:
            vals.setdefault(str(value(m, path)), []).append(name)
        flag = "  <-- DIFFERS" if len(vals) > 1 else ""
        print(f"{path:34s} {len(vals)} distinct{flag}")
        if len(vals) > 1:
            for v, names in sorted(vals.items(), key=lambda kv: -len(kv[1])):
                print(f"    {v:<40s} {len(names)} run(s)  e.g. {names[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="a run directory or its manifest.json")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="consistency check across every run under --root")
    ap.add_argument("--root", default="results")
    args = ap.parse_args()

    if args.all or not args.run:
        check_all(args.root)
        return
    p = args.run
    if os.path.isdir(p):
        p = os.path.join(p, "manifest.json")
    if not os.path.isfile(p):
        sys.exit(f"no manifest at {p}")
    with open(p, encoding="utf-8") as f:
        print(render(json.load(f), args.latex))


if __name__ == "__main__":
    main()
