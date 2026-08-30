"""Why is a run's Token column high and its Acc column low?
"""

import argparse
import glob
import json
import os
import re
import sys


def _rows(run_dir):
    path = os.path.join(run_dir, "trace.jsonl")
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _judger(row):
    for a in reversed(row.get("agents") or []):
        if a.get("role") == "judger":
            return a
    return None


def analyse(run_dir):
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    args = m.get("args") or {}
    rows = _rows(run_dir)

    n = len(rows) or 1
    cut = boxed = nobox = placeholder = 0
    lens, cut_lens, ok_lens = [], [], []
    cache_at_judge = []
    for r in rows:
        j = _judger(r)
        if j is None:
            continue
        tok = int(j.get("output_token_count") or 0)
        lens.append(tok)
        text = j.get("output") or r.get("raw_prediction") or ""
        has_box = bool(re.search(r"\\boxed\{", text))
        boxed += int(has_box)
        nobox += int(not has_box)
        # The grader takes the LAST \boxed{...}. A judger that quotes its own
        # instructions late in a ramble ("...output the final answer inside
        # \boxed{YOUR_FINAL_ANSWER}") therefore gets graded on the placeholder,
        # not on the number it computed correctly 1500 tokens earlier. Empty
        # boxes from "put the final number in `\boxed{}`" do the same thing.
        boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxes:
            last = boxes[-1].strip()
            if not re.search(r"[-+]?\d", last):
                placeholder += 1
        if j.get("truncated"):
            cut += 1
            cut_lens.append(tok)
        else:
            ok_lens.append(tok)
        for a in r.get("agents") or []:
            if a.get("role") == "refiner" and a.get("cache_len") is not None:
                cache_at_judge.append(a["cache_len"])

    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return {
        "run": os.path.basename(run_dir),
        "offset": int(args.get("sample_offset") or 0),
        "method": args.get("method"),
        "model": str(args.get("model_name", "")).split("/")[-1],
        "task": args.get("task"),
        "turn": args.get("latent_turn", "-"),
        "vis": args.get("latent_visibility", "-"),
        "transfer": args.get("state_transfer", "-"),
        "steps": args.get("latent_steps"),
        "realign": bool(args.get("latent_space_realign")),
        "temp": args.get("temperature"),
        "cap": args.get("max_new_tokens"),
        "n": len(lens),
        "acc": m.get("accuracy"),
        "tok": mean(lens),
        "tok_cut": mean(cut_lens),
        "tok_ok": mean(ok_lens),
        "cut_rate": cut / max(1, len(lens)),
        "nobox_rate": nobox / max(1, len(lens)),
        "placeholder_rate": placeholder / max(1, len(lens)),
        "cache": mean(cache_at_judge),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results")
    ap.add_argument("--task", help="only runs on this task")
    ap.add_argument("--offset", type=int, default=None,
                    help="only runs with this --sample_offset (0 = the "
                         "reported slice, excluding tuning runs)")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many judger outputs per run")
    ns = ap.parse_args()

    dirs = sorted({os.path.dirname(p) for p in glob.glob(
        os.path.join(ns.runs, "**", "manifest.json"), recursive=True)})
    found = [a for a in (analyse(d) for d in dirs) if a]
    if ns.task:
        found = [a for a in found if a["task"] == ns.task]
    if ns.offset is not None:
        found = [a for a in found if a["offset"] == ns.offset]
    if not found:
        print(f"no runs with a manifest under {ns.runs}/", file=sys.stderr)
        return 1

    print(f"{'run':26s} {'turn/vis':18s} {'st':4s} {'acc':>6s} {'tok':>7s} "
          f"{'tok|ok':>7s} {'tok|cut':>7s} {'CUT%':>6s} {'NOBOX%':>7s} {'PLCHLD%':>8s} {'cache':>6s}")
    print("-" * 113)
    for a in found:
        flag = "  <<<" if (a["cut_rate"] > 0.02 or a["nobox_rate"] > 0.05
                           or a["placeholder_rate"] > 0.02) else ""
        print(f"{a['run'][:26]:26s} {a['turn'] + '/' + a['vis']:18s} "
              f"{str(a['transfer'])[:4]:4s} "
              f"{(a['acc'] or 0):6.3f} {a['tok']:7.1f} {a['tok_ok']:7.1f} "
              f"{a['tok_cut']:7.1f} {a['cut_rate']:6.1%} {a['nobox_rate']:7.1%} "
              f"{a['placeholder_rate']:8.1%} {a['cache']:6.0f}{flag}")

    print()
    print("how to read this")
    print("  CUT%    judger hit --max_new_tokens. Anything above ~2% means the Acc and")
    print("          Token columns are both partly measuring the cap. Raise the cap and")
    print("          re-run before believing either number.")
    print("  NOBOX%  judger never emitted \\boxed{}. Those samples were graded on the last")
    print("          number appearing anywhere in the text -- often a figure from its own")
    print("          scratch work. High here with low Acc is an extraction failure, not a")
    print("          reasoning failure.")
    print("  PLCHLD% judger emitted a \boxed{} whose contents have no digits -- almost")
    print("          always because it quoted the prompt's own \boxed{YOUR_FINAL_ANSWER}")
    print("          late in a ramble. The grader takes the LAST box, so these are graded")
    print("          on the placeholder even when the right number appears earlier.")
    print("  tok|ok  mean output tokens over samples that finished. Compare against a")
    print("          `--method baseline` run on the SAME model: if latent_mas is not")
    print("          well below it, the judger is re-deriving rather than inheriting,")
    print("          i.e. it is ignoring the shared workspace.")
    print("  cache   KV positions in the workspace when the judger started.")

    if ns.show:
        for d, a in zip(dirs, found):
            print(f"\n{'=' * 76}\n{a['run']}  ({a['turn']}/{a['vis']})\n{'=' * 76}")
            for r in _rows(d)[: ns.show]:
                j = _judger(r) or {}
                text = (j.get("output") or r.get("raw_prediction") or "")
                print(f"--- gold={r.get('gold')} pred={r.get('prediction')} "
                      f"correct={r.get('correct')} tokens={j.get('output_token_count')} "
                      f"truncated={j.get('truncated')}")
                print(text[:1200] + (" …[clipped]" if len(text) > 1200 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
