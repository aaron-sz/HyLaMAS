"""Choose each method's configuration from a held-out tuning slice. READ-ONLY.

    among configurations whose final-answer truncation rate is under 2%,
    take the highest accuracy; break ties by fewer generated tokens.
"""

import argparse
import glob
import json
import os
import sys

CUT_GATE = 0.02


def _runs(root, task, offset):
    out = []
    for mp in sorted(glob.glob(os.path.join(root, "**", "manifest.json"),
                               recursive=True)):
        d = os.path.dirname(mp)
        if not os.path.isfile(mp):
            continue
        try:
            m = json.load(open(mp, encoding="utf-8"))
        except Exception:
            continue
        a = m.get("args") or {}
        if a.get("task") != task:
            continue
        if int(a.get("sample_offset") or 0) != int(offset):
            continue
        if m.get("status") != "completed":
            continue
        gen = m.get("generation") or {}
        tok = m.get("tokens") or {}
        out.append({
            "dir": os.path.basename(d),
            "method": a.get("method"),
            "turn": a.get("latent_turn", "-"),
            "vis": a.get("latent_visibility", "-"),
            "steps": a.get("latent_steps"),
            "chat": a.get("chat_thinking", "-"),
            "judger": a.get("judger_thinking", "-"),
            "acc": m.get("accuracy") or 0.0,
            "cut": gen.get("truncated_final_rate"),
            "tok": tok.get("generated_per_sample") or 0.0,
        })
    return out


def pick(rows):
    """Fixed rule: truncation gate first, then accuracy, then fewer tokens."""
    if not rows:
        return None, "no completed runs on the tuning slice"
    clean = [r for r in rows if (r["cut"] is not None and r["cut"] < CUT_GATE)]
    pool, note = (clean, "") if clean else (
        rows, f"WARNING: no configuration met the {CUT_GATE:.0%} truncation gate; "
              f"selected on accuracy alone and the reported numbers will be "
              f"partly measuring --max_new_tokens")
    best = sorted(pool, key=lambda r: (-r["acc"], r["tok"]))[0]
    return best, note


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results")
    ap.add_argument("--task", required=True)
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--shell", action="store_true", help="emit KEY=value only")
    ns = ap.parse_args()

    rows = _runs(ns.runs, ns.task, ns.offset)
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    if not ns.shell:
        print(f"tuning slice: task={ns.task} offset={ns.offset}  "
              f"({len(rows)} completed runs)\n")
        print(f"{'method':11s} {'turn':8s} {'vis':12s} {'chat':5s} {'judger':8s} "
              f"{'steps':>5s} {'acc':>7s} {'cut':>7s} {'tok':>8s}")
        print("-" * 82)
        for m in sorted(by_method):
            for r in sorted(by_method[m], key=lambda x: -x["acc"]):
                cut = "-" if r["cut"] is None else f"{r['cut']:.1%}"
                gate = "" if (r["cut"] is not None and r["cut"] < CUT_GATE) else "  !cut"
                print(f"{r['method']:11s} {str(r['turn']):8s} {str(r['vis']):12s} "
                      f"{str(r['chat']):5s} {str(r['judger']):8s} {str(r['steps']):>5s} "
                      f"{r['acc']:7.4f} {cut:>7s} {r['tok']:8.1f}{gate}")
        print()

    out = {}
    for m in ("baseline", "text_mas", "latent_mas"):
        best, note = pick(by_method.get(m, []))
        if best is None:
            if not ns.shell:
                print(f"{m:11s} -> {note}")
            continue
        if not ns.shell:
            print(f"{m:11s} -> {best['dir']}  acc={best['acc']:.4f} "
                  f"cut={'-' if best['cut'] is None else format(best['cut'], '.1%')}")
            if note:
                print(f"{'':11s}    {note}")
        key = m.upper()
        out[f"SEL_{key}_CHAT"] = best["chat"]
        if m == "latent_mas":
            out["SEL_LATENT_TURN"] = best["turn"]
            out["SEL_LATENT_VIS"] = best["vis"]
            out["SEL_LATENT_STEPS"] = str(best["steps"])
            out["SEL_LATENT_JUDGER"] = best["judger"]

    if ns.shell:
        for k, v in out.items():
            print(f"{k}={v}")
    else:
        print("\nshell form:")
        for k, v in out.items():
            print(f"  {k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
