"""Paired comparison between two runs on the same task. READ-ONLY.

92.1% and 94.2% on n=1319 have overlapping intervals and look inconclusive --
but if they disagree on only 60 questions and 44 of those go one way, McNemar's
exact test on the discordant pairs settles it easily. The join key is
`sample_id`, the question hash already written into every trace row.
"""

import argparse
import glob
import json
import os
import sys
from math import comb


def _runs(root="results"):
    out = []
    for mp in sorted(glob.glob(os.path.join(root, "**", "manifest.json"),
                               recursive=True)):
        d = os.path.dirname(mp)
        tp = os.path.join(d, "trace.jsonl")
        if not (os.path.isfile(mp) and os.path.isfile(tp)):
            continue
        try:
            m = json.load(open(mp, encoding="utf-8"))
        except Exception:
            continue
        out.append((d, m))
    return out


def _rows(run_dir):
    by_id = {}
    for line in open(os.path.join(run_dir, "trace.jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        key = r.get("sample_id") or r.get("question", "")[:120]
        if key:
            by_id[key] = r
    return by_id


def pick(runs, task, method):
    """Most recent completed run for this task+method."""
    cands = [(d, m) for d, m in runs
             if (m.get("args") or {}).get("task") == task
             and (m.get("args") or {}).get("method") == method]
    if not cands:
        return None, None
    cands.sort(key=lambda dm: dm[1].get("started", ""))
    return cands[-1]


def mcnemar_exact(b, c):
    """Two-sided exact binomial p on the discordant pairs (b vs c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results")
    ap.add_argument("--task")
    ap.add_argument("--a", help="method A (the reference)")
    ap.add_argument("--b", help="method B (the one being argued for)")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many questions B lost that A got right")
    ap.add_argument("--list", action="store_true", help="list available runs and exit")
    ns = ap.parse_args()

    runs = _runs(ns.runs)
    if ns.list or not (ns.task and ns.a and ns.b):
        print(f"{'run':38s} {'task':10s} {'method':11s} {'n':>5s} {'acc':>7s} {'status':10s}")
        for d, m in runs:
            a = m.get("args") or {}
            print(f"{os.path.basename(d)[:38]:38s} {str(a.get('task'))[:10]:10s} "
                  f"{str(a.get('method'))[:11]:11s} {m.get('processed',0):5d} "
                  f"{m.get('accuracy',0):7.4f} {m.get('status','?'):10s}")
        if not ns.list:
            print("\nneed --task, --a and --b", file=sys.stderr)
            return 2
        return 0

    da, ma = pick(runs, ns.task, ns.a)
    db, mb = pick(runs, ns.task, ns.b)
    if da is None or db is None:
        print(f"missing run: {ns.a if da is None else ns.b} on {ns.task}", file=sys.stderr)
        return 1

    ra, rb = _rows(da), _rows(db)
    keys = sorted(set(ra) & set(rb))
    if not keys:
        print("the two runs share no sample_ids", file=sys.stderr)
        return 1

    both = aonly = bonly = neither = 0
    lost, gained = [], []
    for k in keys:
        ca, cb = bool(ra[k].get("correct")), bool(rb[k].get("correct"))
        if ca and cb:
            both += 1
        elif ca and not cb:
            aonly += 1
            lost.append(k)
        elif cb and not ca:
            bonly += 1
            gained.append(k)
        else:
            neither += 1

    n = len(keys)
    acc_a, acc_b = (both + aonly) / n, (both + bonly) / n
    p = mcnemar_exact(aonly, bonly)

    print(f"task {ns.task}   paired on {n} questions "
          f"(of {len(ra)} / {len(rb)} rows)")
    print(f"  A = {ns.a:11s}  {os.path.basename(da)}")
    print(f"  B = {ns.b:11s}  {os.path.basename(db)}")
    print()
    print(f"                     B right   B wrong")
    print(f"    A right          {both:7d}   {aonly:7d}")
    print(f"    A wrong          {bonly:7d}   {neither:7d}")
    print()
    print(f"  acc A = {acc_a:.4f}    acc B = {acc_b:.4f}    delta = {acc_b - acc_a:+.4f}")
    print(f"  discordant: B fixed {bonly}, B broke {aonly}")
    print(f"  McNemar exact two-sided p = {p:.3g}"
          f"   {'(significant at 0.05)' if p < 0.05 else '(NOT significant)'}")
    print()
    if aonly and bonly:
        print(f"  Note both directions: B did not merely add {bonly - aonly} wins, it")
        print(f"  traded {aonly} losses for {bonly} wins. A method that breaks {aonly}")
        print(f"  previously-correct answers is doing something, even when the net")
        print(f"  delta is small -- worth inspecting with --show.")

    if ns.show:
        print(f"\n--- {min(ns.show, len(lost))} questions A got right and B got wrong")
        for k in lost[: ns.show]:
            r = rb[k]
            print(f"\n  gold={r.get('gold')!r}  B_pred={r.get('prediction')!r}  "
                  f"A_pred={ra[k].get('prediction')!r}  B_truncated={r.get('truncated_any')}")
            print(f"  Q: {str(r.get('question',''))[:200]}")
            print(f"  B: {str(r.get('raw_prediction',''))[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
