"""Shrink trace.jsonl files for release without losing anything reported.

Three fields dominate the size and none of them carry information that the
tables, the paired tests or a re-grade need:

  solution            identical to `gold` on every task except gsm8k, where it
                      is the reference derivation and is kept
  agents[].input      the agent's full rendered prompt, reconstructible from
                      prompts.py plus the question
  agents[].input_ids  token ids for the same text

Everything else stays: sample_id, correct, prediction, gold, raw_prediction,
per-agent token counts and timings, cache_len, state_norm, channel bytes,
truncation flags. Measured on the scale sweep this halves 147 MB to about 70 MB.

    python slim_traces.py --check
    python slim_traces.py
"""

import argparse
import glob
import json
import os

DROP_ROW = ("solution",)
DROP_AGENT = ("input", "input_ids", "input_tokens")


def slim_row(row, keep_solution):
    out = {k: v for k, v in row.items()
           if k not in DROP_ROW or (k == "solution" and keep_solution)}
    if "agents" in row:
        out["agents"] = [{k: v for k, v in a.items() if k not in DROP_AGENT}
                         for a in row["agents"]]
    return out


def task_of(trace_path):
    man = os.path.join(os.path.dirname(trace_path), "manifest.json")
    if not os.path.isfile(man):
        return None
    try:
        with open(man, encoding="utf-8") as f:
            return (json.load(f).get("args") or {}).get("task")
    except Exception:
        return None


def process(path, write):
    # gsm8k's `solution` is the worked derivation, not a copy of `gold`
    keep_solution = task_of(path) == "gsm8k"
    before = os.path.getsize(path)
    out_lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out_lines.append(json.dumps(slim_row(json.loads(line), keep_solution),
                                            ensure_ascii=False))
    body = "\n".join(out_lines) + "\n"
    after = len(body.encode("utf-8"))
    if write and after < before:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.replace(tmp, path)
    return before, after, keep_solution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "trace.jsonl"),
                             recursive=True))
    tb = ta = 0
    for p in paths:
        b, a, kept = process(p, write=not args.check)
        tb += b
        ta += a
        print(f"{os.path.relpath(p):64s} {b/2**20:7.1f} -> {a/2**20:6.1f} MB"
              f"{'  (kept solution)' if kept else ''}")
    if paths:
        verb = "would save" if args.check else "saved"
        print(f"\n{len(paths)} traces: {tb/2**20:.1f} -> {ta/2**20:.1f} MB, "
              f"{verb} {(1 - ta/tb)*100:.0f}%")
    else:
        print(f"no trace.jsonl found under {args.root}/")


if __name__ == "__main__":
    main()
