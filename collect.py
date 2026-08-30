"""Collect finished runs into the table the paper reports.

arXiv:2511.20639 reports three numbers per cell, **Acc. / Token / Speed**
and its headline claims are ratios between them: 70.8-83.7% fewer output
tokens and 4x-4.3x faster end-to-end inference than text-based MAS. This reads
`runs/*/manifest.json` and lays those out, because doing it by hand across
three methods, four `--state_transfer` arms and nine tasks is where
"""

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional


def load_runs(root: str) -> List[Dict]:
    runs = []
    if not os.path.isdir(root):
        return runs
    for path in sorted(glob.glob(os.path.join(root, "**", "manifest.json"),
                                 recursive=True)):
        if not os.path.isfile(path):
            continue
        # results/superseded/ holds earlier attempts at configurations that were
        # re-run; including them here would put two rows for the same cell in
        # the table and let --compare pick the wrong reference.
        if "superseded" in path.replace("\\", "/").split("/"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception as exc:
            print(f"[skip] {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return runs


def row_of(m: Dict) -> Dict:
    args = m.get("args") or {}
    _offset = args.get("sample_offset") or 0
    tokens = m.get("tokens") or {}
    timing = m.get("timing") or {}
    gen = m.get("generation") or {}
    ci = m.get("accuracy_ci95") or {}
    channel = m.get("channel") or {}
    model_info = m.get("model") or {}
    mem = ((m.get("memory") or {}).get("peak") or {})
    peak = max((d.get("peak_reserved_mib") or 0) for d in mem.values()) if mem else None
    processed = m.get("processed") or 0
    return {
        "run_id": m.get("run_id"),
        "status": m.get("status"),
        "method": args.get("method"),
        "task": args.get("task"), "sample_offset": _offset,
        "model": args.get("model_name"),
        "transfer": args.get("state_transfer") if args.get("method") == "latent_mas" else "",
        "append": args.get("mamba_append") if args.get("method") == "latent_mas" else "",
        "latent_steps": args.get("latent_steps"),
        "bs": args.get("generate_bs"),
        "dtype": model_info.get("dtype", "").replace("torch.", ""),
        "attn": model_info.get("attn_implementation"),
        "n": processed,
        # --- the paper's three columns ---
        "acc": m.get("accuracy"),
        "acc_halfwidth": ci.get("half_width"),
        "token": tokens.get("generated_per_sample"),
        "token_total": tokens.get("generated_total"),
        "speed_sec": m.get("total_time_sec"),
        # --- the denominators and caveats behind them ---
        "sec_per_sample": m.get("time_per_sample_sec"),
        "steady_sec_per_sample": (timing.get("steady_sec_per_sample") or {}).get("median"),
        "prompt_per_sample": tokens.get("prompt_per_sample"),
        "latent_per_sample": tokens.get("latent_steps_per_sample"),
        "append_fwd_per_sample": tokens.get("append_forwards_per_sample"),
        "append_sec": (timing.get("phase_sec") or {}).get("append"),
        "latent_sec": (timing.get("phase_sec") or {}).get("latent"),
        "decode_sec": (timing.get("phase_sec") or {}).get("decode"),
        "cut_rate": gen.get("truncated_final_rate"),
        "peak_vram_mib": peak,
        "channel_mib": channel.get("transferred_total_mib_max"),
        "seed": args.get("seed"),
        "temperature": args.get("temperature"),
        "gpu": (((m.get("environment") or {}).get("gpus") or [{}])[0]).get("name"),
    }


def _f(v, spec: str = "", dash: str = "-") -> str:
    if v is None or v == "":
        return dash
    try:
        return format(v, spec) if spec else str(v)
    except (TypeError, ValueError):
        return str(v)


def print_table(rows: List[Dict], compare: Optional[str]) -> None:
    if not rows:
        print("no runs found")
        return

    # Baselines for the relative columns, keyed by everything that has to
    # match for a ratio between two runs to mean anything.
    base: Dict = {}
    if compare:
        for r in rows:
            if r["method"] == compare:
                base[(r["task"], r["model"])] = r

    header = ["model", "method", "transfer", "task", "n", "bs", "Acc.", "Token",
              "Speed(s)", "s/sample", "latent", "cut", "VRAM", "chan"]
    if compare:
        header += [f"Tok vs {compare}", f"Speed vs {compare}"]
    widths = [len(h) for h in header]

    body = []
    for r in rows:
        acc = "-"
        if r["acc"] is not None:
            hw = f"±{r['acc_halfwidth'] * 100:.1f}" if r["acc_halfwidth"] else ""
            acc = f"{r['acc'] * 100:.1f}{hw}"
        cut = "-"
        if r["cut_rate"] is not None:
            cut = f"{r['cut_rate'] * 100:.0f}%" + ("!" if r["cut_rate"] > 0 else "")
        flag = "" if r["status"] == "completed" else "*"
        line = [
            str(r["model"] or "?").split("/")[-1].replace("Qwen3.5-", ""),
            (r["method"] or "?") + flag,
            r["transfer"] or "-",
            r["task"] or "?",
            _f(r["n"]),
            _f(r["bs"]),
            acc,
            _f(r["token"], ".0f"),
            _f(r["speed_sec"], ".0f"),
            _f(r["sec_per_sample"], ".2f"),
            _f(r["latent_per_sample"], ".0f"),
            cut,
            _f(r["peak_vram_mib"] and r["peak_vram_mib"] / 1024, ".1f"),
            _f(r["channel_mib"], ".1f"),
        ]
        if compare:
            b = base.get((r["task"], r["model"]))
            for key in ("token", "speed_sec"):
                if b and b.get(key) and r.get(key):
                    line.append(f"{r[key] / b[key]:.2f}x")
                else:
                    line.append("-")
        body.append(line)
        widths = [max(w, len(c)) for w, c in zip(widths, line)]

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for line in body:
        print(fmt.format(*line))

    print()
    print("Acc. = % correct ± Wilson 95% half width · Token = generated tokens/sample")
    print("Speed(s) = end-to-end seconds for the run · chan = MiB crossing the widest")
    print("agent boundary · * = interrupted run · ! = answers hit --max_new_tokens")

    cut_rows = [r for r in rows if (r["cut_rate"] or 0) > 0]
    if cut_rows:
        print()
        print("Runs with clipped answers; Acc. and Token both partly measure the cap:")
        for r in cut_rows:
            print(f"  {r['run_id']}  {r['cut_rate'] * 100:.1f}% of finals hit "
                  f"--max_new_tokens")

    # A ratio between runs on different hardware or dtype is not a ratio
    # between methods, and this is exactly the mistake an L4 run and a later
    # A100 run in the same directory invite.
    for field, label in (("gpu", "GPU"), ("dtype", "dtype"), ("attn", "attention backend")):
        seen = {r[field] for r in rows if r[field]}
        if len(seen) > 1:
            print()
            print(f"NOTE: rows span more than one {label} ({', '.join(sorted(seen))}). "
                  f"Speed is not comparable across them.")


def main() -> int:
    # The table uses ± and ·, which a Windows console (cp1252) cannot encode.
    # run.py does the same at startup for the same reason.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_dir", default="results",
                    help="directory to search recursively for runs")
    ap.add_argument("--task", help="only this task")
    ap.add_argument("--offset", type=int, default=None,
                    help="only runs with this --sample_offset. Use 0 to see the\nreported slice alone, excluding runs made on a tuning slice; a table that\nmixes the two is reporting numbers the configuration was chosen on.")
    ap.add_argument("--method", help="only this method")
    ap.add_argument("--model", help="substring match on --model_name")
    ap.add_argument("--min_samples", type=int, default=1,
                    help="drop runs with fewer processed samples (smoke tests)")
    ap.add_argument("--completed_only", action="store_true")
    ap.add_argument("--compare", choices=["baseline", "text_mas", "latent_mas"],
                    help="add Token/Speed columns relative to this method, matched "
                         "on (task, model)")
    ap.add_argument("--csv", help="also write every field to this CSV")
    ap.add_argument("--sort", default="task",
                    help="field to sort by (task, method, acc, speed_sec, ...)")
    args = ap.parse_args()

    rows = [row_of(m) for m in load_runs(args.runs_dir)]
    rows = [r for r in rows if (r["n"] or 0) >= args.min_samples]
    if args.completed_only:
        rows = [r for r in rows if r["status"] == "completed"]
    if args.task:
        rows = [r for r in rows if r["task"] == args.task]
    if args.offset is not None:
        rows = [r for r in rows if int(r.get("sample_offset") or 0) == args.offset]
    if args.method:
        rows = [r for r in rows if r["method"] == args.method]
    if args.model:
        rows = [r for r in rows if args.model in (r["model"] or "")]

    key = args.sort
    rows.sort(key=lambda r: (r.get(key) is None, r.get(key), r.get("method") or "",
                             r.get("transfer") or ""))

    print_table(rows, args.compare)

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
