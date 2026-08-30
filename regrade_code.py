"""Re-grade code tasks with a typing preamble, and report what it changes.

The grader concatenates the model's extracted ```python block with the task's
test harness and executes it in a bare namespace. HumanEval+ and MBPP+ prompts
carry `from typing import List` in their signature stub, so a model that emits
only the function body fails with `NameError: name 'List' is not defined`. That
is an extraction artifact, not a wrong answer, and it does not hit methods
equally: on Qwen3.5-4B HumanEval+ it accounts for 16 of the single-agent
baseline's 30 failures and 0 of Latent-MAS's 13.

This re-executes only the samples that failed with a NameError, after
prepending the imports the prompt stub would have supplied, and prints both
scores so a paper can report the robust one or footnote the difference.

    python regrade_code.py            # report
    python regrade_code.py --write    # also write regraded_accuracy into manifests
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import extract_markdown_python_block                     # noqa: E402


def _exec(code, timeout=10):
    """Run `code` in a fresh interpreter. A subprocess rather than
    multiprocessing: the fork-based helper in utils.py cannot be pickled under
    Windows' spawn start method, and this has to run on either platform."""
    import subprocess, tempfile
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0, (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        return False, "TimeoutError"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

# What a HumanEval+/MBPP+ prompt stub would already have in scope.
PREAMBLE = (
    "from typing import List, Dict, Tuple, Set, Optional, Any, Union\n"
    "import math, re, itertools, collections, functools, heapq, string\n"
)

TASKS = ("humanevalplus", "mbppplus")


def regrade_run(d):
    mp, tp = os.path.join(d, "manifest.json"), os.path.join(d, "trace.jsonl")
    if not (os.path.isfile(mp) and os.path.isfile(tp)):
        return None
    man = json.load(open(mp, encoding="utf-8"))
    a = man.get("args") or {}
    if a.get("task") not in TASKS:
        return None

    n = correct = fixed = checked = 0
    for line in open(tp, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        if r.get("correct"):
            correct += 1
            continue
        if "is not defined" not in (r.get("error_msg") or ""):
            continue
        # gold is the test harness for code tasks
        pred = extract_markdown_python_block(r.get("raw_prediction") or "")
        gold = r.get("gold") or ""
        if not pred or not gold:
            continue
        checked += 1
        ok, _ = _exec(PREAMBLE + pred + "\n" + gold, timeout=10)
        if ok:
            fixed += 1
            correct += 1
    return dict(task=a.get("task"), method=a.get("method"),
                model=str(a.get("model_name", "")).split("/")[-1],
                n=n, before=man.get("accuracy"), after=correct / n if n else 0.0,
                fixed=fixed, checked=checked, path=mp, man=man)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--write", action="store_true",
                    help="store regraded_accuracy in each manifest")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "**", ""), recursive=True)):
        r = regrade_run(d)
        if r and r["checked"]:
            rows.append(r)

    if not rows:
        print("nothing to regrade")
        return

    print(f"{'model':14s} {'task':14s} {'method':11s} {'n':>4s} {'before':>7s} "
          f"{'after':>7s} {'delta':>7s} {'fixed':>6s}")
    print("-" * 76)
    for r in sorted(rows, key=lambda r: (r["model"], r["task"], r["method"])):
        d = (r["after"] - r["before"]) * 100
        flag = "  <<<" if d >= 5 else ""
        print(f"{r['model']:14s} {r['task']:14s} {r['method']:11s} {r['n']:>4d} "
              f"{r['before']*100:>6.1f}% {r['after']*100:>6.1f}% {d:>+6.1f} "
              f"{r['fixed']:>3d}/{r['checked']:<3d}{flag}")
        if args.write:
            r["man"]["regraded_accuracy"] = r["after"]
            r["man"]["regrade_note"] = ("re-executed NameError failures with a "
                                        "typing/import preamble; see regrade_code.py")
            with open(r["path"], "w", encoding="utf-8", newline="\n") as f:
                json.dump(r["man"], f, ensure_ascii=False, indent=1)
    if args.write:
        print("\nmanifests updated with regraded_accuracy")


if __name__ == "__main__":
    main()
