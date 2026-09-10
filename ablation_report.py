"""Paper-ready tables for the two ablations. READ-ONLY.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from compare import mcnemar_exact
except Exception:                                            # pragma: no cover
    def mcnemar_exact(b, c):
        return float("nan")

AGENTS = ("Planner", "Critic", "Refiner")


def _trace(run_dir):
    p = os.path.join(run_dir, "trace.jsonl")
    if not os.path.isfile(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _load(runs_dir, task, offset, method="latent_mas"):
    rows = []
    for mp in sorted(glob.glob(os.path.join(runs_dir, "**", "manifest.json"),
                               recursive=True)):
        d = os.path.dirname(mp)
        if not os.path.isfile(mp):
            continue
        try:
            m = json.load(open(mp, encoding="utf-8"))
        except Exception:
            continue
        a = m.get("args") or {}
        if a.get("method") != method or a.get("task") != task:
            continue
        if int(a.get("sample_offset") or 0) != offset:
            continue
        if m.get("status") != "completed":
            continue
        tr = _trace(d)
        # per-agent state fingerprints, averaged over samples
        norms, caches = {}, {}
        for name in AGENTS:
            vs = [ag.get("state_norm") for r in tr for ag in (r.get("agents") or [])
                  if ag.get("name") == name and ag.get("state_norm") is not None]
            cs = [ag.get("cache_len") for r in tr for ag in (r.get("agents") or [])
                  if ag.get("name") == name and ag.get("cache_len") is not None]
            norms[name] = (sum(vs) / len(vs)) if vs else None
            caches[name] = (sum(cs) / len(cs)) if cs else None
        ch = m.get("channel") or {}
        gen = m.get("generation") or {}
        tok = m.get("tokens") or {}
        tim = m.get("timing") or {}
        rows.append({
            "dir": os.path.basename(d), "args": a, "n": m.get("processed") or 0,
            "acc": m.get("accuracy") or 0.0,
            "ci": (m.get("accuracy_ci95") or {}).get("half_width") or 0.0,
            "tok": tok.get("generated_per_sample") or 0.0,
            "latent_sec": (tim.get("phase_sec") or {}).get("latent"),
            "sec": m.get("time_per_sample_sec") or 0.0,
            "cut": gen.get("truncated_final_rate"),
            "kv_mib": channel_bytes(a, caches)[0],
            "state_mib": channel_bytes(a, caches)[1],
            "kv_mib_recorded": ch.get("transferred_kv_mib_max"),
            "state_mib_recorded": ch.get("transferred_state_mib_max"),
            "norms": norms, "caches": caches,
            "correct_by_id": {r.get("sample_id"): bool(r.get("correct")) for r in tr},
        })
    return rows


# Measured channel constants for Qwen3.5-9B. Verified across 1824 agent
# boundaries: 32.000 KiB of attention KV per cached position per sequence
# (1808/1824 exact, the rest off by the turn-closing tokens), and 49.500 MiB of
# Gated DeltaNet state per sequence (1824/1824) = 48.0 fp32 recurrent + 1.5
# bf16 conv, invariant in sequence length.
KV_KIB_PER_POSITION = 32.0
STATE_MIB_PER_SEQUENCE = 49.5


def channel_bytes(args, caches):
    """(KV MiB, state MiB) crossing the last inter-agent boundary, for a batch.

    Derived rather than read from `channel.transferred_*_mib_max`. That field
    is a max over samples while the rest of the table reports means, and on
    these runs it predates the `carries_state` gate in hybrid.state_bytes, so
    an arm that stripped the DeltaNet state still reports a full-size state
    crossing. Gating on --state_transfer is exact because the arm defines what
    survives the boundary, and `state_norm` in the same trace agrees: null
    wherever the state was dropped.
    """
    st = args.get("state_transfer", "both")
    bs = args.get("generate_bs") or 1
    last = caches.get("Refiner") or caches.get("Critic") or caches.get("Planner") or 0
    kv = (last * KV_KIB_PER_POSITION * bs / 1024.0) if st in ("both", "kv_only") else 0.0
    state = (STATE_MIB_PER_SEQUENCE * bs) if st in ("both", "state_only") else 0.0
    return kv, state


def _paired(a, b):
    """(B fixed, B broke, p) between two runs over shared sample_ids."""
    keys = set(a["correct_by_id"]) & set(b["correct_by_id"])
    fixed = sum(1 for k in keys if b["correct_by_id"][k] and not a["correct_by_id"][k])
    broke = sum(1 for k in keys if a["correct_by_id"][k] and not b["correct_by_id"][k])
    return fixed, broke, mcnemar_exact(broke, fixed)


def _gate(mode, rows):
    """The invariants each arm must satisfy to be the arm it claims to be."""
    problems = []
    if mode != "channel":
        return problems
    for r in rows:
        st = r["args"].get("state_transfer")
        has_state = any(v is not None for v in r["norms"].values())
        kv = r["kv_mib"]
        cache_grows = None
        cl = [r["caches"][n] for n in AGENTS if r["caches"][n] is not None]
        if len(cl) >= 2:
            cache_grows = all(y > x for x, y in zip(cl, cl[1:]))
        if st in ("both", "state_only") and not has_state:
            problems.append(f"{st}: state_norm is null -- the recurrent state is NOT crossing")
        if st in ("kv_only", "none") and has_state:
            problems.append(f"{st}: state_norm is non-null -- the state was NOT dropped")
        if st == "state_only" and (kv or 0) > 0:
            problems.append(f"state_only: derived kv_mib={kv} -- KV was NOT dropped")
        if st == "state_only" and any((c or 0) > 0 for c in cl):
            problems.append(f"state_only: cache_len {cl} -- expected 0 at every boundary")
        if st in ("both", "kv_only") and cache_grows is False:
            problems.append(f"{st}: cache_len {cl} does not grow monotonically")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["channel", "steps"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--runs", default="results")
    # An ablation must vary ONE dimension. The two sweeps overlap -- every
    # steps-sweep run is also state_transfer=both, and every channel-arm run
    # is also at one latent_steps -- so each mode pins the other dimension or
    # the table silently gains duplicate rows that differ in something else.
    ap.add_argument("--steps", type=int, default=None,
                    help="channel mode: hold --latent_steps at this value")
    ap.add_argument("--transfer", default=None,
                    help="steps mode: hold --state_transfer at this value")
    ap.add_argument("--md", help="also write a markdown table here")
    ap.add_argument("--csv", help="also write a CSV here")
    ns = ap.parse_args()

    rows = _load(ns.runs, ns.task, ns.offset)
    key = "state_transfer" if ns.mode == "channel" else "latent_steps"
    rows = [r for r in rows if r["args"].get(key) is not None]
    if ns.steps is not None:
        rows = [r for r in rows if int(r["args"].get("latent_steps") or -1) == ns.steps]
    if ns.transfer is not None:
        rows = [r for r in rows if r["args"].get("state_transfer") == ns.transfer]
    # Last line of defence: if the pinning flags were not passed and an arm is
    # still duplicated, keep the most recent run per arm rather than printing
    # rows that differ in an unstated dimension.
    seen, unique = {}, []
    for r in rows:
        seen[str(r["args"][key])] = r
    unique = list(seen.values())
    if len(unique) != len(rows):
        print(f"note: collapsed {len(rows)} runs to {len(unique)} distinct "
              f"{key} values (most recent per value); pass --steps/--transfer "
              f"to pin the other dimension explicitly", file=sys.stderr)
    rows = unique
    if not rows:
        print(f"no completed latent_mas runs for task={ns.task} offset={ns.offset}",
              file=sys.stderr)
        return 1

    if ns.mode == "channel":
        order = {"both": 0, "kv_only": 1, "state_only": 2, "none": 3}
        rows.sort(key=lambda r: order.get(r["args"][key], 9))
        ref = next((r for r in rows if r["args"][key] == "both"), rows[0])
    else:
        rows.sort(key=lambda r: int(r["args"][key]))
        ref = min(rows, key=lambda r: abs(int(r["args"][key]) - 40))

    label = "arm" if ns.mode == "channel" else "m"
    print(f"\n{ns.mode} ablation  --  task={ns.task}  offset={ns.offset}  "
          f"n={rows[0]['n']}  (reference: {ref['args'][key]})\n")
    hdr = (f"{label:>10s}  {'Acc.':>13s}  {'Token':>7s}  {'s/samp':>7s}  {'CUT%':>6s}  "
           f"{'KV MiB':>7s}  {'St MiB':>7s}  {'sn:P/C/R':>22s}  {'vs ref (McNemar)':>22s}")
    print(hdr)
    print("-" * len(hdr))
    md = [f"| {label} | Acc. | Token | s/sample | CUT% | KV MiB | State MiB | "
          f"state_norm P/C/R | vs ref |", "|---|---|---|---|---|---|---|---|---|"]
    csv = [f"{label},n,acc,ci95,token,sec_per_sample,cut_rate,kv_mib,state_mib,"
           f"sn_planner,sn_critic,sn_refiner,fixed,broke,p"]

    for r in rows:
        v = r["args"][key]
        sn = "/".join("-" if r["norms"][a] is None else f"{r['norms'][a]:.4f}"
                      for a in AGENTS)
        if r is ref:
            cmp_s, fixed, broke, p = "(reference)", "", "", ""
        else:
            fixed, broke, p = _paired(ref, r)
            cmp_s = f"+{fixed}/-{broke}  p={p:.3g}"
        cut = "-" if r["cut"] is None else f"{r['cut']:.1%}"
        print(f"{str(v):>10s}  {r['acc']*100:6.2f}+/-{r['ci']*100:4.1f}  "
              f"{r['tok']:7.1f}  {r['sec']:7.2f}  {cut:>6s}  "
              f"{(r['kv_mib'] or 0):7.1f}  {(r['state_mib'] or 0):7.1f}  "
              f"{sn:>22s}  {cmp_s:>22s}")
        md.append(f"| {v} | {r['acc']*100:.2f} ± {r['ci']*100:.1f} | {r['tok']:.0f} | "
                  f"{r['sec']:.2f} | {cut} | {(r['kv_mib'] or 0):.1f} | "
                  f"{(r['state_mib'] or 0):.1f} | {sn} | {cmp_s} |")
        csv.append(f"{v},{r['n']},{r['acc']:.6f},{r['ci']:.6f},{r['tok']:.2f},"
                   f"{r['sec']:.4f},{r['cut']},{r['kv_mib']},{r['state_mib']},"
                   + ",".join("" if r["norms"][a] is None else f"{r['norms'][a]:.6f}"
                              for a in AGENTS)
                   + f",{fixed},{broke},{p}")

    problems = _gate(ns.mode, rows)
    print()
    if problems:
        print("!! CORRECTNESS GATE FAILED -- these arms are not what they claim:")
        for p_ in problems:
            print(f"     {p_}")
        print("   Do not report these numbers.")
    elif ns.mode == "channel":
        print("correctness gate: PASS  (state present/absent per arm, KV zeroed under")
        print("                        state_only, cache growing under both/kv_only)")

    if ns.mode == "channel":
        print("\nRead the arms as a DEPTH partition, not a symmetric split: attention is")
        print("8 of 32 layers, Gated DeltaNet is 24. `kv_only` leaves 25% of the depth")
        print("with history and is NOT equivalent to upstream's full-KV handoff on a")
        print("pure transformer. The honest contrast is kv_only vs both on this model.")
    else:
        print("\nIf accuracy peaks and then falls while state_norm flattens, that is the")
        print("recurrent state saturating: later thoughts overwrite earlier ones through")
        print("the delta rule rather than accumulating. Attention cannot show this.")

    if ns.md:
        open(ns.md, "w", encoding="utf-8").write("\n".join(md) + "\n")
        print(f"\nwrote {ns.md}")
    if ns.csv:
        open(ns.csv, "w", encoding="utf-8").write("\n".join(csv) + "\n")
        print(f"wrote {ns.csv}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
