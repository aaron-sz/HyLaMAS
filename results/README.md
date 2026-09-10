# Results

92 runs. Every directory holds `manifest.json` (config, environment, accuracy
with a Wilson interval, token and timing breakdowns, channel bytes),
`trace.jsonl` (one row per question) and `batches.jsonl`.

Traces are slimmed for release: `solution` is dropped where it duplicates
`gold` (every task but GSM8K, where it is the reference derivation) and the
per-agent rendered prompt is dropped, since it is reconstructible from
`prompts.py`. Everything the tables read is intact. Hostnames and kernel
strings are scrubbed; GPU, driver, torch, transformers and CUDA versions are
kept.

## Layout

| directory | runs | what it is |
|---|---:|---|
| `main/n100/` | 18 | **The headline table.** Qwen3.5-9B, 6 benchmarks x 3 methods, n=100, offset 0 |
| `main/logs/` | 19 | stdout of those runs plus the sweep summary |
| `scale/{0.8B,2B,4B}/` | 42 | model-size sweep, same grid |
| `ablations/runs/` | 10 | 9B MedQA: 4 `--state_transfer` arms and a 5-point latent-step sweep |
| `ablations/{logs,tables}/` | | stdout and generated markdown/csv |
| `framing/` | 5 | 9B, n=200: `--latent_turn` raw vs content, realignment on/off |
| `tuning/` | 8 | offset 100. Configuration selection, deliberately a different slice from the reported one |
| `diagnostics/` | 2 | traces from the judger-reasoning failure and its fix |
| `superseded/` | 10 | earlier attempts, not used in any table. See its README |
| `tables/` | | generated output |

## Configuration of the reported runs

`main/n100/` and `scale/` share one configuration:

    greedy (do_sample=False), seed 42, --generate_bs 8
    --max_new_tokens 4096 (MedQA, MBPP+, HumanEval+) / 2048 (GSM8K, ARC-E, ARC-C)
    --chat_thinking off
    --latent_steps 40 --latent_space_realign
    --latent_turn content --latent_visibility full --state_transfer both

`ablations/` differs in two respects, both deliberate: it sweeps
`--state_transfer` and `--latent_steps`, and it ran with realignment off.
`framing/` sweeps `--latent_turn` at n=200 with a 2048 cap.

## Hardware

Two environments; wall-clock is comparable only within one.

| group | GPU | torch | CUDA |
|---|---|---|---|
| `scale/` | 1x A100-SXM4-40GB | 2.13.0 | 13.0 |
| `main/`, `ablations/`, `framing/`, `tuning/` | 1x L40S | 2.8.0 | 12.9 |

transformers 5.15.1 throughout.

## Reading them

    python collect.py --compare baseline          # the table
    python diagnose.py                            # truncation and placeholder rates
    python compare.py --task gsm8k --a baseline --b latent_mas
    python ablation_report.py --mode channel --task medqa
    python envreport.py results/main/n100/<run>   # full reproducibility record
    python envreport.py --all                     # cross-run consistency check

## Channel-byte figures

`ablation_report.py --mode channel` derives the KV and state bytes from the
architecture and the measured cache length, gated on `--state_transfer`, rather
than reading `channel.transferred_*_mib_max` from the manifest. That recorded
field is a max over samples, and on the ablation runs it predates the
`carries_state` gate in `hybrid.state_bytes`, so an arm that stripped the
DeltaNet state still records a full-size state crossing. The trace's
`state_norm` is null for exactly those arms. The derived figures are the ones
reported.

## HumanEval+ regrading

Manifests for the code tasks carry `regraded_accuracy` alongside `accuracy`.
HumanEval+ prompt stubs supply `from typing import List`; a model that emits
only the function body fails execution with a `NameError` that is an
extraction artifact rather than a wrong answer, and it does not affect methods
equally. `regrade_code.py` re-executes those cases with the imports the stub
would have provided. Report `regraded_accuracy` for HumanEval+. MBPP+ is
unaffected (zero cases fixed in every run).
