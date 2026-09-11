# HyLaMAS: Hybrid-State LatentMAS

An implementation of LatentMAS ([Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS),
Apache-2.0) for hybrid attention and Gated DeltaNet models, together with the
instrumentation required to measure the contribution of each memory channel.

LatentMAS executes four agents (Planner, Critic, Refiner, Judger) over a single
shared cache. The first three emit no text: they perform latent rollout steps,
and the next agent inherits the resulting state. On a dense transformer,
transferring the KV cache transfers the entire inference state of the model.
This does not hold for Qwen3.5, where three of every four layers are Gated
DeltaNet and maintain a fixed-size recurrent and convolution state rather than
per-token KV. HyLaMAS (Hybrid-State LatentMAS) transfers both channels across
every agent boundary and provides a flag to disable either one, so that their
individual contributions can be isolated.

See `DESIGN.md` for implementation notes and `REPRODUCIBILITY.md` for the
complete experimental record.

## Installation

Qwen3.5 requires `transformers >= 5.2`.

```bash
pip install -r requirements.txt
python verify_hybrid.py          # 26 checks against a locally constructed Qwen3.5
```

`verify_hybrid.py` constructs a small model from `Qwen3_5TextConfig` and
requires no download and no GPU. It should be run before any experiment, as it
verifies that appends over a live cache continue both the recurrent and the
convolution state, that padded batches produce results identical to unpadded
per-row runs, and that each `--state_transfer` mode carries exactly what it
claims.

## Usage

```bash
# single agent, four text agents, four latent agents
python run.py --method baseline   --model_name Qwen/Qwen3.5-9B --task gsm8k \
  --max_samples 500 --generate_bs 8 --max_new_tokens 2048 --temperature 0 \
  --chat_thinking off --print2

python run.py --method text_mas   ...same...

python run.py --method latent_mas ...same... \
  --latent_steps 40 --latent_space_realign \
  --latent_turn content --latent_visibility full
```

Complete tables:

```bash
bash scripts/reproduce_main.sh            # 6 benchmarks x 3 methods
bash scripts/reproduce_ablations.sh       # scale sweep, channel, steps
```

Both scripts are resumable and dispatch work across all visible GPUs. The
`--dry-run` flag prints the execution plan without running anything.

## Additional flags

| flag | description |
|---|---|
| `--state_transfer {both,kv_only,state_only,none}` | which memory channel crosses each agent boundary |
| `--mamba_append {auto,chunk,split,stepwise}` | how a padded prompt is appended over a live recurrent state |
| `--latent_turn {raw,content,think}` | how each latent block is framed as a chat turn |
| `--latent_visibility {full,latent_only}` | whether an agent's prompt remains in the shared cache |
| `--chat_thinking {auto,on,off}` | override the chat template's reasoning default |
| `--judger_thinking {inherit,auto,on,off}` | the same, applied to the Judger alone |
| `--sample_offset N` | skip N examples, allowing a configuration to be selected on one slice and reported on another |

Default values reproduce the behaviour of the upstream implementation.

## Experimental considerations

**Reasoning mode affects accuracy more than the choice of method.** The
Qwen3.5-9B chat template opens a reasoning block by default, causing the Judger
to reason until it reaches `--max_new_tokens`. A truncated Judger is graded on
the last `\boxed{}` it emitted, which is typically the answer placeholder
contained in its own prompt. Measured on GSM8K: 20.5% of examples truncated,
with accuracy of 96.2% among examples that completed against 76.5% overall.
Pass `--chat_thinking off` and inspect `generation.truncated_final_rate`.

**Batch size affects the methods asymmetrically.** Only `latent_mas` appends
over a live recurrent state, and therefore incurs `pad_prefix + 1` forward
passes per append where the other methods incur one. Increasing
`--generate_bs` inflates its reported time alone. All reported runs use a batch
size of 8.

**Wall-clock time is comparable only within a single machine.** `collect.py`
emits a warning when a set of runs mixes GPU models or dtypes.

**Significance should be assessed with paired tests.** Overlapping confidence
intervals do not establish that two arms are indistinguishable. `compare.py`
computes McNemar's exact test over the same examples.

## Results

```
results/
  main/n100/    primary table: 9B, 6 benchmarks x 3 methods, n=100
  scale/        0.8B / 2B / 4B, identical grid
  ablations/    --state_transfer arms and the latent-step sweep
  framing/      --latent_turn raw against content
  tuning/       configuration selection, performed on a disjoint slice (offset 100)
  diagnostics/  traces illustrating the Judger reasoning failure and its correction
  superseded/   earlier runs, not used in any reported table
```

`results/README.md` documents the directory layout, the shared configuration
and the two hardware groups. Each run directory contains `manifest.json`,
`trace.jsonl` and `batches.jsonl`.

```bash
python collect.py --compare baseline        # the table
python diagnose.py                          # truncation and placeholder rates
python compare.py --task gsm8k --a baseline --b latent_mas
python ablation_report.py --mode channel --task medqa
python envreport.py --all                   # cross-run configuration consistency
```

Release tooling: `anonymize.py` removes hostnames and kernel strings from
manifests, `slim_traces.py` discards redundant trace fields, and
`regrade_code.py` addresses the HumanEval+ import artifact described in
`results/README.md`.

## Attribution

Derived from Gen-Verse/LatentMAS (Apache-2.0). `run.py`, `prompts.py`,
`methods/` and the grading path follow the upstream implementation so that
results remain comparable; `hybrid.py` and the two-channel handling are new.
See `NOTICE` for the file-level record and `LICENSE` for terms.
