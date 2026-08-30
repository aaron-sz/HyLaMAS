# HyLaMAS: Hybrid-State LatentMAS

A port of LatentMAS ([Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS),
Apache-2.0) to a hybrid attention + Gated DeltaNet model, plus the machinery
needed to measure what the two memory channels each contribute.

LatentMAS runs four agents (Planner, Critic, Refiner, Judger) over one shared
cache. The first three emit no text; they think in latent space and the next
agent inherits the result. On a dense transformer, handing over the KV cache
hands over all of the model's memory. On Qwen3.5 it does not: three of every
four layers are Gated DeltaNet and keep a fixed-size recurrent and convolution
state instead of per-token KV. HyLaMAS (Hybrid-State LatentMAS) carries both
channels across every agent boundary, and adds a flag to turn each one off so
their contributions can be separated.

See `DESIGN.md` for implementation notes and `REPRODUCIBILITY.md` for the
full experimental record.

## Install

Qwen3.5 needs `transformers >= 5.2`.

```bash
pip install -r requirements.txt
python verify_hybrid.py          # 26 checks against a small local Qwen3.5
```

`verify_hybrid.py` builds a tiny model from `Qwen3_5TextConfig` and needs no
download or GPU. It must pass before any run is worth trusting: it checks that
appends over a live cache continue the recurrent and convolution state,
that padded batches
land in the same place as unpadded ones, and that each `--state_transfer` mode
carries what it claims.

## Running

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

Full tables:

```bash
bash scripts/reproduce_main.sh            # 6 benchmarks x 3 methods
bash scripts/reproduce_ablations.sh       # scale sweep, channel, steps
```

Both are resumable and dispatch over all visible GPUs. `--dry-run` prints the
plan without running anything.

## Flags beyond upstream's

| flag | what it does |
|---|---|
| `--state_transfer {both,kv_only,state_only,none}` | which memory channel crosses each agent boundary |
| `--mamba_append {auto,chunk,split,stepwise}` | how a padded prompt is appended over a live recurrent state |
| `--latent_turn {raw,content,think}` | how each latent block is framed as a chat turn |
| `--latent_visibility {full,latent_only}` | whether an agent's prompt stays in the shared cache |
| `--chat_thinking {auto,on,off}` | override the template's reasoning default |
| `--judger_thinking {inherit,auto,on,off}` | the same, for the judger alone |
| `--sample_offset N` | skip N questions, so a configuration can be chosen on one slice and reported on another |

Defaults reproduce upstream behaviour.

## Things that will bite you

**The judger's reasoning mode changes the score more than the method does.**
Qwen3.5-9B's chat template opens `<think>` by default, so the judger reasons
until it hits `--max_new_tokens`. A truncated judger is graded on the last
`\boxed{}` it emitted, which is usually the prompt's own placeholder. Measured
on GSM8K: 20.5% truncated, accuracy 96.2% on samples that finished but 76.5%
overall. Pass `--chat_thinking off` and check `generation.truncated_final_rate`.

**Batch size is not free.** Only `latent_mas` appends over a live recurrent
state, so it pays `pad_prefix + 1` forwards per append where the other methods
pay one. Raising `--generate_bs` inflates its Speed column alone. All reported
runs use 8.

**Speed is only comparable within one machine.** `collect.py` warns if a set of
runs mixes GPUs or dtypes.

**Use paired tests.** Two confidence intervals overlapping does not mean two
arms are indistinguishable. `compare.py` runs McNemar's exact test on the same
questions.

## Results

```
results/
  main/n100/    the headline table: 9B, 6 benchmarks x 3 methods, n=100
  scale/        0.8B / 2B / 4B, same grid
  ablations/    --state_transfer arms and the latent-step sweep
  framing/      --latent_turn raw vs content
  tuning/       configuration selection, on a different slice (offset 100)
  diagnostics/  traces from the judger-reasoning failure and its fix
  superseded/   earlier attempts, not used in any table
```

`results/README.md` documents the layout, the shared configuration and the two
hardware groups. Each run directory holds `manifest.json`, `trace.jsonl` and
`batches.jsonl`.

```bash
python collect.py --compare baseline        # the table
python diagnose.py                          # truncation and placeholder rates
python compare.py --task gsm8k --a baseline --b latent_mas
python ablation_report.py --mode channel --task medqa
python envreport.py --all                   # cross-run config consistency
```

Release tooling: `python anonymize.py` strips hostnames and kernel strings from
manifests, `python slim_traces.py` drops redundant trace fields, and
`python regrade_code.py` handles the HumanEval+ import artifact described in
`results/README.md`.

## Attribution

Derived from Gen-Verse/LatentMAS (Apache-2.0). `run.py`, `prompts.py`,
`methods/` and the grading path follow upstream so numbers stay comparable;
`hybrid.py` and the two-channel handling are new. See `NOTICE` for the file-level
record and `LICENSE` for terms.
