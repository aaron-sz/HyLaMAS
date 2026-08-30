# Reproducibility record

Everything needed to re-run and re-check the reported numbers. Values marked
*(manifest)* are recorded automatically in every `results/**/manifest.json` and
should be quoted from there rather than retyped; `python envreport.py <run>`
prints the whole table for one run, and `python envreport.py --all` flags any
field that differs across runs when it should not.

## 1. Model

| | |
|---|---|
| Checkpoint | `Qwen/Qwen3.5-9B` (main table, ablations); `Qwen3.5-4B`, `Qwen3.5-2B`, `Qwen3.5-0.8B` (scale sweep) *(manifest: `model.name`)* |
| Revision | *(manifest: `model.chat.revision`)*, pin this; see §8 |
| Architecture | 32 layers, 24 Gated DeltaNet + 8 full attention (3:1), hidden 4096, vocab 248,077 *(manifest)* |
| Weights dtype | `torch.bfloat16` (`--dtype auto` resolves to bf16) *(manifest: `model.dtype`)* |
| Attention implementation | `sdpa` *(manifest)* |
| Weights on device | 17,078 MiB *(manifest: `model.weights_mib`)* |

**All four agents are the same model.** One `ModelWrapper`, one set of weights,
no adapters, no per-role fine-tuning. Agents differ only in their prompt and in
what the shared cache already contains.

### Cache dtypes

These are not all the same and a single "bf16" line would misdescribe the run:

| channel | dtype | size (9B, per sequence) |
|---|---|---|
| Attention KV | bf16 | 32 KiB / token (grows) |
| DeltaNet recurrent state | **fp32** | 48.0 MiB (fixed) |
| DeltaNet conv state | bf16 | 1.5 MiB (fixed) |

*(manifest: `channel.dtypes`)*. The recurrent state is fp32 because the gated
delta rule accumulates in fp32 (`g = -A_log.float().exp() * softplus(...)`).

## 2. Software environment

*(all manifest: `environment.*`)*

| | main runs | ablations |
|---|---|---|
| Python | 3.12.6 | 3.12.13 |
| PyTorch | 2.13.0 | 2.8.0 |
| Transformers | 5.15.1 | 5.15.1 |
| CUDA | 13.0 | 12.9 |
| GPU | 1x A100-SXM4-40GB (39.5 GiB) | 1x L40S (44.4 GiB) |

`causal-conv1d` and `flash-linear-attention` versions, plus the availability
flags Transformers itself uses to pick a kernel path, are recorded as
`environment.causal-conv1d`, `environment.flash-linear-attention`,
`environment.causal_conv1d_available` and
`environment.flash_linear_attention_available`. These decide whether the fused
or the reference Gated DeltaNet path runs, so they change results and not only
speed.

**State the hardware split explicitly.** The main table and the ablations ran on
different cards and different CUDA/PyTorch versions. Accuracy is unaffected but
the Speed column is only comparable within a table, never across them.
`collect.py` warns when a set of runs mixes GPUs or dtypes.

## 3. Agents

Order is fixed: **Planner -> Critic -> Refiner -> Judger** (`methods/__init__.py`,
`default_agents()`). The first three emit no text; only the Judger decodes.

The four role prompts are in `prompts.py`,
`build_agent_message_sequential_latent_mas()`, unchanged from upstream
LatentMAS. Reproduce them verbatim in the appendix. In short:

- **Planner** sees the question, is asked for a concise bulleted plan and told
  not to produce the final answer.
- **Critic** sees the question, is told the plan is available in latent form,
  and is asked to restate it and give feedback.
- **Refiner** sees the question, is told it has latent plan-plus-feedback and
  text-format question, and is asked for a refined plan.
- **Judger** sees the question, is told latent information is available for
  reference and may be irrelevant, and is asked to reason step by step and emit
  `\boxed{...}`. The exact wording is task-conditioned (answer letter for
  multiple choice, python block for code).

Under `--latent_visibility full` (the reported setting) each agent's prompt
stays in the shared cache and later agents can attend to it.

## 4. How a receiver's prompt enters the shared workspace

Not `generate()` on a concatenated string. Each agent's rendered prompt is
tokenised and appended onto the live cache by `ModelWrapper.append()`, which:

1. computes padding-aware absolute positions (`hybrid.chunk_positions`), because
   Transformers otherwise numbers a chunk `arange(width) + past_seen_tokens` and
   pushes a padded row's real tokens right by its pad count;
2. plans the append (`hybrid.plan_append`). With `--mamba_append auto`, a batch
   whose rows have equal length goes through in one forward; a ragged batch has
   its **padding prefix** fed one column at a time, snapshotting and restoring
   the DeltaNet state for rows still on padding, then the all-real remainder in
   a single forward.

The Judger is the exception: all but its last prompt token go through `append`,
and `generate()` receives one token so it only ever takes its native
single-token decode path.

Cost, measured: at `--generate_bs 8` the pad prefix is 37 columns
(163 append-forwards/sample); at 20 it is 85 (810/sample). Only `latent_mas`
appends over a live recurrent state, so **batch size inflates its Speed column
alone**. All reported runs use 8.

## 5. Latent steps

`--latent_steps 40`. One step, in `ModelWrapper.latent_rollout()`:

1. take the last hidden state `h` of the previous position;
2. map it into embedding space: `v = h @ M`, then rescale so `||v||` equals the
   mean row norm of the input embedding table;
3. extend the workspace mask by one position and forward `inputs_embeds=v` at
   position `base_pos + step` over the shared cache;
4. `h <- ` the new last hidden state.

No token is ever sampled or decoded. Each step appends exactly one position to
the KV of the attention layers and advances the recurrent and conv state of the
DeltaNet layers.

**Realignment** (`--latent_space_realign`): `M` is the least-squares map from
output-embedding space to input-embedding space,
`M = (W_out^T W_out + 1e-5 I)^{-1} W_out^T W_in`, built once per run in fp32.
With the flag off, `M = I` and only the norm rescaling is applied. Report which
you used; see the caveat in §9.

## 6. Decoding

`--temperature 0` takes a true greedy path: `_sampling_kwargs` returns
`{"do_sample": False}` and does **not** pass `temperature` or `top_p`, so no
warning is emitted and no sampling logic runs. State it as **`do_sample=False`**,
not as `T=0`.

| | |
|---|---|
| `max_new_tokens` | 2048 (GSM8K, ARC-E, ARC-C); 4096 (MedQA, MBPP+, HumanEval+) |
| Stop tokens | union of `tokenizer.eos_token_id` and `generation_config.eos_token_id` |
| Seed | 42 (`random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`) |
| Thinking mode | `--chat_thinking off` *(manifest: `model.chat.chat_thinking`)* |

The chat template is pinned by sha256 in every manifest along with the
generation tail it produces under each thinking setting. On Qwen3.5-9B the tail
is `'<think>\n'` by default (the model reasons) and `'<think>\n\n</think>\n\n'`
under `off`. This matters: with thinking left on, the Judger reasons until it
hits the cap and is then graded on the last `\boxed{}` it emitted, usually the
prompt's own placeholder. Measured on GSM8K: 20.5% truncated, 96.2% accuracy on
samples that finished but 76.5% overall. Traces from both sides are in
`results/diagnostics/`.

## 7. Data, grading, failure handling

**Selection.** Datasets are loaded in their native order, materialised to a list
before the clock starts, then `--sample_offset N` skips the first N and
`--max_samples n` takes the next n. No shuffling, no filtering, no
cherry-picking. `--sample_offset 100` was used for configuration selection and
`0` for everything reported; `collect.py --offset 0` excludes the tuning slice.

| task | source | split | n reported |
|---|---|---|---|
| GSM8K | `openai/gsm8k`, config `main` | test | 500 (1319 available) |
| ARC-Easy / ARC-Challenge | `allenai/ai2_arc` | test | 500 |
| MedQA | `data/medqa.json` (ships with the repo) | - | 300 (all) |
| MBPP+ | `evalplus/mbppplus` | test | 378 (all) |
| HumanEval+ | `evalplus/humanevalplus` | test | 164 (all) |

**Answer extraction** (`utils.py`), identical to upstream:

- Multiple choice and math: last `\boxed{...}`; a number inside the box if there
  is one, otherwise the box's stripped contents. If no box exists, the last
  number anywhere in the reply. Compared case-insensitively after stripping.
- Code: last ```` ```python ... ``` ```` block.

**Code evaluation.** The extracted block is concatenated with the task's test
harness and executed in a **separate process** with a **10-second timeout**
(`utils.run_with_timeout`). Timeout or exception counts as incorrect and the
traceback is stored in the row's `error_msg`. No retries, no partial credit.

**Failed runs.** A batch that raises kills the run; nothing is silently skipped.
Per-sample rows are written as they complete, so a crash or Ctrl-C loses at most
the in-flight batch and the manifest records `status` plus `processed`. A run
whose `status` is not `completed` is marked `*` by `collect.py` and must not be
reported as a full-split number. One such run is included here
(`results/main/gsm8k_text_mas_partial`, 140/1319) and is labelled partial.

**Truncation.** `generation.truncated_final` counts finals that hit
`max_new_tokens`; `diagnose.py` reports it as `CUT%` alongside a
`PLCHLD%` for replies whose box holds the prompt's placeholder. The gate used
here was CUT < 2% on full-cap rows.

## 8. Timing

- **Warm-up:** `--warmup 1`. One batch is run and discarded before the clock
  starts, then `torch.cuda.reset_peak_memory_stats()` so the memory watermark
  describes measured work only. *(manifest: `timing.warmup_batches_discarded`)*
- **Excluded from the clock:** model loading, tokenizer loading, dataset
  download and materialisation (the dataset is a list before `start_time`), and
  the realignment matrix build.
- **CUDA synchronisation:** `metrics.cuda_sync()` is called immediately before
  and immediately after every timed region, each batch in `process_batch`, and
  each of the append / latent / decode phases inside the model wrapper. Without
  it the timer measures kernel-launch enqueue time.
- **Repetitions:** each configuration is run **once**. Variance is reported
  across the batches of that run, not across repeated runs:
  `timing.steady_sec_per_sample` carries `n`, mean, median, p90, min and max
  over batches, and `batches.jsonl` has every batch individually. The first
  batch is flagged `is_warmup` and excluded from the steady figures.
- **What "Speed" means in the tables:** end-to-end wall-clock seconds for the
  whole run, one machine, no other jobs on the GPU. For the scale sweep all 42
  runs were executed sequentially on one A100 with zero overlap.

## 9. Workspace semantics: references, not copies

`Workspace` is a **handle**, not a snapshot. It holds the model's own
`DynamicCache` and a mask, and passing it from one agent to the next passes a
reference; there is exactly one cache and appending advances it in place. This
is forced by the implementation: `LinearAttentionLayer.update_*_state` writes
with `copy_` into a fixed buffer so the address stays stable for cudagraphs.

Cloning happens only where a rollback is needed:

- `hybrid.snapshot_state()` clones `(conv, recurrent)` per layer, used by the
  padded-append path and by `--latent_visibility latent_only`;
- `restore_state_rows()` writes those clones back for selected rows.

Consequence worth stating: the "handoff" is not a serialise-and-transmit step.
Its cost is zero and its content is whatever the previous agent left in the
cache. What `--state_transfer` does is *delete* one channel at the boundary, not
copy the other.

## 10. Runtime state-continuation test

Recorded per run in `model.state_continuation` *(manifest)*, measured on the
real checkpoint at startup by `hybrid.probe_state_continuation()`:

For each of the **recurrent** and **conv** channels it runs three forwards over
an 8-token sequence, one-shot `AB`; `A` then `B` over the live cache; `A` then
`B` with the state dropped at the boundary, and reports
`split`, `reset` and `ratio = split / reset`. A channel counts as continued when
`ratio < 0.1`.

The comparison is deliberately relative. The states are stored in the model
dtype, so in bf16 a correct append lands about 1% of the state's magnitude from
the one-shot reference while a genuine reset lands about 58% away. An absolute
tolerance would read the first as a failure.

`verify_hybrid.py` runs the offline version of the same checks (26 assertions on
a small model built from `Qwen3_5TextConfig`, no download or GPU required),
including exactness of padded appends against per-row unpadded runs
(1.3e-09 recurrent, 1.3e-07 conv) and that each `--state_transfer` mode carries
exactly what it claims.

## 11. Why the Transformers version is a method detail, not an environment detail

Gated DeltaNet has two kernel paths: a **recurrent** path for one token at a
time, and a **chunked** path for several tokens at once. An agent handoff always
takes the chunked path, because the next agent's prompt is many tokens.

In `transformers` 4.57, `modeling_qwen3_next.py` gates the recurrent path on
`seq_len == 1` and calls the chunked path with `initial_state=None`:

```python
if not use_precomputed_states:          # any seq_len > 1
    core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
        ..., initial_state=None, ...)   # previous state discarded
else:                                    # seq_len == 1 only
    core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
        ..., initial_state=recurrent_state, ...)
```

The same branch also rebuilds the conv window from the current chunk alone
(`conv_state = F.pad(mixed_qkv, (kernel - len, 0))`) without consulting the
cached one. So on that implementation **every multi-token append over a live
cache silently reset both DeltaNet channels**. The code would run, the cache
object would be passed along, and three quarters of the model's memory would
restart at each agent boundary, with no error and no warning. A latent
multi-agent result measured there would be measuring something else.

Qwen3.5 in `transformers` 5.x threads the state through on both paths, which is
why this port needs no kernel patching. Because the difference is silent, the
version is not merely recorded: it is **measured at run time on the actual
checkpoint** (§10) and the result is stored in every manifest. Report the
Transformers version, and report that the continuation test passed.

## 12. Command lines

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3.5-9B --device cuda \
  --task gsm8k --max_samples 500 --sample_offset 0 \
  --generate_bs 8 --max_new_tokens 2048 --temperature 0 --seed 42 \
  --chat_thinking off --latent_steps 40 --latent_space_realign \
  --latent_turn content --latent_visibility full --state_transfer both \
  --print2 --debug
```

The exact argv of every run is stored verbatim in `manifest.json` under
`command`. `scripts/reproduce_main.sh` and `scripts/reproduce_ablations.sh`
regenerate the full set.
