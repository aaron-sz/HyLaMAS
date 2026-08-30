# Implementation notes

Background for the parts of this fork that are not obvious from the code.
Numbers quoted here are reproduced by `verify_hybrid.py`.

## 1. Why the cache needed rewriting

Upstream LatentMAS passes one agent's working memory to the next by handing
over the KV cache. On a dense transformer that is all of the model's memory.

Qwen3.5 is hybrid. Its `layer_types` interleave `full_attention` with
`linear_attention` (Gated DeltaNet) on a 3:1 pattern, so three of every four
layers keep no per-token KV at all. Those layers keep two fixed-size tensors
instead:

    conv_states       [B, 2*key_dim + value_dim, linear_conv_kernel_dim]
    recurrent_states  [B, linear_num_value_heads, key_head_dim, value_head_dim]

`recurrent_states` is overwritten each step rather than appended to. Carrying
only the KV across an agent boundary would therefore carry a quarter of the
model's memory and silently restart the rest.

For Qwen3.5-9B (24 linear layers of 32) the two channels are:

| channel | dtype | bytes per sequence | grows with tokens |
|---|---|---:|---|
| attention KV | bf16 | 32 KiB / token | yes |
| DeltaNet recurrent | fp32 | 48.0 MiB | no |
| DeltaNet conv | bf16 | 1.5 MiB | no |

The two cross over at 1,584 tokens: below that the recurrent state is the
larger channel, above it the KV is.

## 2. Multi-token appends continue the recurrent state

`Qwen3_5GatedDeltaNet.forward` passes `initial_state=recurrent_state` on the
chunked path as well as the single-token path, with no `seq_len == 1` gate.
One-shot `AB` versus `A` then `B` agree to 1.6e-09, so an agent handoff needs
no kernel patching.

This was not true of the predecessor. In `transformers` 4.57,
`modeling_qwen3_next.py` gates the recurrent path on `seq_len == 1` and passes
`initial_state=None` on the chunked path, so every multi-token append reset the
state. `probe_append_carries_state` re-measures this at startup instead of
trusting the version number.

The 1.6e-09 figure is fp32. The recurrent state is stored in the model dtype,
so in bf16 it round-trips through about three decimal digits and a correct
append lands roughly 1% of the state's magnitude from the one-shot reference.
An absolute tolerance would read that as a failure. A genuine state reset is
about 58% away, so the probe compares the two cases against each other rather
than against a fixed threshold.

## 3. Left-padded batches over a live cache

Two separate problems, both of which have to be fixed.

**Positions.** `Qwen3_5TextModel` derives `position_ids` as
`arange(width) + past_seen_tokens`, ignoring padding, so a row with p pads has
every real token sitting p slots too far right. `chunk_positions` builds
padding-aware positions. That reduces the error but does not remove it.

**The pads themselves.** With the 2D mask reaching it the DeltaNet zeroes
padded hidden states, but a zeroed position still decays the recurrent state
(`g = -A * softplus(dt_bias) < 0` regardless of input), and the conv window at
the first real token is filled with pads rather than the cached tokens that
actually precede it. Measured on a 4-pad row, after the position fix:
recurrent 1.9e-04, conv 9.7e-03.

`AppendPlan` routes the padded prefix through single-token steps, snapshotting
and restoring state for rows still sitting on padding. Result against per-row
unpadded runs: 1.3e-09 recurrent, 1.3e-07 conv.

A padded prefill on a *fresh* cache is already exact (8e-10) once positions are
right, because a zero state decays to zero. Only appends over a live state need
the split treatment.

**Cost.** The split append pays one forward per column of the padding prefix
(`width - min_len`) plus one forward for the all-real remainder, not one
forward per token. Similar-length batches stay near one forward per agent, and
`--generate_bs 1` has no padding and always takes the single-forward path.

At `--generate_bs 8` the measured pad prefix is 37 columns (163 append-forwards
per sample); at 20 it is 85 (810 per sample). Since only `latent_mas` appends
over a live state, raising the batch size inflates its Speed column and nothing
else's. All reported runs use 8.

## 4. `--state_transfer`

Which channel crosses each agent boundary:

| mode | KV | DeltaNet state |
|---|---|---|
| `both` | yes | yes |
| `kv_only` | yes | no |
| `state_only` | no | yes |
| `none` | no | no |

This is a 25/75 split by depth, not a symmetric one. `kv_only` is what upstream
LatentMAS measures on a dense transformer. `state_only` restricts the
inter-agent channel to a fixed size regardless of how long the agents thought.

`strip_state` calls each layer's `reset()`, which zeroes the buffers and lowers
`has_previous_state` but keeps the allocation, since the address has to stay
stable for cudagraphs. `state_bytes` therefore gates on `carries_state` rather
than on tensor size; without that gate it would report a full recurrent state
crossing every `kv_only` and `none` boundary when nothing crosses at all.

Note that `state_norm` in the traces sums the recurrent states only. The
`state_bytes` column covers both recurrent and conv.

## 5. Workspace framing

Concatenating four full chat renders into one cache produces a conversation the
model has never seen: the system block repeats once per agent, no assistant
turn is closed with `<|im_end|>`, and on Qwen3.5 the template caps every
generation prompt with a reasoning block. `--latent_turn` and
`--latent_visibility` make that structure adjustable; the defaults reproduce
upstream byte for byte.

The assistant-turn tail differs across model generations, and it decides
whether an agent reasons at all:

    Qwen3          ''                          model chooses
    Qwen3.5-0.8B   '<think>\n\n</think>\n\n'   non-thinking
    Qwen3.5-9B     '<think>\n'                 thinking by default

Hardcoding any one of these mis-frames the others, so `_discover_tails` reads
them off the live template at startup.

This matters for accuracy, not only style. On a thinking-by-default template
the judger reasons until it runs out of budget, and a truncated judger is
graded on the last `\boxed{}` it emitted, which is usually the prompt's own
placeholder. Measured on Qwen3.5-9B / GSM8K: 20.5% of samples truncated, and
accuracy fell from 96.2% on samples that finished to 76.5% overall. Pass
`--chat_thinking off` or `--judger_thinking off` for a direct answer, and check
`generation.truncated_final_rate` afterwards.

`results/diagnostics/` holds a trace from each side of this.
