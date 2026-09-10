# HyLaMAS ablation tables

Regenerated from `results/` on the runs themselves. Three ablation families:
**channel** (`--state_transfer`), **latent steps** (`m`), and **workspace
framing** (`--latent_turn`), plus a **realignment control**.

---

## Shared setup: channel and latent-step ablations

| | |
|---|---|
| Model | Qwen3.5-9B (32 layers: 24 Gated DeltaNet + 8 full attention) |
| Task | MedQA, n = 100, `--sample_offset 0` (file has 300 rows) |
| Decoding | `do_sample=False` (greedy), `max_new_tokens` 4096, seed 42 |
| Batch | `--generate_bs 8`, left-padded |
| Framing | `latent_turn=content`, `latent_visibility=full`, `chat_thinking=off` |
| Realignment | **OFF** (`latent_space_realign=False` in all 10 runs) |
| Hardware | 1x L40S, torch 2.8.0, CUDA 12.9, transformers 5.15.1 |
| Prompt tokens | 1388 per sample, identical across arms by construction |

`tok` = generated tokens/sample. `tok|ok` = generated tokens among samples that
produced a final answer within the budget. `CUT` = hit `max_new_tokens`.

---

## Table 1. Channel ablation (m = 40)

| Transfer | attn 8/32 | SSM 24/32 | Acc. (95% CI) | Tokens | tok\|completed | CUT | KV MiB | State MiB | s/sample | Δacc | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `both` | yes | yes | 88.0 ± 6.4 | 923.3 | 825.1 | 3 | 406 | 396 | 21.89 | ref | ref |
| `kv_only` | yes | no | 87.0 ± 6.6 | 1169.6 | 982.8 | 6 | 406 | 0 | 20.51 | -1.0 | 1.000 |
| `state_only` | no | yes | 84.0 ± 7.2 | 791.1 | 723.7 | 2 | 0 | 396 | 15.53 | -4.0 | 0.344 |
| `none` | no | no | **90.0 ± 6.0** | 1041.0 | 978.6 | 2 | 0 | 0 | 14.66 | +2.0 | 0.754 |

**Channel bytes are derived, not read from `channel.transferred_*_mib_max`.**
The recorded field has two problems on these runs. It is a max over samples
while every other column is a mean, and it predates the `carries_state` gate in
`hybrid.state_bytes`, so an arm that stripped the DeltaNet state still reports a
full-size state crossing: `kv_only` records 388 MiB alongside
`state_norm: null`. The null is correct.

The derivation, reproduced by `ablation_report.py --mode channel`:

    KV MiB    = cache_len(Refiner) x 32.0 KiB x batch,  if the arm carries KV
              = 1624 x 32.0 KiB x 8 / 1024 = 406.0
    State MiB = 49.5 MiB x batch,               if the arm carries the GDN state
              = 49.5 x 8 = 396.0

Both constants are measured rather than assumed: across 1824 agent boundaries
the attention KV is 32.000 KiB per cached position (1808/1824 exact, the rest
off by the turn-closing tokens) and the GDN state is 49.500 MiB per sequence
(1824/1824).

Per sequence rather than per batch this reads **50.75 MiB of KV against 49.5 MiB
of state** at the Refiner-to-Judger boundary, within 3% of each other, which is
what the 1,584-position crossover predicts.

## Table 2. Channel ablation, paired significance

McNemar exact, two-sided, same 100 questions.

| contrast | fixed | broke | net | p |
|---|---:|---:|---:|---:|
| `both` → `kv_only` | 5 | 6 | -1 | 1.000 |
| `both` → `state_only` | 3 | 7 | -4 | 0.344 |
| `both` → `none` | 6 | 4 | +2 | 0.754 |
| `kv_only` → `state_only` | 6 | 9 | -3 | 0.607 |
| `kv_only` → `none` | 6 | 3 | +3 | 0.508 |
| `state_only` → `none` | 8 | 2 | +6 | 0.109 |

**Null result on accuracy, and `none` scores highest.** Nothing here is
significant. At n = 100 a paired gap needs roughly 8 points for p < 0.05. Do
not describe the channel ablation as showing an accuracy benefit.

## Table 3. Token efficiency

| | tok\|ok |
|---|---:|
| With GDN state, `both` 825.1, `state_only` 723.7 | **774.4** |
| Without, `kv_only` 982.8, `none` 978.6 | **980.7** |

| contrast | change |
|---|---:|
| `both` → `kv_only` | **+19.1%** |
| `state_only` → `none` | **+35.2%** |
| Pooled reduction from carrying the state | **21.0%** |

Two independent contrasts agree. The reading is that the judger compensates for
a missing channel by reasoning longer in text.

## Table 4. Arm-signature verification

Mean over the run at each agent boundary. Every arm's recorded state matches
exactly what its name claims. Usable as a correctness gate.

| arm | state_norm P / C / R | cache_len P / C / R |
|---|---|---|
| `both` | 0.041676 / 0.043809 / 0.045990 | 531 / 1085 / 1624 |
| `kv_only` | absent | 531 / 1085 / 1624 |
| `state_only` | 0.041676 / 0.044488 / 0.047022 | 0 / 0 / 0 |
| `none` | absent | 0 / 0 / 0 |

`state_norm` is the RMS over the **recurrent** states only; the conv state is
not included in that statistic (it is included in `state_bytes`).

## Table 5. Phase timing and append cost

Seconds over the whole 100-sample run.

| arm | append | latent | decode | total | append-forwards/sample |
|---|---:|---:|---:|---:|---:|
| `both` | 500.0 | 69.2 | 1612.6 | 2188.8 | **810.8** |
| `kv_only` | 31.0 | 67.8 | 1945.4 | 2050.9 | **4.0** |
| `state_only` | 482.4 | 67.0 | 996.7 | 1552.8 | **810.8** |
| `none` | 22.6 | 67.3 | 1369.4 | 1466.1 | **3.0** |

Arms carrying a live GDN state pay ~811 append-forwards per sample; arms
without pay 3-4. That is `plan_append` taking the padded-split path over a live
recurrent state versus the single-forward "padded prefill on a fresh state"
path. **Most of the s/samp gap in Table 1 is padding-correction overhead at
bs=8, not the channel.** Report the `decode` column, or state this explicitly.

---

## Table 6. Latent-step sweep (`state_transfer=both`)

| m | Acc % | ±95% | correct | tok | tok\|ok | CUT | s/samp | latent s | cache_len | KV MiB | state_norm P | C | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.0 | 6.0 | 90 | 1041.0 | 978.6 | 2 | 17.20 | 0.0 | 1504 | 376.0 | 0.048239 | 0.049944 | 0.052203 |
| 10 | 89.0 | 6.2 | 89 | 920.0 | 855.2 | 2 | 18.59 | 16.9 | 1534 | 383.5 | 0.044877 | 0.046837 | 0.049385 |
| 20 | 89.0 | 6.2 | 89 | 926.8 | 794.8 | 4 | 19.88 | 34.1 | 1564 | 391.0 | 0.043628 | 0.045755 | 0.048164 |
| 40 | 88.0 | 6.4 | 88 | 923.3 | 825.1 | 3 | 21.80 | 68.5 | 1624 | 406.0 | 0.041676 | 0.043809 | 0.045990 |
| 80 | 86.0 | 6.8 | 86 | 916.6 | 784.1 | 4 | 20.99 | 138.0 | 1744 | 436.0 | 0.038795 | 0.041038 | 0.042841 |

KV MiB is derived as `cache_len x 32.0 KiB x 8`, the same formula as Table 1, so
the m = 40 row equals the `both` arm there (406.0). The GDN state is 396.0 MiB
at every m.

Accuracy is **monotone decreasing**, Pearson r(acc, m) = **-0.990**.

## Table 7. Latent-step sweep, paired significance

| contrast | fixed | broke | net | p |
|---|---:|---:|---:|---:|
| m=40 → m=0 | 6 | 4 | +2 | 0.754 |
| m=40 → m=10 | 3 | 2 | +1 | 1.000 |
| m=40 → m=20 | 6 | 5 | +1 | 1.000 |
| m=40 → m=80 | 4 | 6 | -2 | 0.754 |
| m=0 → m=10 | 2 | 3 | -1 | 1.000 |
| m=10 → m=20 | 3 | 3 | 0 | 1.000 |
| m=20 → m=40 | 5 | 6 | -1 | 1.000 |
| m=40 → m=80 | 4 | 6 | -2 | 0.754 |
| **m=0 → m=80** | 2 | 6 | -4 | **0.289** |

**No individual contrast is significant.** The trend is the finding, not any
pair. Report r = -0.990 with the caveat.

## Table 8. The O(1) demonstration

From the same sweep, as m goes 0 → 80:

| m | KV MiB (batch of 8) | GDN state MiB (batch of 8) |
|---:|---:|---:|
| 0 | 376.0 | **396.0** |
| 10 | 383.5 | **396.0** |
| 20 | 391.0 | **396.0** |
| 40 | 406.0 | **396.0** |
| 80 | 436.0 | **396.0** |

KV grows 60.0 MiB; the GDN state does not move by one byte. The arithmetic
checks exactly: 80 steps x 3 agents = 240 extra cached positions x 32 KiB x 8
rows = 60.0 MiB.

The sweep also brackets the crossover. At m = 0 the KV channel is **smaller**
than the state (376.0 vs 396.0); by m = 40 it is larger (406.0 vs 396.0). The
two are equal at 1,584 cached positions, and the Refiner boundary passes
through that point between m = 20 (1564) and m = 40 (1624).

Two structural facts from the `state_norm` columns of Table 6:

- **`state_norm` falls as m rises** (Refiner 0.052203 → 0.042841). The gated
  decay dominates accumulation, the state shrinks rather than saturating.
- **`state_norm` rises across agents within a run** (P → C → R at every m), so
  each successive agent does add to it.

## Table 9. Identity check, m = 0 versus `none`

| | |
|---|---|
| Agreement | **100/100 questions** |
| Accuracy | 90.0 vs 90.0 |
| Discordant pairs | 0 / 0 |

With zero latent steps the judger decodes cold and the workspace provably does
nothing. A clean validation that the shared cache is the only path between
agents.

---

## Table 10. Workspace-framing ablation

Qwen3.5-9B, n = 200, bs = 8, `max_new_tokens` 2048, m = 40,
`state_transfer=both`, 1x L40S.

| task | `latent_turn` | realign | n | Acc % | ±95% | tok | tok\|ok | CUT | sec |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | raw | False | 200 | 76.5 | 5.8 | 1232.8 | 1022.6 | 41/200 | 2689 |
| GSM8K | raw | True | 200 | 67.5 | 6.4 | 1530.8 | 1239.9 | 72/200 | 2736 |
| GSM8K | **content** | True | 200 | **95.0** | 3.1 | 366.2 | 340.6 | 3/200 | 1351 |
| ARC-E | content | True | 200 | 99.5 | 1.3 | 335.1 | 335.1 | 0/200 | 1003 |
| MedQA | content | True | 104* | 88.5 | 6.2 | 966.5 | 740.6 | 7/104 | 2602 |

\* interrupted at 104 of 200; `max_new_tokens` 4096 for this row.

## Table 11. Framing ablation, paired significance

**Framing, realignment held ON, `raw` → `content`:**

| | |
|---|---|
| Accuracy | 67.5 → **95.0** (**+27.5 pp**) |
| Discordant | 56 fixed / 1 broke, net +55 |
| McNemar exact | **p = 8.05 x 10⁻¹⁶** (n paired = 200) |
| tok\|ok | 1240 → 341 (**-72.5%**) |
| Truncation | 72/200 → 3/200 |
| Wall clock | 2736 s → 1351 s |

**Realignment, framing held at `raw`; False → True:**

| | |
|---|---|
| Accuracy | 76.5 → 67.5 (**-9.0 pp**) |
| Discordant | 19 fixed / 37 broke, net -18 |
| McNemar exact | **p = 0.0222** |
| Truncation | 41/200 → 72/200 |

The two variables interact, present as a 2x2, not two independent knobs. The
fourth cell (`content` + realign=False, GSM8K n=200) was never run.

## Table 12. Realignment control

Resolves the config mismatch between the ablations (realign off) and the scale
sweep (realign on). MedQA, `content`/`both`/m=40, same questions.

| | n | accuracy | correct on the shared 100 |
|---|---:|---:|---:|
| realign **ON** | 104 | 88.5 | 88/100 |
| realign **OFF** | 100 | 88.0 | 88/100 |

Discordant 5 each way, **McNemar p = 1.000**. Realignment is immaterial on
MedQA, so the ablation/scale-sweep inconsistency can be stated as measured
rather than hedged.
