# Superseded runs

Kept for the record; not used in any reported table.

- `n100_bs4_cap2048/`, an earlier attempt at the 9B main table, run at
  `--generate_bs 4 --max_new_tokens 2048`. Superseded by `results/main/n100/`,
  which uses bs=8 and the per-task caps (4096 for MedQA/MBPP+/HumanEval+) that
  match the ablations. Three of its nine runs were killed before finishing.
  MedQA latent_mas scores 81.0 here against 88.0 in the reported run; the
  difference is truncation, 16/100 against 7/100 at the larger cap.

- `gsm8k_text_mas_partial/`, GSM8K text_mas at `--max_samples 1319`,
  terminated at 140 samples (bs=20). The only run in the corpus at bs=20.
