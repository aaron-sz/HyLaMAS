"""LatentMAS on Qwen3.5, entry point.
LatentMAS on Qwen3.5, entry point.

Derived from Gen-Verse/LatentMAS `run.py` (Apache-2.0 Sec. 4(b)). Same CLI,
same datasets, same result JSON. Differences: the model list is Qwen3.5, there
is no vLLM backend (see README.md, vLLM cannot hand over a recurrent state),
and two flags exist for the hybrid cache: --state_transfer and --mamba_append.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

from tqdm import tqdm

import debug
from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa,
)
import metrics
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from output import RunWriter
from utils import auto_device, set_seed

# The Qwen3.5 family as published. Every one of these is hybrid
# attention + Gated DeltaNet; the MoE entries (A3B/A10B/A17B) differ only in the
# MLP, so the cache machinery here is identical for them.
KNOWN_MODELS = (
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
)


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
    writer: RunWriter = None,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    debug.dbg(f"batch: {len(current_batch)} items (done {processed}/{max_samples})")
    # Close the timer only once the GPU has actually finished. CUDA launches
    # are asynchronous, so without the sync this measures how long it took to
    # enqueue the batch and the real cost leaks into whichever later line
    # happens to read a tensor value.
    metrics.cuda_sync()
    t0 = time.perf_counter()
    results = method.run_batch(current_batch)
    metrics.cuda_sync()
    batch_wall = time.perf_counter() - t0
    if len(results) > remaining:
        results = results[:remaining]
    batch_latency = batch_wall / max(1, len(results))
    if writer is not None:
        writer.begin_batch(len(results))
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        if args.print2 and writer is not None:
            ok = writer.record(problem_idx, res, batch_latency)
            writer.print_sample(problem_idx, max_samples, res, batch_latency, ok)
            continue
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        for a in res.get("agents", []):
            name = a.get("name", "Agent")
            role = a.get("role", "")
            print(f"----- Agent: {name} ({role}) -----")
            print("[To Tokenize]")
            print(a.get("input", "").rstrip())
            latent_steps = a.get("latent_steps", None)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            if a.get("state_norm") is not None:
                print(f"[Mamba state RMS] {a['state_norm']}  [cache_len] {a.get('cache_len')}")
            print("[Output]")
            print(a.get("output", "").rstrip())
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if writer is not None:
        writer.end_batch(batch_wall)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser(
        description="LatentMAS on Qwen3.5 (hybrid attention + Gated DeltaNet).",
    )

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"],
                        required=True,
                        help="Which method to run: 'baseline' (one agent, plain "
                             "CoT), 'text_mas' (the same 4 agents relaying text), "
                             "or 'latent_mas' (the 4 agents relaying latent "
                             "thoughts through the shared hybrid cache).")
    parser.add_argument("--model_name", type=str, required=True,
                        help="A Qwen3.5 checkpoint: " + ", ".join(KNOWN_MODELS) +
                             ". A local directory is also accepted, which is how "
                             "verify_hybrid.py smoke-tests the pipeline without a "
                             "download. Qwen3.5 needs transformers >= 5.2.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Number of questions to evaluate; -1 uses all samples.")
    parser.add_argument("--sample_offset", type=int, default=0,
                        help="Skip this many questions before evaluating. "
                             "Exists so a configuration can be chosen on one "
                             "slice and reported on another: picking "
                             "--latent_turn / --latent_visibility / "
                             "--latent_steps on the same questions you report "
                             "is selection on the test set. Convention: tune "
                             "with --sample_offset 100, report with 0.")
    parser.add_argument("--task",
                        choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy",
                                 "arc_challenge", "mbppplus", "humanevalplus", "medqa"],
                        default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"],
                        default="sequential",
                        help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0,
                        help="Latent thoughts each non-judger agent takes. NOTE: "
                             "the default 0 means no latent thinking happens at "
                             "all, a run still reports accuracy but is not "
                             "measuring LatentMAS. Always pass this.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20,
                        help="Batch size for generation. Unlike a pure "
                             "transformer, batching is not free here: ragged "
                             "batches are padded, and padding has to be kept out "
                             "of the recurrent state (see --mamba_append). "
                             "--generate_bs 1 is padding-free.")
    parser.add_argument("--text_mas_context_length", type=int, default=-1,
                        help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true",
                        help="Manually add the think token to each prompt.")
    parser.add_argument("--latent_space_realign", action="store_true",
                        help="Map each latent thought through a least-squares "
                             "output->input embedding realignment before feeding "
                             "it back. Off by default, as upstream.")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Weight dtype. 'auto' uses bfloat16 only where the "
                             "GPU actually supports it and float16 otherwise, "
                             "pre-Ampere cards (e.g. T4) have no native bf16.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print2", action="store_true",
                        help="Enhanced output: compact per-sample console + saved "
                             "runs/<id>/trace.jsonl, batches.jsonl and manifest.json.")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose lifecycle tracing to stderr: model load, "
                             "per-agent activity, appends, latent rollout, tok/s.")

    meas = parser.add_argument_group(
        "measurement",
        "The paper reports Acc. / Token / Speed per cell. These control how "
        "defensible the Speed column is.")
    meas.add_argument("--warmup", type=int, default=1,
                      help="Batches to run and discard before the clock starts. "
                           "The first batch on a cold CUDA context pays module "
                           "load, kernel autotuning and allocator growth, which "
                           "on a short run can dominate the mean. Warmup batches "
                           "are still written to batches.jsonl, flagged, and "
                           "excluded from the steady-state timing summary. "
                           "0 disables.")
    meas.add_argument("--gpu_telemetry", action="store_true",
                      help="Sample GPU power/temperature/clocks to "
                           "runs/<id>/telemetry.jsonl on a background thread. "
                           "Worth it on a power-capped card (an L4 will clock "
                           "down on a sustained run, which shows up as Speed "
                           "drifting across the run rather than being wrong).")
    meas.add_argument("--telemetry_interval", type=float, default=5.0,
                      help="Seconds between --gpu_telemetry samples.")

    # Qwen3.5 hybrid cache
    hyb = parser.add_argument_group(
        "hybrid cache",
        "Qwen3.5 keeps two kinds of memory: per-token KV in the full-attention "
        "layers, and a fixed-size conv + recurrent state in the Gated DeltaNet "
        "layers (3 of every 4). See hybrid.py and README.md.")
    hyb.add_argument("--state_transfer", type=str, default="both",
                     choices=["both", "kv_only", "state_only", "none"],
                     help="Which memory channel crosses each agent boundary: "
                          "'both' (the full hybrid handoff), 'kv_only' (drop the "
                          "DeltaNet state, what upstream LatentMAS measures on a "
                          "pure transformer), 'state_only' (drop the KV, "
                          "collaboration through the fixed-size recurrent state "
                          "alone), 'none' (no-collaboration control).")
    hyb.add_argument("--mamba_append", type=str, default="auto",
                     choices=["auto", "chunk", "split", "stepwise"],
                     help="How a multi-token prompt is appended onto a live "
                          "cache. 'chunk' is one forward and is exact only when "
                          "the batch is unpadded; 'split' feeds the padding "
                          "prefix one token at a time (restoring each padded "
                          "row's state) and the rest in one forward; 'stepwise' "
                          "does every column that way. 'auto' picks chunk when "
                          "nothing needs protecting and split otherwise. Only "
                          "'chunk' can be inexact, it is there to measure what "
                          "the correctness costs.")

    # Workspace hygiene. Concatenating four full renders into one cache makes a
    # conversation the model has never seen: the system block repeats once per
    # agent, no assistant turn is ever closed with <|im_end|>, and on Qwen3.5
    ws_grp = parser.add_argument_group(
        "workspace framing",
        "How the shared cache is structured as a conversation. Defaults are "
        "upstream's behavior; the other settings are the ablation ladder.")
    ws_grp.add_argument("--latent_turn", type=str, default="raw",
                        choices=["raw", "content", "think"],
                        help="How each agent's latent block is framed. 'raw' = "
                             "upstream (system block per agent, turns never "
                             "closed, template's default reasoning tail). "
                             "'content' = one system block, later agents are "
                             "user turns, latents become the assistant turn's "
                             "content and the turn is closed with <|im_end|>. "
                             "'think' = same, but the agent's prompt ends on an "
                             "OPEN <think> and the latents are closed with "
                             "</think>, so they sit where the model puts "
                             "reasoning.")
    ws_grp.add_argument("--latent_visibility", type=str, default="full",
                        choices=["full", "latent_only"],
                        help="What a later agent sees of an earlier one. 'full' "
                             "= upstream: the whole prompt stays in the cache. "
                             "'latent_only' = an agent reads its own prompt, "
                             "then the workspace is rewound so only its latent "
                             "thoughts (plus a one-line role label) cross the "
                             "boundary. Correct on BOTH channels: the KV is "
                             "cropped and the DeltaNet state is restored from a "
                             "snapshot, then the thoughts are replayed.")

    ws_grp.add_argument("--chat_thinking", type=str, default="auto",
                        choices=["auto", "on", "off"],
                        help="Whether rendered prompts open a reasoning block, for "
                             "EVERY method (baseline, text_mas, latent_mas). 'auto' "
                             "= whatever the chat template does, which is upstream "
                             "behavior; on Qwen3.5-9B that is ON, and unbounded "
                             "reasoning is the usual cause of a high "
                             "truncated_final_rate. Set this the same across every "
                             "arm you intend to compare -- an Acc/Token table whose "
                             "rows decode differently is not a comparison.")
    ws_grp.add_argument("--judger_thinking", type=str, default="inherit",
                        choices=["inherit", "auto", "on", "off"],
                        help="Whether the judger decodes inside an open <think> "
                             "block. 'auto' = whatever the chat template does "
                             "(upstream behavior; on Qwen3.5-9B that is ON, and "
                             "unbounded reasoning there is the usual cause of a "
                             "high truncated_final_rate). 'off' makes the judger "
                             "answer directly, which typically cuts output tokens "
                             "several-fold and stops it running past "
                             "--max_new_tokens before it emits \\boxed{}. Works "
                             "in every --latent_turn mode. Default 'inherit' takes "
                             "--chat_thinking, so the judger and the controls "
                             "decode alike unless you deliberately split them.")

    # Remote generation for the text-only controls. latent_mas is rejected:
    # the method is the shared hybrid cache, and no serving API can hand a
    # Gated DeltaNet recurrent state from one request to the next.
    api = parser.add_argument_group(
        "remote backend (baseline / text_mas only)",
        "Run the controls against a served model instead of the local GPU. "
        "Accuracy and Token stay comparable to a local row ONLY if the served "
        "checkpoint is the same one; Speed is never comparable across backends.")
    api.add_argument("--backend", type=str, default="hf",
                     choices=["hf", "openai", "bedrock"],
                     help="'hf' = local weights (default). 'openai' = any "
                          "OpenAI-compatible /v1/completions server (vLLM, "
                          "SGLang, TGI, LiteLLM, OpenAI); the rendered prompt "
                          "is sent verbatim so --chat_thinking still applies. "
                          "'bedrock' = Amazon Bedrock Converse; it applies the "
                          "served model's own chat template, so --chat_thinking "
                          "does NOT reach it.")
    api.add_argument("--api_model", type=str, default=None,
                     help="Model id at the remote end. For Bedrock this is a "
                          "model id or inference-profile ARN.")
    api.add_argument("--api_base", type=str, default=None,
                     help="Base URL for --backend openai, e.g. http://localhost:8000/v1")
    api.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY",
                     help="Env var holding the bearer token for --backend openai.")
    api.add_argument("--aws_region", type=str, default=None,
                     help="Bedrock region (default: $AWS_REGION, else us-east-1).")
    api.add_argument("--aws_profile", type=str, default=None,
                     help="Named AWS profile for Bedrock credentials.")
    api.add_argument("--api_concurrency", type=int, default=8,
                     help="In-flight requests per batch. Bedrock throttles per "
                          "account; 4 or less is safer there.")
    api.add_argument("--api_timeout", type=float, default=600.0,
                     help="Per-request timeout in seconds.")

    parser.add_argument("--use_vllm", action="store_true",
                        help=argparse.SUPPRESS)  # kept only to fail with a reason

    args = parser.parse_args()

    # Model output is arbitrary Unicode, and the default console path prints it
    # verbatim. On a Windows console (cp1252) that raises UnicodeEncodeError
    # mid-run and loses the results. --print2 and --debug each reconfigure their
    # own streams; do it unconditionally so the no-flag path is safe too.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.use_vllm:
        parser.error(
            "there is no vLLM backend in this fork. The method's whole mechanism "
            "is the local hybrid cache, and vLLM has no way to hand a DeltaNet "
            "recurrent state from one request to the next. Drop --use_vllm.")

    if args.backend != "hf":
        if args.method == "latent_mas":
            parser.error(
                "--backend cannot run latent_mas. The method IS the shared hybrid "
                "cache: agent n's Gated DeltaNet recurrent state must reach agent "
                "n+1, and no serving API exposes it. Use --backend only for "
                "--method baseline / text_mas; run latent_mas locally.")
        print(f"[warn] --backend {args.backend}: Speed from this run is NOT "
              f"comparable to a local run, and Acc/Token are comparable only if "
              f"{args.api_model!r} is the same checkpoint as your local rows.",
              file=sys.stderr)
        if args.backend == "bedrock":
            print("[warn] Bedrock Converse applies the served model's own chat "
                  "template, so --chat_thinking does not reach this run.",
                  file=sys.stderr)

    if args.backend == "hf" and args.model_name not in KNOWN_MODELS and not os.path.isdir(args.model_name):
        parser.error(
            f"--model_name {args.model_name!r} is not a known Qwen3.5 checkpoint "
            f"and is not a local directory. Known: {', '.join(KNOWN_MODELS)}")

    if args.debug:
        debug.enable()
    set_seed(args.seed)
    writer = RunWriter(args) if args.print2 else None
    device = auto_device(args.device)
    debug.dbg(f"init: method={args.method} model={args.model_name} task={args.task} "
              f"device={args.device} state_transfer={args.state_transfer}")
    if args.method == "latent_mas" and args.latent_steps == 0:
        print("[warn] --latent_steps is 0: no latent thinking will happen, so this "
              "run does not measure LatentMAS.", file=sys.stderr)

    model = ModelWrapper(args.model_name, device, args=args)
    if writer is not None:
        # Records what actually loaded, resolved dtype, attention backend,
        # weight footprint, none of which `args` can say, because --dtype
        # auto and the attention kernel are decided by the machine.
        writer.attach_model(model)

    common_kwargs = dict(temperature=args.temperature, top_p=args.top_p)

    if args.method == "baseline":
        method = BaselineMethod(
            model, max_new_tokens=args.max_new_tokens, **common_kwargs,
            generate_bs=args.generate_bs, use_vllm=False, args=args,
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model, max_new_tokens_each=args.max_new_tokens, **common_kwargs,
            generate_bs=args.generate_bs, args=args,
        )
    else:
        method = LatentMASMethod(
            model, latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens, **common_kwargs,
            generate_bs=args.generate_bs, args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []

    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split="train")
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split="test")
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split="test")
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split="test")
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split="test")
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split="test")
    elif args.task == "medqa":
        dataset_iter = load_medqa(split="test")
    else:
        raise ValueError(f"no {args.task} support")

    # Materialised before the clock starts, always. `load_dataset` downloads
    # and builds the arrow cache at call time, and on a fresh pod that is
    # minutes, minutes that used to land inside total_time_sec and therefore
    # inside the Speed column. Every task here fits comfortably in memory
    # (1319 rows at the largest), and having a list also makes --warmup able
    # to reuse the head of the set.
    with debug.phase(f"load dataset {args.task}"):
        dataset_iter = list(dataset_iter)
    if args.sample_offset:
        if args.sample_offset >= len(dataset_iter):
            raise ValueError(
                f"--sample_offset {args.sample_offset} is past the end of "
                f"{args.task} ({len(dataset_iter)} rows)")
        dataset_iter = dataset_iter[args.sample_offset:]
        debug.dbg(f"dataset {args.task}: skipped {args.sample_offset}, "
                  f"{len(dataset_iter)} remain")
    if args.max_samples == -1:
        args.max_samples = len(dataset_iter)
    args.max_samples = min(args.max_samples, len(dataset_iter))

    debug.dbg(f"dataset {args.task}: {args.max_samples} samples to run")

    # Warm the kernels before measuring. The first batch on a cold CUDA
    # context pays module load, kernel autotuning and allocator growth; on a
    # 20-sample debug run that single batch can be most of the mean. Results
    # are discarded, and the same samples are then run properly.
    warmed = 0
    if args.warmup > 0 and dataset_iter:
        warm_items = dataset_iter[: max(1, args.generate_bs)]
        for i in range(args.warmup):
            debug.dbg(f"warmup batch {i + 1}/{args.warmup} ({len(warm_items)} items, discarded)")
            try:
                method.run_batch(warm_items)
                warmed += 1
            except Exception as exc:  # a warmup must never be why a run dies
                print(f"[warn] warmup batch failed ({type(exc).__name__}: {exc}); "
                      f"continuing, the first measured batch will be cold.",
                      file=sys.stderr)
                break
        metrics.cuda_sync()
        # The watermark from here on describes measured work only.
        metrics.reset_peak_memory()
    if writer is not None:
        writer.note_warmup(warmed)

    progress = tqdm(total=args.max_samples, disable=args.debug or args.print2)
    interrupted = False
    start_time = time.time()

    def _refresh_manifest(status):
        # Rewrite manifest.json to reflect samples completed so far. No-op
        # without --print2. Called each batch so a crash/Ctrl-C loses at most
        # the in-flight batch (per-sample traces are already on disk).
        if writer is None:
            return
        acc_i, correct_i = evaluate(preds)
        writer.write_manifest(
            accuracy=acc_i, correct=correct_i, processed=processed,
            total=args.max_samples, elapsed=time.time() - start_time, status=status,
        )

    try:
        for item in dataset_iter:
            if processed >= args.max_samples:
                break
            batch.append(item)
            if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
                processed, preds = process_batch(
                    method, batch, processed, preds, progress,
                    args.max_samples, args, writer,
                )
                batch = []
                _refresh_manifest("running")
                if processed >= args.max_samples:
                    break

        if batch and processed < args.max_samples:
            processed, preds = process_batch(
                method, batch, processed, preds, progress,
                max_samples=args.max_samples, args=args, writer=writer,
            )
            _refresh_manifest("running")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[interrupted] Ctrl-C, finalizing results for {processed} samples so far...",
              file=sys.stderr, flush=True)

    progress.close()
    metrics.cuda_sync()
    total_time = time.time() - start_time
    acc, correct = evaluate(preds)
    status = "interrupted" if interrupted else "completed"

    # The paper's three columns are Acc. / Token / Speed, so the one line this
    # prints without --print2 now carries all three rather than only the
    # first and third. Aggregated from `preds` so it does not depend on the
    # writer existing.
    generated_tokens = latent_steps_total = prompt_tokens = 0
    truncated_final = 0
    for p in preds:
        agents = p.get("agents", [])
        for a in agents:
            generated_tokens += int(a.get("output_token_count") or 0)
            prompt_tokens += int(a.get("input_token_count") or 0)
            latent_steps_total += int(a.get("latent_steps") or 0)
        if agents and agents[-1].get("truncated"):
            truncated_final += 1

    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "task": args.task,
                "sample_offset": args.sample_offset,
                "latent_steps": args.latent_steps,
                "latent_turn": args.latent_turn,
                "latent_visibility": args.latent_visibility,
                "chat_thinking": args.chat_thinking,
                "judger_thinking": args.judger_thinking,
                "latent_space_realign": args.latent_space_realign,
                "max_new_tokens": args.max_new_tokens,
                "generate_bs": args.generate_bs,
                "state_transfer": args.state_transfer,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "processed": processed,
                "status": status,
                "accuracy": acc,
                "accuracy_ci95": metrics.wilson_interval(correct, processed),
                "correct": correct,
                "generated_tokens": generated_tokens,
                "generated_tokens_per_sample": round(generated_tokens / processed, 2) if processed else 0.0,
                "prompt_tokens": prompt_tokens,
                "latent_steps_total": latent_steps_total,
                "truncated_final": truncated_final,
                "peak_vram_mib": metrics.peak_reserved_mib(),
                "total_time_sec": round(total_time, 4),
                "time_per_sample_sec": round(total_time / processed, 4) if processed else 0.0,
            },
            ensure_ascii=False,
        )
    )

    if writer is not None:
        writer.write_manifest(
            accuracy=acc, correct=correct, processed=processed,
            total=args.max_samples, elapsed=total_time, status=status,
        )
        writer.close()


if __name__ == "__main__":
    main()
