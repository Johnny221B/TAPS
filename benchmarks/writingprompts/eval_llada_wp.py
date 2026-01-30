# eval_diversity.py
import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

import accelerate
from transformers import AutoTokenizer, AutoModel

# from generate_diversity import generate, WRITER_INSTR_TEXT
from src.methods.adapters.llada_generate import generate, WRITER_INSTR_TEXT


# ------------------------
# Utilities
# ------------------------
def set_seed(seed: int, deterministic: bool = True):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def simple_tokenize(text: str) -> List[str]:
    text = text.strip().lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def ngrams(tokens: List[str], n: int):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: List[str], n: int) -> float:
    total = 0
    uniq = set()
    for t in texts:
        toks = simple_tokenize(t)
        ngs = ngrams(toks, n)
        total += len(ngs)
        uniq.update(ngs)
    return 0.0 if total == 0 else (len(uniq) / float(total))


def pick_split(ds_dict) -> str:
    for s in ["test", "validation", "val", "dev", "train"]:
        if s in ds_dict:
            return s
    return list(ds_dict.keys())[0]


def pick_prompt_column(colnames: List[str]) -> str:
    candidates = ["prompt", "wp_prompt", "writing_prompt",
                  "title", "question", "input", "text"]
    for c in candidates:
        if c in colnames:
            return c
    return colnames[0]


def sanitize_wp(raw_prompt: str) -> str:
    """
    Remove leading [WP] tags which are dataset markers and often get copied.
    """
    s = str(raw_prompt).strip()
    s = re.sub(r"^\[\s*wp\s*\]\s*", "", s, flags=re.IGNORECASE)
    return s


def build_prompt_text(raw_prompt: str, tokenizer: AutoTokenizer, chat_template: bool) -> str:
    raw_prompt = str(raw_prompt).strip()
    if not chat_template:
        return raw_prompt
    msg = [{"role": "user", "content": raw_prompt}]
    return tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def json_dump(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_write_line(f, obj: Any):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def is_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg) or ("cublas" in msg and "alloc" in msg)


def cuda_mem_gb(device) -> Tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return alloc, reserved, peak


# ------------------------
# Data structures
# ------------------------
@dataclass
class PerPromptMetrics:
    dataset_index: int
    prompt_id_global: int
    distinct_1: float
    distinct_2: float
    distinct_3: float
    avg_len_tokens: float
    num_success_samples: int
    num_failed_samples: int


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(
        "Story-telling diversity eval (multi-GPU via accelerate)")

    # model / data
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="euclaise/writingprompts")
    ap.add_argument("--split", type=str, default="",
                    help="default: auto-pick test/val/train")
    ap.add_argument("--prompt_column", type=str,
                    default="", help="default: auto-detect")
    ap.add_argument("--chat_template", action="store_true",
                    help="use tokenizer.apply_chat_template (NOT recommended for WritingPrompts)")
    ap.add_argument("--num_prompts", type=int, default=200)
    ap.add_argument("--start_index", type=int, default=0)

    # generation
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--remasking", type=str, default="low_confidence",
                    choices=["low_confidence", "random"])
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="same temperature for all modes")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)

    # reproducibility / stability
    ap.add_argument("--deterministic", action="store_true",
                    help="enable cudnn deterministic")
    ap.add_argument("--empty_cache_every", type=int, default=0,
                    help="if >0, call torch.cuda.empty_cache() every N prompts")

    # mode switch
    ap.add_argument("--mode", type=str, default="normal",
                    choices=["normal", "token", "embedding"])

    # condition-noise params
    ap.add_argument("--cond_noise_start", type=float, default=0.05)
    ap.add_argument("--cond_noise_until", type=float, default=0.95)
    ap.add_argument("--cond_noise_anneal", type=str,
                    default="cosine", choices=["linear", "cosine"])

    # token mode
    ap.add_argument("--cond_prompt_mask_ratio", type=float, default=0.05)

    # embedding mode
    ap.add_argument("--cond_embed_noise_std", type=float, default=0.05)
    ap.add_argument("--cond_embed_impl", type=str,
                    default="hook", choices=["hook", "inputs_embeds"])
    ap.add_argument("--cond_embed_psi", type=float, default=1.0)

    # misc
    ap.add_argument("--mask_id", type=int, default=126336)
    ap.add_argument("--out_dir", type=str, default="diversity_runs")
    ap.add_argument("--run_name", type=str, default="",
                    help="if empty, auto timestamp (synced across ranks)")

    # anti prompt leak
    ap.add_argument("--no_writer_instruction",
                    action="store_true", help="disable writer instruction")
    ap.add_argument("--no_ban_chat_tokens", action="store_true",
                    help="do not ban chat special tokens")
    ap.add_argument("--no_prompt_leak_guard", action="store_true",
                    help="disable prompt leak guard")
    ap.add_argument("--leak_signature_len", type=int, default=24)

    args = ap.parse_args()

    accelerator = accelerate.Accelerator()
    rank = accelerator.process_index
    world = accelerator.num_processes
    device = accelerator.device

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(accelerator.local_process_index)
        except Exception:
            torch.cuda.set_device(device)

    # ------------------------
    # Sync run_name across ranks
    # ------------------------
    ensure_dir(args.out_dir)
    run_name_file = os.path.join(args.out_dir, "_current_run_name.txt")

    if args.run_name.strip():
        run_name = args.run_name.strip()
        if accelerator.is_main_process:
            with open(run_name_file, "w", encoding="utf-8") as f:
                f.write(run_name)
    else:
        if accelerator.is_main_process:
            run_name = f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(run_name_file, "w", encoding="utf-8") as f:
                f.write(run_name)
        accelerator.wait_for_everyone()

        if not accelerator.is_main_process:
            for _ in range(200):
                if os.path.exists(run_name_file):
                    break
                time.sleep(0.05)
            with open(run_name_file, "r", encoding="utf-8") as f:
                run_name = f.read().strip()

    out_run_dir = os.path.join(args.out_dir, run_name)
    if accelerator.is_main_process:
        ensure_dir(out_run_dir)
    accelerator.wait_for_everyone()

    # ------------------------
    # Save config (main proc)
    # ------------------------
    config = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "split": args.split,
        "prompt_column": args.prompt_column,
        "chat_template": args.chat_template,
        "num_prompts": args.num_prompts,
        "start_index": args.start_index,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "mode": args.mode,
        "generation": {
            "steps": args.steps,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "remasking": args.remasking,
            "cfg": args.cfg,
            "temperature": args.temperature,
            "mask_id": args.mask_id,
        },
        "cond_noise": {
            "cond_noise_until": args.cond_noise_until,
            "cond_noise_anneal": args.cond_noise_anneal,
            "cond_prompt_mask_ratio": args.cond_prompt_mask_ratio,
            "cond_embed_noise_std": args.cond_embed_noise_std,
            "cond_embed_impl": args.cond_embed_impl,
            "cond_embed_psi": args.cond_embed_psi,
        },
        "anti_leak": {
            "add_writer_instruction": (not args.no_writer_instruction),
            "ban_chat_special_tokens": (not args.no_ban_chat_tokens),
            "prompt_leak_guard": (not args.no_prompt_leak_guard),
            "leak_signature_len": args.leak_signature_len,
        },
        "distributed": {"world_size": world},
    }
    if accelerator.is_main_process:
        json_dump(os.path.join(out_run_dir, "config.json"), config)
    accelerator.wait_for_everyone()

    # ------------------------
    # Load model/tokenizer per process
    # ------------------------
    set_seed(args.seed + rank, deterministic=args.deterministic)

    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    if tokenizer.pad_token_id == args.mask_id:
        raise ValueError(
            "pad_token_id equals mask_id; please change pad token or modify generate accordingly.")

    # WritingPrompts is not chat -> force disable chat_template to avoid template leakage
    if args.dataset.lower().endswith("writingprompts") or "writingprompts" in args.dataset.lower():
        if args.chat_template:
            if accelerator.is_main_process:
                print(
                    "[WARN] --chat_template is ON but WritingPrompts is not chat. Forcing chat_template=False.")
        args.chat_template = False

    # Precompute instruction length (for correct slicing)
    instr_len = int(tokenizer(WRITER_INSTR_TEXT, add_special_tokens=False,
                    return_tensors="pt")["input_ids"].shape[1])

    # ------------------------
    # Load dataset + select prompts
    # ------------------------
    ds_all = load_dataset(args.dataset)
    split = args.split.strip() or pick_split(ds_all)
    ds = ds_all[split]

    prompt_col = args.prompt_column.strip() or pick_prompt_column(ds.column_names)

    end_idx = min(len(ds), args.start_index + args.num_prompts)
    global_indices = list(range(args.start_index, end_idx))

    my_indices = global_indices[rank::world]
    ds_sub = ds.select(my_indices)

    # output per rank
    gen_path = os.path.join(out_run_dir, f"generations.rank{rank}.jsonl")
    metric_path = os.path.join(out_run_dir, f"metrics.rank{rank}.jsonl")
    err_path = os.path.join(out_run_dir, f"errors.rank{rank}.jsonl")
    done_flag = os.path.join(out_run_dir, f"rank{rank}.done")

    # ------------------------
    # Run generation
    # ------------------------
    local_sum_d1 = 0.0
    local_sum_d2 = 0.0
    local_sum_d3 = 0.0
    local_sum_len = 0.0
    local_cnt = 0

    pbar = tqdm(
        enumerate(ds_sub),
        total=len(ds_sub),
        desc=f"[rank {rank}] generating",
        disable=not accelerator.is_local_main_process
    )

    with open(gen_path, "w", encoding="utf-8") as f_gen, \
            open(metric_path, "w", encoding="utf-8") as f_met, \
            open(err_path, "w", encoding="utf-8") as f_err:

        for local_i, row in pbar:
            dataset_index = my_indices[local_i]

            raw_prompt_orig = row[prompt_col]
            raw_prompt_clean = sanitize_wp(raw_prompt_orig)

            # For WP baseline: use plain text prompt (no chat template)
            prompt_text = build_prompt_text(
                raw_prompt_clean, tokenizer, args.chat_template)

            generations: List[str] = []
            failed_samples = 0

            # Encode once per prompt
            with torch.inference_mode():
                enc = tokenizer(
                    [prompt_text],
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device, non_blocking=True)
                attention_mask = enc["attention_mask"].to(
                    device, non_blocking=True)

            for k in range(args.num_samples):
                set_seed(args.seed + dataset_index * 1000 +
                         k, deterministic=args.deterministic)

                gen_kwargs = dict(
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                )

                if args.mode == "normal":
                    gen_kwargs.update(dict(cond_noise_mode="none"))
                elif args.mode == "token":
                    gen_kwargs.update(dict(
                        cond_noise_mode="prompt_mask",
                        cond_prompt_mask_ratio=args.cond_prompt_mask_ratio,
                        cond_noise_until=args.cond_noise_until,
                        cond_noise_anneal=args.cond_noise_anneal,
                    ))
                elif args.mode == "embedding":
                    gen_kwargs.update(dict(
                        cond_noise_mode="embed_gaussian",
                        cond_embed_noise_std=args.cond_embed_noise_std,
                        cond_noise_start=args.cond_noise_start,
                        cond_noise_until=args.cond_noise_until,
                        cond_noise_anneal=args.cond_noise_anneal,
                        cond_embed_impl=args.cond_embed_impl,
                        cond_embed_psi=args.cond_embed_psi,
                    ))
                else:
                    raise ValueError(f"Unknown mode: {args.mode}")

                try:
                    with torch.inference_mode():
                        out_ids = generate(
                            model=model,
                            prompt_ids=input_ids,
                            tokenizer=tokenizer,
                            attention_mask=attention_mask,
                            add_writer_instruction=(
                                not args.no_writer_instruction),
                            ban_chat_special_tokens=(
                                not args.no_ban_chat_tokens),
                            ban_extra_strings=["[WP]", "[ WP ]", "[ WP]"],
                            prompt_leak_guard=(not args.no_prompt_leak_guard),
                            leak_signature_len=args.leak_signature_len,
                            **gen_kwargs,
                        )

                        # IMPORTANT: slice after (instruction + raw prompt)
                        prompt_total_len = int(
                            input_ids.shape[1]) + (0 if args.no_writer_instruction else instr_len)
                        gen_ids = out_ids[0][prompt_total_len:]
                        gen_text = tokenizer.decode(
                            gen_ids, skip_special_tokens=True).strip()

                    generations.append(gen_text)

                    rec = {
                        "prompt_id_global": dataset_index - args.start_index,
                        "dataset_index": dataset_index,
                        "mode": args.mode,
                        "sample_id": k,
                        "raw_prompt": raw_prompt_orig,
                        "raw_prompt_clean": raw_prompt_clean,
                        "prompt_text_used_for_tokenize": prompt_text,
                        "generation": gen_text,
                        "gen_kwargs": {
                            **gen_kwargs,
                            "add_writer_instruction": (not args.no_writer_instruction),
                            "ban_chat_special_tokens": (not args.no_ban_chat_tokens),
                            "prompt_leak_guard": (not args.no_prompt_leak_guard),
                            "leak_signature_len": args.leak_signature_len,
                        },
                    }
                    safe_write_line(f_gen, rec)

                except RuntimeError as e:
                    failed_samples += 1
                    safe_write_line(f_err, {
                        "dataset_index": dataset_index,
                        "prompt_id_global": dataset_index - args.start_index,
                        "mode": args.mode,
                        "sample_id": k,
                        "error_type": type(e).__name__,
                        "error_msg": str(e)[:2000],
                    })

                    if torch.cuda.is_available() and is_oom_error(e):
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue
                    else:
                        raise
                finally:
                    if "out_ids" in locals():
                        del out_ids
                    if "gen_ids" in locals():
                        del gen_ids

            # per-prompt metrics
            if len(generations) > 0:
                d1 = distinct_n(generations, 1)
                d2 = distinct_n(generations, 2)
                d3 = distinct_n(generations, 3)
                avg_len = float(
                    np.mean([len(simple_tokenize(g)) for g in generations]))

                pm = PerPromptMetrics(
                    dataset_index=dataset_index,
                    prompt_id_global=dataset_index - args.start_index,
                    distinct_1=d1,
                    distinct_2=d2,
                    distinct_3=d3,
                    avg_len_tokens=avg_len,
                    num_success_samples=len(generations),
                    num_failed_samples=failed_samples,
                )

                safe_write_line(f_met, {
                    "dataset_index": pm.dataset_index,
                    "prompt_id_global": pm.prompt_id_global,
                    "distinct_1": pm.distinct_1,
                    "distinct_2": pm.distinct_2,
                    "distinct_3": pm.distinct_3,
                    "avg_len_tokens": pm.avg_len_tokens,
                    "num_success_samples": pm.num_success_samples,
                    "num_failed_samples": pm.num_failed_samples,
                })

                local_sum_d1 += d1
                local_sum_d2 += d2
                local_sum_d3 += d3
                local_sum_len += avg_len
                local_cnt += 1
            else:
                safe_write_line(f_err, {
                    "dataset_index": dataset_index,
                    "prompt_id_global": dataset_index - args.start_index,
                    "mode": args.mode,
                    "error_type": "AllSamplesFailed",
                    "error_msg": f"all {args.num_samples} samples failed for this prompt",
                })

            if args.empty_cache_every > 0 and (local_i + 1) % args.empty_cache_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if torch.cuda.is_available() and accelerator.is_local_main_process:
                alloc, reserved, peak = cuda_mem_gb(device)
                pbar.set_postfix(
                    {"allocGB": f"{alloc:.2f}", "resvGB": f"{reserved:.2f}", "peakGB": f"{peak:.2f}"})

    # mark done
    with open(done_flag, "w", encoding="utf-8") as f:
        f.write("done\n")

    # ------------------------
    # Reduce metrics across ranks
    # ------------------------
    t_sum_d1 = torch.tensor(local_sum_d1, device=device, dtype=torch.float64)
    t_sum_d2 = torch.tensor(local_sum_d2, device=device, dtype=torch.float64)
    t_sum_d3 = torch.tensor(local_sum_d3, device=device, dtype=torch.float64)
    t_sum_len = torch.tensor(local_sum_len, device=device, dtype=torch.float64)
    t_cnt = torch.tensor(local_cnt, device=device, dtype=torch.int64)

    g_sum_d1 = accelerator.reduce(t_sum_d1, reduction="sum").item()
    g_sum_d2 = accelerator.reduce(t_sum_d2, reduction="sum").item()
    g_sum_d3 = accelerator.reduce(t_sum_d3, reduction="sum").item()
    g_sum_len = accelerator.reduce(t_sum_len, reduction="sum").item()
    g_cnt = accelerator.reduce(t_cnt, reduction="sum").item()

    accelerator.wait_for_everyone()

    # ------------------------
    # Merge jsonl + write summary (main proc)
    # ------------------------
    if accelerator.is_main_process:
        merged_gen_path = os.path.join(out_run_dir, "generations.all.jsonl")
        merged_met_path = os.path.join(out_run_dir, "metrics.all.jsonl")
        merged_err_path = os.path.join(out_run_dir, "errors.all.jsonl")

        with open(merged_gen_path, "w", encoding="utf-8") as fout:
            for r in range(world):
                p = os.path.join(out_run_dir, f"generations.rank{r}.jsonl")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)

        with open(merged_met_path, "w", encoding="utf-8") as fout:
            for r in range(world):
                p = os.path.join(out_run_dir, f"metrics.rank{r}.jsonl")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)

        with open(merged_err_path, "w", encoding="utf-8") as fout:
            for r in range(world):
                p = os.path.join(out_run_dir, f"errors.rank{r}.jsonl")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)

        mean_d1 = g_sum_d1 / max(g_cnt, 1)
        mean_d2 = g_sum_d2 / max(g_cnt, 1)
        mean_d3 = g_sum_d3 / max(g_cnt, 1)
        mean_len = g_sum_len / max(g_cnt, 1)

        summary = {
            "run_name": run_name,
            "model_path": args.model_path,
            "dataset": args.dataset,
            "split": split,
            "prompt_column": prompt_col,
            "mode": args.mode,
            "num_prompts_success": int(g_cnt),
            "num_prompts_requested": int(len(global_indices)),
            "num_samples_per_prompt": args.num_samples,
            "distributed_world_size": world,
            "metrics": {
                "distinct_1_mean": float(mean_d1),
                "distinct_2_mean": float(mean_d2),
                "distinct_3_mean": float(mean_d3),
                "avg_len_tokens_mean": float(mean_len),
            },
            "artifacts": {
                "generations_all": merged_gen_path,
                "metrics_all": merged_met_path,
                "errors_all": merged_err_path,
                "config": os.path.join(out_run_dir, "config.json"),
            }
        }

        json_dump(os.path.join(out_run_dir, "summary.json"), summary)

        print("\n=== Summary ===")
        print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
        print(f"\nSaved to: {out_run_dir}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
