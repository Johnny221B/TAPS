# benchmarks/writingprompts/run_diversity.py
import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
import accelerate
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from src.methods.taps import generate_ids
from src.methods.adapters.llada_generate import WRITER_INSTR_TEXT as WRITER_INSTR_TEXT_LLADA
from src.methods.adapters.trado_generate import WRITER_INSTR_TEXT as WRITER_INSTR_TEXT_TRADO


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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def json_dump(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_write_line(f, obj: Any):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


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
    s = str(raw_prompt).strip()
    s = re.sub(r"^\[\s*wp\s*\]\s*", "", s, flags=re.IGNORECASE)
    return s


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(
        "WritingPrompts diversity eval (LLaDA / TraDo)")

    ap.add_argument("--backbone", type=str, required=True,
                    choices=["llada", "trado"])
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="euclaise/writingprompts")
    ap.add_argument("--split", type=str, default="")
    ap.add_argument("--prompt_column", type=str, default="")
    ap.add_argument("--num_prompts", type=int, default=200)
    ap.add_argument("--start_index", type=int, default=0)

    # generation shared
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--remasking", type=str, default="low_confidence",
                    choices=["low_confidence", "random"])
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--empty_cache_every", type=int, default=0)

    # mode
    ap.add_argument("--mode", type=str, default="normal",
                    choices=["normal", "token", "embedding"])

    # noise params
    ap.add_argument("--cond_noise_start", type=float, default=0.05)
    ap.add_argument("--cond_noise_until", type=float, default=0.95)
    ap.add_argument("--cond_noise_anneal", type=str,
                    default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--cond_prompt_mask_ratio", type=float, default=0.05)
    ap.add_argument("--cond_embed_noise_std", type=float, default=0.05)
    ap.add_argument("--cond_embed_impl", type=str, default="hook")
    ap.add_argument("--cond_embed_psi", type=float, default=1.0)

    # sampling filters (TraDo uses these; LLaDA will just ignore safely)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--min_p", type=float, default=0.0)

    # misc
    ap.add_argument("--mask_id", type=int, default=126336)
    ap.add_argument("--out_dir", type=str, default="diversity_runs")
    ap.add_argument("--run_name", type=str, default="")

    # WP anti-leak
    ap.add_argument("--no_writer_instruction", action="store_true")
    ap.add_argument("--no_ban_chat_tokens", action="store_true")
    ap.add_argument("--no_prompt_leak_guard", action="store_true")
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
            pass

    # ------------------------
    # Sync run_name
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
            run_name = f"wp_{args.backbone}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    # save config (main proc)
    if accelerator.is_main_process:
        json_dump(os.path.join(out_run_dir, "config.json"), vars(args))

    # ------------------------
    # Load model/tokenizer per backbone
    # ------------------------
    set_seed(args.seed + rank, deterministic=args.deterministic)

    if args.backbone == "llada":
        model = AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True)
        writer_instr = WRITER_INSTR_TEXT_LLADA
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True)
        writer_instr = WRITER_INSTR_TEXT_TRADO

    # padding side for WP
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    # instruction length for slicing
    instr_len = int(tokenizer(writer_instr, add_special_tokens=False,
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

    # outputs per rank
    gen_path = os.path.join(out_run_dir, f"generations.rank{rank}.jsonl")
    metric_path = os.path.join(out_run_dir, f"metrics.rank{rank}.jsonl")
    err_path = os.path.join(out_run_dir, f"errors.rank{rank}.jsonl")

    # ------------------------
    # Run generation
    # ------------------------
    pbar = tqdm(enumerate(ds_sub), total=len(ds_sub),
                disable=not accelerator.is_local_main_process)
    with open(gen_path, "w", encoding="utf-8") as f_gen, \
            open(metric_path, "w", encoding="utf-8") as f_met, \
            open(err_path, "w", encoding="utf-8") as f_err:

        for local_i, row in pbar:
            dataset_index = my_indices[local_i]
            raw_prompt = row[prompt_col]
            prompt_text = sanitize_wp(raw_prompt)

            enc = tokenizer([prompt_text], add_special_tokens=False,
                            padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            generations: List[str] = []
            failed = 0

            for k in range(args.num_samples):
                set_seed(args.seed + dataset_index * 1000 +
                         k, deterministic=args.deterministic)

                # build kwargs
                kw = dict(
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                    add_writer_instruction=(not args.no_writer_instruction),
                    ban_chat_special_tokens=(not args.no_ban_chat_tokens),
                    prompt_leak_guard=(not args.no_prompt_leak_guard),
                    leak_signature_len=args.leak_signature_len,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    attention_mask=attention_mask,
                )

                # mode switch
                if args.mode == "normal":
                    kw.update(dict(cond_noise_mode="none"))
                elif args.mode == "token":
                    kw.update(dict(
                        cond_noise_mode="prompt_mask",
                        cond_prompt_mask_ratio=args.cond_prompt_mask_ratio,
                        cond_noise_start=args.cond_noise_start,
                        cond_noise_until=args.cond_noise_until,
                        cond_noise_anneal=args.cond_noise_anneal,
                    ))
                else:
                    kw.update(dict(
                        cond_noise_mode="embed_gaussian",
                        cond_embed_noise_std=args.cond_embed_noise_std,
                        cond_noise_start=args.cond_noise_start,
                        cond_noise_until=args.cond_noise_until,
                        cond_noise_anneal=args.cond_noise_anneal,
                        cond_embed_impl=args.cond_embed_impl,
                        cond_embed_psi=args.cond_embed_psi,
                    ))

                try:
                    out_ids = generate_ids(
                        backbone=args.backbone,
                        model=model,
                        prompt_ids=input_ids,
                        tokenizer=tokenizer,
                        **kw,
                    )
                    # LLaDA generate returns full sequence; TraDo adapter returns only gen tokens
                    # We unify by slicing after instruction+prompt tokens.
                    prompt_total_len = int(
                        input_ids.shape[1]) + (0 if args.no_writer_instruction else instr_len)
                    if out_ids.dim() == 2 and out_ids.size(1) >= prompt_total_len:
                        gen_ids = out_ids[0][prompt_total_len:]
                    else:
                        gen_ids = out_ids[0]
                    text = tokenizer.decode(
                        gen_ids, skip_special_tokens=True).strip()
                    generations.append(text)

                    safe_write_line(f_gen, {
                        "dataset_index": dataset_index,
                        "mode": args.mode,
                        "sample_id": k,
                        "generation": text,
                    })

                except Exception as e:
                    failed += 1
                    safe_write_line(f_err, {
                        "dataset_index": dataset_index,
                        "mode": args.mode,
                        "sample_id": k,
                        "error": str(e)[:2000],
                    })
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if generations:
                d1 = distinct_n(generations, 1)
                d2 = distinct_n(generations, 2)
                d3 = distinct_n(generations, 3)
                avg_l = float(np.mean([len(simple_tokenize(g))
                              for g in generations]))
                safe_write_line(f_met, {"dataset_index": dataset_index, "d1": d1,
                                "d2": d2, "d3": d3, "len": avg_l, "failed": failed})

            if args.empty_cache_every > 0 and (local_i + 1) % args.empty_cache_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Done. Output dir:", out_run_dir)


if __name__ == "__main__":
    main()
