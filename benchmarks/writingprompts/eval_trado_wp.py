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
from transformers import AutoTokenizer, AutoModelForCausalLM

# [IMPORTANT] Import from our NEW file
from src.methods.adapters.trado_generate import generate, WRITER_INSTR_TEXT
# from generate_trado import generate, WRITER_INSTR_TEXT


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
    s = str(raw_prompt).strip()
    s = re.sub(r"^\[\s*wp\s*\]\s*", "", s, flags=re.IGNORECASE)
    return s


def ensure_dir(path: str): os.makedirs(path, exist_ok=True)


def safe_write_line(f, obj: Any):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def main():
    ap = argparse.ArgumentParser("Trado Diversity Eval")

    ap.add_argument("--model_path", type=str,
                    default="/mnt/data/wujx/DLM/models/Gen-Verse__TraDo-8B-Instruct")
    ap.add_argument("--dataset", type=str, default="euclaise/writingprompts")
    ap.add_argument("--split", type=str, default="")
    ap.add_argument("--prompt_column", type=str, default="")
    ap.add_argument("--num_prompts", type=int, default=10)
    ap.add_argument("--start_index", type=int, default=0)

    # Defaults for Trado
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=4)
    ap.add_argument("--remasking", type=str, default="low_confidence",
                    choices=["low_confidence", "random"])
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--empty_cache_every", type=int, default=10)
    ap.add_argument("--mode", type=str, default="normal",
                    choices=["normal", "token", "embedding"])
    # noise params
    ap.add_argument("--cond_noise_start", type=float, default=0.05)
    ap.add_argument("--cond_noise_until", type=float, default=0.95)
    ap.add_argument("--cond_noise_anneal", type=str, default="cosine")
    ap.add_argument("--cond_prompt_mask_ratio", type=float, default=0.05)
    ap.add_argument("--cond_embed_noise_std", type=float, default=0.20)
    ap.add_argument("--cond_embed_psi", type=float, default=1.0)  # <-- NEW
    ap.add_argument("--cond_embed_impl", type=str, default="hook")
    ap.add_argument("--mask_id", type=int, default=151669)
    ap.add_argument("--out_dir", type=str, default="diversity_runs_trado")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--no_writer_instruction", action="store_true")
    ap.add_argument("--no_ban_chat_tokens", action="store_true")
    # Sampling
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--min_p", type=float, default=0.0)

    args = ap.parse_args()

    accelerator = accelerate.Accelerator()
    rank = accelerator.process_index
    device = accelerator.device

    # Setup Run Name
    ensure_dir(args.out_dir)
    run_name_file = os.path.join(args.out_dir, "_current_run_name.txt")

    if args.run_name.strip():
        run_name = args.run_name.strip()
        if accelerator.is_main_process:
            with open(run_name_file, "w") as f:
                f.write(run_name)
    else:
        if accelerator.is_main_process:
            run_name = f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(run_name_file, "w") as f:
                f.write(run_name)
        accelerator.wait_for_everyone()
        time.sleep(0.5)
        with open(run_name_file, "r") as f:
            run_name = f.read().strip()

    out_run_dir = os.path.join(args.out_dir, run_name)

    # Save Config
    if accelerator.is_main_process:
        ensure_dir(out_run_dir)
        config_path = os.path.join(out_run_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        print(f"Config saved to {config_path}")

    accelerator.wait_for_everyone()

    # Load Model
    set_seed(args.seed + rank, deterministic=args.deterministic)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype="float16"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)

    # Load Data
    ds = load_dataset(args.dataset, split=args.split or "train")
    prompt_col = args.prompt_column.strip() or pick_prompt_column(ds.column_names)

    # Select Subset
    global_indices = list(range(args.start_index, min(
        len(ds), args.start_index + args.num_prompts)))
    my_indices = global_indices[rank::accelerator.num_processes]
    ds_sub = ds.select(my_indices)

    gen_path = os.path.join(out_run_dir, f"generations.rank{rank}.jsonl")
    metric_path = os.path.join(out_run_dir, f"metrics.rank{rank}.jsonl")

    with open(gen_path, "w", encoding="utf-8") as f_gen, open(metric_path, "w", encoding="utf-8") as f_met:
        for idx_in_sub, row in tqdm(enumerate(ds_sub), total=len(ds_sub), disable=not accelerator.is_local_main_process):
            dataset_index = my_indices[idx_in_sub]
            prompt_text = sanitize_wp(row[prompt_col])

            enc = tokenizer([prompt_text], add_special_tokens=False,
                            padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            generations = []
            for k in range(args.num_samples):
                set_seed(args.seed + dataset_index*1000 + k)

                # Default kwargs
                kwargs = {}

                # -------------------------------------------------
                # [MODIFIED] Mode Selection & Baseline Logic
                # -------------------------------------------------
                if args.mode == "normal":
                    # Normal Mode = Baseline = Pure Temperature Sampling
                    # Force disable Top-K/P/Min-P
                    kwargs.update({
                        "top_k": 0,
                        "top_p": 1.0,
                        "min_p": 0.0,
                        "cond_noise_mode": "none"
                    })
                else:
                    # Other modes (token/embedding) use CLI args for sampling
                    kwargs.update({
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "min_p": args.min_p,
                        "cond_noise_mode": "none"  # default
                    })

                # Add noise params for non-normal modes
                if args.mode == "token":
                    kwargs.update({
                        "cond_noise_mode": "prompt_mask",
                        "cond_prompt_mask_ratio": args.cond_prompt_mask_ratio,
                        "cond_noise_start": args.cond_noise_start,
                        "cond_noise_until": args.cond_noise_until,
                        "cond_noise_anneal": args.cond_noise_anneal,
                    })
                elif args.mode == "embedding":
                    kwargs.update({
                        "cond_noise_mode": "embed_gaussian",
                        "cond_embed_noise_std": args.cond_embed_noise_std,
                        "cond_noise_start": args.cond_noise_start,
                        "cond_noise_until": args.cond_noise_until,
                        "cond_noise_anneal": args.cond_noise_anneal,
                        "cond_embed_impl": args.cond_embed_impl,
                        "cond_embed_psi": args.cond_embed_psi,
                    })

                try:
                    # Call generate
                    out_gen_ids = generate(
                        model=model, prompt_ids=input_ids, tokenizer=tokenizer,
                        add_writer_instruction=(
                            not args.no_writer_instruction),
                        ban_chat_special_tokens=(not args.no_ban_chat_tokens),
                        steps=args.steps,
                        gen_length=args.gen_length,
                        block_length=args.block_length,
                        temperature=args.temperature,
                        mask_id=args.mask_id,
                        **kwargs
                    )

                    text = tokenizer.decode(
                        out_gen_ids[0], skip_special_tokens=True)
                    text = text.replace('<|endoftext|>', '').strip()
                    generations.append(text)

                    safe_write_line(f_gen, {
                        "prompt_id": dataset_index,
                        "sample_id": k,
                        "mode": args.mode,
                        "text": text
                    })
                except Exception as e:
                    print(
                        f"Error rank{rank} prompt{dataset_index} sample{k}: {e}")

            if generations:
                d1 = distinct_n(generations, 1)
                d2 = distinct_n(generations, 2)
                d3 = distinct_n(generations, 3)
                avg_l = np.mean([len(simple_tokenize(g)) for g in generations])
                safe_write_line(
                    f_met, {"prompt_id": dataset_index, "d1": d1, "d2": d2, "d3": d3, "len": avg_l})

            if args.empty_cache_every > 0:
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Done. Output dir:", args.out_dir)


if __name__ == "__main__":
    main()
