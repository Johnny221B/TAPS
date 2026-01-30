#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# -------------------------
# Text cleaning
# -------------------------
_SPECIAL_PATTERNS = [
    re.compile(r"<\|.*?\|>", flags=re.DOTALL),
    re.compile(r"<[^>\n]{1,200}>", flags=re.DOTALL),
]


def clean_generation(text: str) -> str:
    """
    Remove template / special tokens and normalize whitespace.
    """
    if text is None:
        return ""
    t = str(text)

    t = t.replace("Tokenizer", " ")
    t = t.replace("\uFEFF", " ")
    for pat in _SPECIAL_PATTERNS:
        t = pat.sub(" ", t)

    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -------------------------
# Tokenization helpers
# -------------------------
def _simple_tokens(text: str) -> List[str]:
    # lightweight tokenization: words + punctuation
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def _ngrams_from_tokens(toks: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(toks) < n:
        return []
    return [tuple(toks[i: i + n]) for i in range(len(toks) - n + 1)]


# -------------------------
# N-gram intra-diversity (per response)
# -------------------------
def ngram_intra_diversity(text: str, n: int) -> float:
    toks = _simple_tokens(text)
    grams = _ngrams_from_tokens(toks, n)
    if not grams:
        return 0.0
    return len(set(grams)) / float(len(grams))


# -------------------------
# EAD-n (1..5 gram) per prompt (over its multiple samples)
# -------------------------
def ead_ngram_over_set(texts: List[str], n: int, vocab_size_n: int) -> float:
    if vocab_size_n is None or vocab_size_n <= 1:
        return 0.0

    all_ngrams = []
    for t in texts:
        toks = _simple_tokens(t)
        all_ngrams.extend(_ngrams_from_tokens(toks, n))

    Cn = len(all_ngrams)
    if Cn == 0:
        return 0.0
    Nn = len(set(all_ngrams))

    Vn = float(vocab_size_n)
    denom = Vn * (1.0 - ((Vn - 1.0) / Vn) ** Cn)
    if denom <= 0:
        return 0.0
    return float(Nn / denom)


def compute_global_ngram_vocab_sizes(
    groups: Dict[str, List[str]], n_max: int = 5, max_set_size: int = 5_000_000
) -> Dict[int, int]:
    vocab_sets = {n: set() for n in range(1, n_max + 1)}
    capped = {n: False for n in range(1, n_max + 1)}

    for _, texts in groups.items():
        for t in texts:
            if not t.strip():
                continue
            toks = _simple_tokens(t)
            for n in range(1, n_max + 1):
                if capped[n]:
                    continue
                grams = _ngrams_from_tokens(toks, n)
                vocab_sets[n].update(grams)
                if len(vocab_sets[n]) >= max_set_size:
                    capped[n] = True

    return {n: len(vocab_sets[n]) for n in range(1, n_max + 1)}


# -------------------------
# Self-BLEU diversity: 100 - SelfBLEU
# -------------------------
def _self_bleu_avg(texts: List[str]) -> float:
    texts = [t for t in texts if t.strip()]
    if len(texts) <= 1:
        return 0.0

    # 优先尝试 sacrebleu (更快更标准)
    try:
        import sacrebleu
        scores = []
        for i, hyp in enumerate(texts):
            refs = [t for j, t in enumerate(texts) if j != i]
            # sacrebleu expect list of refs
            s = sacrebleu.sentence_bleu(hyp, refs)
            scores.append(float(s.score))
        return float(np.mean(scores))
    except ImportError:
        pass

    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        scores = []
        for i, hyp in enumerate(texts):
            refs = [t for j, t in enumerate(texts) if j != i]
            hyp_tok = hyp.split()
            refs_tok = [r.split() for r in refs]
            s = sentence_bleu(refs_tok, hyp_tok, smoothing_function=smooth)
            scores.append(float(s) * 100.0)
        return float(np.mean(scores))
    except ImportError as e:
        print("Warning: Neither 'sacrebleu' nor 'nltk' found. Self-BLEU will be 0.0.")
        return 0.0


# -------------------------
# Sentence-BERT diversity/dissimilarity
# -------------------------
_SBERTER = {}


def _get_sbert(model_name: str):
    if model_name in _SBERTER:
        return _SBERTER[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        _SBERTER[model_name] = m
        return m
    except ImportError:
        raise RuntimeError(
            "Please install sentence-transformers: pip install sentence-transformers")
    except Exception as e:
        print(f"Error loading SBERT model '{model_name}': {e}")
        return None


def _sbert_pairwise_diversity(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> float:
    texts = [t for t in texts if t.strip()]
    if len(texts) <= 1:
        return 0.0

    model = _get_sbert(model_name)
    if model is None:
        return 0.0

    emb = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
    )
    sim = emb @ emb.T
    k = sim.shape[0]
    iu = np.triu_indices(k, k=1)
    dis = 1.0 - sim[iu]
    return float(np.mean(dis) * 100.0)


# -------------------------
# Pre-clean + save cleaned jsonl
# -------------------------
def save_cleaned_jsonl(
    input_jsonl: str,
    mode_filter: Optional[str] = None,
    text_field: str = "text", 
    output_jsonl: Optional[str] = None,
    clean: bool = True,
) -> str:
    in_path = Path(input_jsonl)
    if output_jsonl is None:
        mode_tag = f".{mode_filter}" if mode_filter else ""
        output_jsonl = str(
            in_path.parent / f"{in_path.stem}.cleaned{mode_tag}.jsonl")

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if mode_filter is not None and obj.get("mode") != mode_filter:
                continue

            if clean:
                content = obj.get(text_field)
                if content is None:
                    content = obj.get("generation", "")

                obj[text_field] = clean_generation(content)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return str(out_path)


# -------------------------
# Loading + grouping
# -------------------------
def load_grouped_generations(
    jsonl_path: str,
    key_fields: Tuple[str, ...] = (
        "prompt_id", "prompt_id_global", "dataset_index"),
    text_field: str = "text",  
    mode_filter: Optional[str] = None,
    clean: bool = True,
) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if mode_filter is not None and obj.get("mode") != mode_filter:
                continue

            key_val = None
            for k in key_fields:
                if k in obj:
                    key_val = obj[k]
                    if key_val is not None:
                        break
            if key_val is None:
                # Fallback
                key_val = obj.get("raw_prompt", "UNKNOWN")

            key = str(key_val)

            txt = obj.get(text_field)
            if txt is None:
                txt = obj.get("generation", "")

            txt = clean_generation(txt) if clean else str(txt)
            groups[key].append(txt)

    return groups


# -------------------------
# Main evaluation function
# -------------------------
def evaluate_diversity(
    jsonl_path: str,
    mode_filter: Optional[str] = None,
    n_list: Tuple[int, ...] = (1, 2, 3),
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sbert_batch_size: int = 32,
    out_json: Optional[str] = None,
    out_per_prompt_jsonl: Optional[str] = None,
    text_field: str = "text"  # Added arg
) -> Dict:

    print(f"Cleaning data from {jsonl_path}...")
    # 1) Auto-clean using text_field
    cleaned_jsonl_path = save_cleaned_jsonl(
        input_jsonl=jsonl_path,
        mode_filter=mode_filter,
        text_field=text_field,
        output_jsonl=None,
        clean=True,
    )
    print(f"Cleaned data saved to {cleaned_jsonl_path}")

    # 2) Load groups
    groups = load_grouped_generations(
        cleaned_jsonl_path,
        mode_filter=None,  # Already filtered during cleaning if needed
        text_field=text_field,
        clean=False,  # Already cleaned
    )

    print(f"Loaded {len(groups)} prompts. Computing metrics...")

    # Global n-gram vocab sizes
    global_vocab_sizes = compute_global_ngram_vocab_sizes(groups, n_max=5)

    per_prompt = []
    sum_ng = {n: 0.0 for n in n_list}
    sum_selfbleu_div = 0.0
    sum_sbert_div = 0.0
    sum_sbert_dis = 0.0

    sum_ead = {n: 0.0 for n in range(1, 6)}
    cnt = 0

    for pid, texts in groups.items():
        texts = [t for t in texts if t.strip()]
        if len(texts) < 2:
            # Need at least 2 samples to compute diversity
            continue

        # N-gram intra-diversity
        ng_scores = {}
        for n in n_list:
            vals = [ngram_intra_diversity(t, n) for t in texts]
            ng_scores[n] = float(np.mean(vals)) if vals else 0.0

        # Self-BLEU
        self_bleu = _self_bleu_avg(texts)
        self_bleu_div = 100.0 - float(self_bleu)

        # SBERT diversity
        sbert_div = _sbert_pairwise_diversity(
            texts, model_name=sbert_model, batch_size=sbert_batch_size
        )
        sbert_dis = float(sbert_div) / 100.0

        # EAD
        ead_scores = {}
        for n in range(1, 6):
            ead_scores[n] = ead_ngram_over_set(
                texts, n=n, vocab_size_n=global_vocab_sizes.get(n, 0)
            )

        rec = {
            "prompt_key": pid,
            "num_samples": len(texts),
            **{f"ngram_intra_div_{n}": ng_scores[n] for n in n_list},
            "self_bleu": float(self_bleu),
            "self_bleu_diversity": float(self_bleu_div),
            "sbert_diversity": float(sbert_div),
            "sent_bert_dissimilarity": float(sbert_dis),
            **{f"ead_{n}": float(ead_scores[n]) for n in range(1, 6)},
        }
        per_prompt.append(rec)

        for n in n_list:
            sum_ng[n] += ng_scores[n]
        sum_selfbleu_div += self_bleu_div
        sum_sbert_div += sbert_div
        sum_sbert_dis += sbert_dis
        for n in range(1, 6):
            sum_ead[n] += ead_scores[n]
        cnt += 1

    results = {
        "input_path": jsonl_path,
        "cleaned_input_path": cleaned_jsonl_path,
        "mode_filter": mode_filter,
        "num_prompts_evaluated": cnt,
        "ead_global_vocab_sizes": {f"V_{n}": int(global_vocab_sizes.get(n, 0)) for n in range(1, 6)},
        "averages": {
            **{f"ngram_intra_div_{n}_avg": (sum_ng[n] / max(cnt, 1)) for n in n_list},
            "self_bleu_diversity_avg": (sum_selfbleu_div / max(cnt, 1)),
            "sbert_diversity_avg": (sum_sbert_div / max(cnt, 1)),
            "sent_bert_dissimilarity_avg": (sum_sbert_dis / max(cnt, 1)),
            **{f"ead_{n}_avg": (sum_ead[n] / max(cnt, 1)) for n in range(1, 6)},
        },
    }

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    if out_per_prompt_jsonl:
        with open(out_per_prompt_jsonl, "w", encoding="utf-8") as f:
            for rec in per_prompt:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return results


def _resolve_output_path(output_arg: str, input_jsonl: str) -> str:
    if output_arg is None:
        return None
    out = Path(output_arg)
    if out.is_absolute() or str(out.parent) not in (".", ""):
        return str(out)
    in_dir = Path(input_jsonl).parent
    return str(in_dir / out.name)


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True,
                    help="path to generations.jsonl from trado eval")
    ap.add_argument("--mode", type=str, default=None,
                    help="optional: normal/token/embedding")
    ap.add_argument("--out_json", type=str, default="diversity_summary.json")
    ap.add_argument("--out_per_prompt", type=str,
                    default="diversity_per_prompt.jsonl")
    ap.add_argument("--sbert_model", type=str,
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sbert_batch_size", type=int, default=32)
    ap.add_argument("--ngrams", type=str, default="1,2,3")

    # 适配 Trado: 允许用户指定字段名，但默认改为 text
    ap.add_argument("--text_field", type=str, default="text",
                    help="JSON key for generated text")

    args = ap.parse_args()

    n_list = tuple(int(x) for x in args.ngrams.split(",") if x.strip())

    out_json = _resolve_output_path(args.out_json, args.input)
    out_per_prompt = _resolve_output_path(args.out_per_prompt, args.input)

    res = evaluate_diversity(
        jsonl_path=args.input,
        mode_filter=args.mode,
        n_list=n_list,
        sbert_model=args.sbert_model,
        sbert_batch_size=args.sbert_batch_size,
        out_json=out_json,
        out_per_prompt_jsonl=out_per_prompt,
        text_field=args.text_field
    )

    print(json.dumps(res, ensure_ascii=False, indent=2))
