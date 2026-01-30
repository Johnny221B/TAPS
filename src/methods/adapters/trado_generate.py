import torch
import torch.nn.functional as F
import numpy as np
from transformers.cache_utils import DynamicCache

# ==========================================
# 0. Constants
# ==========================================
WRITER_INSTR_TEXT = (
    "You are a professional fiction writer.\n"
    "Task: Continue the story based on the writing prompt.\n"
    "Rules:\n"
    "1) Do NOT repeat or paraphrase the writing prompt. Do NOT quote it.\n"
    "2) Do NOT include any tags like [WP] or any template/special tokens.\n"
    "3) Start writing the story immediately. Maintain coherence and add new events.\n"
    "4) If you feel stuck, invent plausible details instead of reusing prompt text.\n"
    "\n"
    "Write the continuation:\n"
)

# ==========================================
# 1. Helpers
# ==========================================


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return logits
    U = torch.rand_like(logits, dtype=torch.float32)
    g = -torch.log(-torch.log(U + 1e-20) + 1e-20)
    return logits / float(temperature) + g


def _anneal_strength(progress_0to1: float, max_strength: float, anneal: str = "cosine") -> float:
    p = float(np.clip(progress_0to1, 0.0, 1.0))
    if anneal == "linear":
        return max_strength * (1.0 - p)
    elif anneal == "cosine":
        return max_strength * float(np.cos(0.5 * np.pi * p))
    else:
        return max_strength * float(np.cos(0.5 * np.pi * p))


def top_k_logits(logits, k: int):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def top_p_logits(logits, p: float):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool), -
        1, sorted_indices, sorted_mask
    )
    return logits.masked_fill(mask_indices, float("-inf"))


def min_p_logits(logits, min_p: float = 0.0):
    if min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    scaled_min_p = min_p * top_probs
    tokens_to_remove = probs < scaled_min_p
    return logits.masked_fill(tokens_to_remove, float("-inf"))


def get_num_transfer_tokens(block_length: int, steps: int):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens

# ==========================================
# 2. Core Generation Function
# ==========================================


@torch.no_grad()
def block_diffusion_generate_diversity(
    model,
    prompt,
    mask_id,
    tokenizer=None,
    gen_length=128,
    block_length=4,
    denoising_steps=4,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    min_p=0.0,
    cfg_scale=0.0,
    remasking_strategy="low_confidence_dynamic",
    confidence_threshold=0.9,
    cond_noise_mode="none",
    cond_noise_anneal="cosine",
    cond_noise_start=0.0,
    cond_noise_until=1.0,
    cond_prompt_mask_ratio=0.1,
    cond_embed_noise_std=0.05,
    cond_embed_impl="hook",
    ban_token_ids=None,
    # ---- Optional quality controls (safe defaults) ----
    # ψ mixing in [0,1]; 1.0 is most stable
    cond_embed_psi: float = 1.0,
    cond_embed_rescale: bool = True,             # CADS-style mean/std matching
    # clamp σ as ratio of embedding weight std
    cond_embed_sigma_max_ratio: float = 0.30,
    cond_embed_eps: float = 1e-6,
):
    """
    embed_gaussian strategy in this version (TraDo-friendly):
      - interpret cond_noise_start/until as *block-window* (over generated blocks)
      - only inject noise at step==0 inside each noisy block
      - noise magnitude decays across noisy blocks using cond_noise_anneal
      - hook: CADS-style rescale + ψ mixing
      - σ: relative to embedding scale + clamp
    """

    model.eval()
    device = model.device
    input_ids = prompt["input_ids"]
    bsz = input_ids.shape[0]
    prompt_length = input_ids.shape[1]
    instr_len = prompt.get("instr_len", 0)

    # --- Hook context ---
    hook_ctx = {
        "sigma": 0.0,
        "prompt_len": prompt_length,
        "instr_len": instr_len,
        "psi": float(cond_embed_psi),
        "rescale": bool(cond_embed_rescale),
        "eps": float(cond_embed_eps),
    }
    hook_handle = None

    def _embed_hook(module, inp, out):
        sigma = float(hook_ctx.get("sigma", 0.0))
        if sigma <= 0.0:
            return out

        pl = int(hook_ctx["prompt_len"])
        il = int(hook_ctx["instr_len"])
        if out.shape[1] < pl:
            return out

        psi = float(hook_ctx.get("psi", 1.0))
        psi = max(0.0, min(1.0, psi))
        rescale = bool(hook_ctx.get("rescale", True))
        eps = float(hook_ctx.get("eps", 1e-6))

        # Only perturb user prompt region (exclude instruction)
        target = out[:, il:pl, :]
        if target.numel() == 0:
            return out

        target_f = target.float()
        noisy = target_f + torch.randn_like(target_f) * sigma

        if rescale:
            # Match mean/std (per hidden dim) back to clean target stats
            mu_c = target_f.mean(dim=(0, 1), keepdim=True)
            std_c = target_f.std(dim=(0, 1), keepdim=True).clamp_min(eps)

            mu_n = noisy.mean(dim=(0, 1), keepdim=True)
            std_n = noisy.std(dim=(0, 1), keepdim=True).clamp_min(eps)

            noisy = (noisy - mu_n) / std_n * std_c + mu_c

        # ψ mixing for quality control
        final_f = psi * noisy + (1.0 - psi) * target_f

        out_new = out.clone()
        out_new[:, il:pl, :] = final_f.to(dtype=out.dtype)
        return out_new

    if cond_noise_mode == "embed_gaussian":
        hook_handle = model.get_input_embeddings().register_forward_hook(_embed_hook)

    try:
        # ---- Layout ----
        num_blocks = (prompt_length + gen_length +
                      block_length - 1) // block_length
        total_length = num_blocks * block_length

        prefill_blocks = prompt_length // block_length
        gen_blocks_count = max(0, num_blocks - prefill_blocks)

        x = torch.full((bsz, total_length), mask_id,
                       dtype=torch.long, device=device)
        x[:, :prompt_length] = input_ids

        position_ids = torch.arange(total_length, device=device).unsqueeze(0)

        block_mask = torch.tril(torch.ones(
            num_blocks, num_blocks, device=device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
        )

        past_key_values = DynamicCache()

        # ---- Prefill ----
        prefill_length = prefill_blocks * block_length
        if prefill_length > 0:
            cur_x = x[:, :prefill_length]
            cur_attn_mask = block_diffusion_attention_mask[:,
                                                           :prefill_length, :prefill_length]
            if cur_attn_mask.dim() == 3:
                cur_attn_mask = cur_attn_mask[:, None, :, :]
            cur_pos = position_ids[:, :prefill_length]
            model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_pos,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )

        # ---- Per-step transfer count (TraDo fill schedule) ----
        # note: your original code uses steps+1 loop; we keep it
        num_transfer_tokens = get_num_transfer_tokens(
            block_length, denoising_steps)

        # ---- Compute embedding scale proxy for relative sigma ----
        # alpha = cond_embed_noise_std; sigma_base = alpha * emb_std; clamp sigma
        emb_w = model.get_input_embeddings().weight
        emb_std = float(emb_w.float().std().item())
        sigma_base_raw = float(cond_embed_noise_std) * emb_std
        sigma_max = float(cond_embed_sigma_max_ratio) * emb_std
        sigma_base_raw = min(sigma_base_raw, sigma_max)

        # ---- Block-window for noise (interpret start/until over generated blocks) ----
        # e.g., 50 gen blocks; want first 30 => until = 0.6
        start_frac = float(cond_noise_start)
        until_frac = float(cond_noise_until)
        start_frac = max(0.0, min(1.0, start_frac))
        until_frac = max(0.0, min(1.0, until_frac))

        # Map fraction -> [start_block, end_block] (inclusive)
        # Define gen_block_id in [0..gen_blocks_count-1]
        if gen_blocks_count <= 0 or until_frac <= start_frac:
            noisy_block_start = 0
            noisy_block_end = -1
        else:
            noisy_block_start = int(np.floor(start_frac * gen_blocks_count))
            noisy_block_end = int(np.ceil(until_frac * gen_blocks_count)) - 1
            noisy_block_start = max(
                0, min(gen_blocks_count - 1, noisy_block_start))
            noisy_block_end = max(-1,
                                  min(gen_blocks_count - 1, noisy_block_end))

        noisy_blocks_total = max(0, noisy_block_end - noisy_block_start + 1)

        # ------------ Main Loop ------------
        for num_block in range(prefill_blocks, num_blocks):
            start_idx = num_block * block_length
            end_idx = (num_block + 1) * block_length

            gen_block_id = num_block - prefill_blocks  # 0-based within generated blocks
            block_is_noisy = (
                (cond_noise_mode != "none")
                and (gen_block_id >= noisy_block_start)
                and (gen_block_id <= noisy_block_end)
            )

            # precompute this block's sigma (decays across noisy blocks)
            # within_block_progress: 0 at first noisy block -> 1 at last noisy block
            if block_is_noisy and noisy_blocks_total > 1:
                within_block_progress = (
                    gen_block_id - noisy_block_start) / float(noisy_blocks_total - 1)
            elif block_is_noisy and noisy_blocks_total == 1:
                within_block_progress = 0.0
            else:
                within_block_progress = 1.0  # unused

            # Decay across blocks: early noisy blocks stronger, later weaker
            sigma_block = 0.0
            if block_is_noisy and cond_noise_mode == "embed_gaussian":
                sigma_block = _anneal_strength(
                    within_block_progress, sigma_base_raw, anneal=cond_noise_anneal)

            block_kv_stored = False

            for step in range(denoising_steps + 1):
                # --- Key design: ONLY inject noise at the FIRST step of each noisy block ---
                # apply_noise = block_is_noisy and (step < 2) 这里可以调整
                apply_noise = block_is_noisy and (step == 0)

                # Set hook sigma
                hook_ctx["sigma"] = float(sigma_block if apply_noise else 0.0)

                # --- Prepare input ---
                if apply_noise:
                    # Recompute full prefix so prompt noise can influence logits
                    past_kv_current = None
                    input_seq = x[:, :end_idx].clone()
                    input_mask = block_diffusion_attention_mask[:,
                                                                :end_idx, :end_idx]
                    input_pos = position_ids[:, :end_idx]
                else:
                    # Standard cached inference
                    past_kv_current = past_key_values
                    input_seq = x[:, start_idx:end_idx]
                    input_mask = block_diffusion_attention_mask[:,
                                                                start_idx:end_idx, :end_idx]
                    input_pos = position_ids[:, start_idx:end_idx]

                    # Early exit if no masks left in this block
                    if (input_seq == mask_id).sum() == 0:
                        if input_mask.dim() == 3:
                            input_mask = input_mask[:, None, :, :]
                        model(
                            input_seq,
                            attention_mask=input_mask,
                            position_ids=input_pos,
                            past_key_values=past_key_values,
                            use_cache=True,
                            store_kv=True,
                        )
                        block_kv_stored = True
                        break

                if input_mask.dim() == 3:
                    input_mask = input_mask[:, None, :, :]

                # --- Forward ---
                logits = model(
                    input_seq,
                    attention_mask=input_mask,
                    position_ids=input_pos,
                    past_key_values=past_kv_current,
                    use_cache=True,
                    store_kv=False,
                ).logits

                if apply_noise:
                    logits = logits[:, -block_length:, :]

                if ban_token_ids:
                    logits[:, :, ban_token_ids] = float("-inf")

                # Sampling filters
                logits = top_k_logits(logits, top_k)
                logits = min_p_logits(logits, min_p)
                logits = top_p_logits(logits, top_p)

                if temperature > 0:
                    logits_noise = add_gumbel_noise(logits, temperature)
                    x0 = torch.argmax(logits_noise, dim=-1)
                    x0_p = torch.gather(
                        F.softmax(logits, dim=-1), -1, x0.unsqueeze(-1)).squeeze(-1)
                else:
                    x0 = torch.argmax(logits, dim=-1)
                    x0_p = torch.ones_like(x0, dtype=torch.float)

                # Remasking / transfer
                cur_block_x = x[:, start_idx:end_idx]
                mask_index = (cur_block_x == mask_id)
                x0 = torch.where(mask_index, x0, cur_block_x)

                confidence = torch.where(
                    mask_index, x0_p, torch.tensor(-float("inf"), device=device))
                k = num_transfer_tokens[step].item() if step < len(
                    num_transfer_tokens) else 0

                if k > 0:
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(bsz):
                        _, select_index = torch.topk(
                            confidence[j], k=min(k, block_length))
                        transfer_index[j, select_index] = True
                    cur_block_x[transfer_index] = x0[transfer_index]
                    x[:, start_idx:end_idx] = cur_block_x

            # --- End of Block: store KV ---
            if not block_kv_stored:
                hook_ctx["sigma"] = 0.0
                final_input = x[:, start_idx:end_idx]
                final_mask = block_diffusion_attention_mask[:,
                                                            start_idx:end_idx, :end_idx]
                if final_mask.dim() == 3:
                    final_mask = final_mask[:, None, :, :]
                final_pos = position_ids[:, start_idx:end_idx]
                model(
                    final_input,
                    attention_mask=final_mask,
                    position_ids=final_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                )

        return x

    finally:
        if hook_handle:
            hook_handle.remove()


# ==========================================
# 3. Adapter for your pipeline (unchanged API)
# ==========================================
@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    tokenizer,
    attention_mask=None,
    add_writer_instruction: bool = False,
    ban_chat_special_tokens: bool = False,
    ban_extra_strings=None,
    prompt_leak_guard: bool = False,
    leak_signature_len: int = 24,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 4,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 151669,
    **kwargs
):
    device = model.device
    bsz = prompt_ids.shape[0]

    # 1) Instruction + user prompt
    final_input_ids = prompt_ids
    instr_len = 0
    if add_writer_instruction:
        instr_tokens = tokenizer(
            WRITER_INSTR_TEXT, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        instr_len = instr_tokens.shape[1]
        instr_tokens = instr_tokens.expand(bsz, -1)
        final_input_ids = torch.cat([instr_tokens, prompt_ids], dim=1)

    exact_prompt_len = final_input_ids.shape[1]

    # 2) ban tokens
    ban_ids = []
    if ban_chat_special_tokens:
        candidates = [
            "<|startoftext|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
            "<|im_start|>", "<|im_end|>"
        ]
        if ban_extra_strings:
            candidates.extend(ban_extra_strings)
        for s in candidates:
            tid = tokenizer.convert_tokens_to_ids(s)
            if tid is not None and isinstance(tid, int):
                ban_ids.append(tid)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
            ban_ids.append(tokenizer.eos_token_id)

    prompt_dict = {"input_ids": final_input_ids, "instr_len": instr_len}

    full_sequence = block_diffusion_generate_diversity(
        model=model,
        prompt=prompt_dict,
        mask_id=mask_id,
        tokenizer=tokenizer,
        gen_length=gen_length,
        block_length=block_length,
        denoising_steps=steps,
        temperature=temperature,
        top_k=kwargs.get("top_k", 0),
        top_p=kwargs.get("top_p", 1.0),
        min_p=kwargs.get("min_p", 0.0),
        cfg_scale=cfg_scale,
        ban_token_ids=ban_ids,
        cond_noise_mode=kwargs.get("cond_noise_mode", "none"),
        cond_noise_start=kwargs.get("cond_noise_start", 0.0),
        cond_noise_until=kwargs.get("cond_noise_until", 1.0),
        cond_noise_anneal=kwargs.get("cond_noise_anneal", "cosine"),
        cond_prompt_mask_ratio=kwargs.get("cond_prompt_mask_ratio", 0.0),
        cond_embed_noise_std=kwargs.get("cond_embed_noise_std", 0.0),
        cond_embed_impl=kwargs.get("cond_embed_impl", "hook"),
        # optional knobs (pipeline can ignore)
        cond_embed_psi=kwargs.get("cond_embed_psi", 1.0),
        cond_embed_rescale=kwargs.get("cond_embed_rescale", True),
        cond_embed_sigma_max_ratio=kwargs.get(
            "cond_embed_sigma_max_ratio", 0.30),
        cond_embed_eps=kwargs.get("cond_embed_eps", 1e-6),
    )

    return full_sequence[:, exact_prompt_len:]
