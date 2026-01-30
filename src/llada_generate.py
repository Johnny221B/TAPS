# generate.py (Noise Window + Quality Protection for embed_gaussian)
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional

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


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return logits
    U = torch.rand_like(logits, dtype=torch.float32)
    g = -torch.log(-torch.log(U + 1e-20) + 1e-20)
    return logits / float(temperature) + g


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(
        0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def _anneal_strength(progress_0to1: float, max_strength: float, anneal: str = "cosine") -> float:
    # progress 是全局进度；即便 start 前不加噪，一旦开始也按全局进度退火，避免 start 处突变
    p = float(np.clip(progress_0to1, 0.0, 1.0))
    if anneal == "linear":
        return max_strength * (1.0 - p)
    elif anneal == "cosine":
        return max_strength * float(np.cos(0.5 * np.pi * p))
    else:
        raise NotImplementedError(f"Unknown anneal: {anneal}")


def _make_prompt_valid_mask(prompt_len: int, attention_mask: Optional[torch.Tensor], device, bsz: int) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones((bsz, prompt_len), dtype=torch.bool, device=device)
    return attention_mask[:, :prompt_len].bool()


def _try_token_id(tokenizer, s: str) -> Optional[int]:
    try:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is None or (hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id):
            return None
        return int(tid)
    except:
        return None


def _collect_ban_token_ids(tokenizer, ban_extra_strings: Optional[List[str]] = None) -> List[int]:
    candidates = ["<|startoftext|>", "<|eot_id|>",
                  "<|start_header_id|>", "<|end_header_id|>"]
    if ban_extra_strings:
        candidates.extend(ban_extra_strings)
    ban = set()
    for s in candidates:
        tid = _try_token_id(tokenizer, s)
        if tid is not None:
            ban.add(tid)
    for attr in ["bos_token_id", "eos_token_id", "sep_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            ban.add(int(tid))
    return sorted(ban)


def _apply_ban_ids_to_logits(logits: torch.Tensor, ban_ids: List[int]) -> None:
    if not ban_ids:
        return
    logits[:, :, ban_ids] = -torch.inf


def _truncate_on_prompt_leak(x: torch.Tensor, prompt_len_total: int, signature_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    bsz = x.size(0)
    gen = x[:, prompt_len_total:]
    sig_len = signature_ids.size(1)
    if sig_len <= 0 or gen.size(1) < sig_len:
        return x
    for b in range(bsz):
        sig = signature_ids[b]
        gb = gen[b]
        found_pos = None
        for pos in range(0, gb.numel() - sig_len + 1):
            if torch.equal(gb[pos:pos + sig_len], sig):
                found_pos = pos
                break
        if found_pos is not None:
            x[b, prompt_len_total + found_pos:] = eos_id
    return x


# =========================
# Quality protection helpers
# =========================
def _cads_rescale_patch(perturbed: torch.Tensor, clean: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    CADS-style rescaling: match perturbed patch mean/std to clean patch mean/std.
    Operate per-sample over (token, hidden) dims.
    perturbed/clean: [B, T, D]
    """
    clean_mean = clean.mean(dim=(1, 2), keepdim=True)
    clean_std = clean.std(dim=(1, 2), keepdim=True) + eps

    pert_mean = perturbed.mean(dim=(1, 2), keepdim=True)
    pert_std = perturbed.std(dim=(1, 2), keepdim=True) + eps

    normed = (perturbed - pert_mean) / pert_std
    return normed * clean_std + clean_mean


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    tokenizer,
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    add_writer_instruction: bool = True,
    ban_chat_special_tokens: bool = True,
    ban_extra_strings: Optional[List[str]] = None,
    prompt_leak_guard: bool = True,
    leak_signature_len: int = 24,
    # --- Noise Params ---
    cond_noise_mode: str = "none",
    cond_noise_anneal: str = "cosine",
    cond_noise_start: float = 0.0,
    cond_noise_until: float = 1.0,
    cond_prompt_mask_ratio: float = 0.2,
    cond_embed_noise_std: float = 0.05,
    cond_embed_impl: str = "hook",

    # --- NEW: Quality protection for embed_gaussian ---
    # 默认保持“原始行为不变”：rescale=False, psi=1.0, fixed_noise=False
    cond_embed_enable_rescale: bool = True,
    cond_embed_psi: float = 1.0,
    cond_embed_fixed_noise: bool = False,
):
    device = model.device
    bsz = prompt_ids.shape[0]
    raw_prompt_len = int(prompt_ids.shape[1])
    eos_id = getattr(tokenizer, "eos_token_id", mask_id) or mask_id

    # 1. Build Instruction
    if add_writer_instruction:
        instr_ids = tokenizer(WRITER_INSTR_TEXT, add_special_tokens=False, return_tensors="pt")[
            "input_ids"].to(device)
        instr_len = int(instr_ids.shape[1])
        instr_ids = instr_ids.expand(bsz, -1)
        prompt = torch.cat([instr_ids, prompt_ids.to(device)], dim=1)
        prompt_len = int(prompt.shape[1])
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            instr_attn = torch.ones(
                (bsz, instr_len), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([instr_attn, attention_mask], dim=1)
    else:
        instr_len = 0
        prompt = prompt_ids.to(device)
        prompt_len = int(prompt.shape[1])
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    # 2. Init
    x = torch.full((bsz, prompt_len + gen_length), mask_id,
                   dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones(
                (bsz, gen_length), dtype=attention_mask.dtype, device=device)],
            dim=-1
        )

    prompt_index = torch.zeros_like(x, dtype=torch.bool, device=device)
    prompt_index[:, :prompt_len] = True

    valid_prompt = _make_prompt_valid_mask(
        prompt_len, attention_mask, device, bsz)  # (目前仍未使用，保留不动)
    ban_ids = _collect_ban_token_ids(
        tokenizer, ban_extra_strings) if ban_chat_special_tokens else []

    # 3. Hook (embed_gaussian only)
    hook_handle = None
    hook_ctx = {}

    def _install_embed_hook():
        nonlocal hook_handle
        emb = model.get_input_embeddings()

        # hook 内 cache（window 内固定噪声方向）
        hook_ctx["in_window"] = False
        hook_ctx["noise_cache"] = None

        def _hook(module, inp, out):
            sigma = float(hook_ctx.get("sigma", 0.0))
            if sigma <= 0.0:
                # 退出 window 时清理缓存，避免跨 window 复用
                hook_ctx["in_window"] = False
                hook_ctx["noise_cache"] = None
                return out

            cond_bsz = hook_ctx.get("cond_bsz", out.shape[0])
            pl = hook_ctx.get("prompt_len", prompt_len)
            il = hook_ctx.get("instr_len", 0)
            if out.shape[1] < pl:
                return out

            # Slice Target (User Prompt Only)
            out_cond = out[:cond_bsz]
            target_region = out_cond[:, il:pl, :]  # [B, T_user, D]

            # --- Adaptive Noise Scaling (Short Prompt Protection) ---
            user_prompt_len = float(pl - il)
            threshold_len = 32.0
            raw_ratio = user_prompt_len / threshold_len if threshold_len > 0 else 1.0
            length_scale = max(0.5, min(1.0, raw_ratio))  # 保底50%
            actual_sigma = float(sigma) * length_scale
            # ------------------------------------------------------

            # --- Quality Protection (optional) ---
            # (1) window 内固定噪声方向：只缓存 N(0,1)，每步乘 actual_sigma
            if cond_embed_fixed_noise:
                if (not hook_ctx.get("in_window", False)) or (hook_ctx.get("noise_cache") is None) or (hook_ctx["noise_cache"].shape != target_region.shape):
                    hook_ctx["in_window"] = True
                    hook_ctx["noise_cache"] = torch.randn_like(
                        target_region)  # N(0,1)
                noise = hook_ctx["noise_cache"] * actual_sigma
            else:
                noise = torch.randn_like(target_region) * actual_sigma

            perturbed_region = target_region + noise

            # (2) CADS-style rescale：匹配 mean/std（强噪声更稳）
            if cond_embed_enable_rescale:
                perturbed_region = _cads_rescale_patch(
                    perturbed_region, target_region, eps=1e-6)

            # (3) psi-mix：可控拉回 clean（保护质量）
            psi = float(cond_embed_psi)
            if psi < 0.0:
                psi = 0.0
            elif psi > 1.0:
                psi = 1.0
            mixed_region = psi * perturbed_region + (1.0 - psi) * target_region

            # (4) 保留你原来的 per-token norm projection（进一步稳）
            orig_norm = torch.norm(target_region, dim=-1, keepdim=True)
            mixed_norm = torch.norm(mixed_region, dim=-1, keepdim=True)
            final_region = (mixed_region / (mixed_norm + 1e-6)) * orig_norm

            out2 = out.clone()
            out2[:cond_bsz, il:pl, :] = final_region
            return out2

        hook_handle = emb.register_forward_hook(_hook)

    def _remove_embed_hook():
        nonlocal hook_handle
        if hook_handle:
            hook_handle.remove()
            hook_handle = None

    if cond_noise_mode == "embed_gaussian" and cond_embed_impl == "hook":
        _install_embed_hook()

    # 4. Loop
    try:
        num_blocks = gen_length // block_length
        steps_per_block = steps // num_blocks
        total_steps = steps_per_block * num_blocks

        for num_block in range(num_blocks):
            block_slice = slice(prompt_len + num_block * block_length,
                                prompt_len + (num_block + 1) * block_length)
            block_mask_index = (x[:, block_slice] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                global_step = num_block * steps_per_block + i
                progress = (global_step + 1) / float(total_steps)  # 0.0 -> 1.0

                # --- WINDOW LOGIC: Start <= Progress <= Until ---
                apply_noise = (cond_noise_mode != "none") and \
                              (progress >= float(cond_noise_start)) and \
                              (progress <= float(cond_noise_until))

                sigma_t = 0.0
                mask_ratio_t = 0.0
                if apply_noise:
                    if cond_noise_mode == "prompt_mask":
                        mask_ratio_t = _anneal_strength(progress, float(
                            cond_prompt_mask_ratio), anneal=cond_noise_anneal)
                    elif cond_noise_mode == "embed_gaussian":
                        sigma_t = _anneal_strength(progress, float(
                            cond_embed_noise_std), anneal=cond_noise_anneal)

                mask_index = (x == mask_id)

                if cond_noise_mode == "prompt_mask" and mask_ratio_t > 0:
                    x_cond = x.clone()
                    # Apply mask only to user prompt
                    r = torch.rand(
                        (bsz, prompt_len - instr_len), device=device)
                    m = (r < mask_ratio_t)
                    x_cond[:, instr_len:prompt_len][m] = mask_id
                else:
                    x_cond = x

                if cond_noise_mode == "embed_gaussian" and cond_embed_impl == "hook":
                    hook_ctx["sigma"] = float(sigma_t)
                    hook_ctx["prompt_len"] = prompt_len
                    hook_ctx["instr_len"] = instr_len
                    hook_ctx["cond_bsz"] = bsz
                else:
                    # 非 embed_gaussian 时确保 sigma=0（避免残留）
                    if hook_ctx is not None:
                        hook_ctx["sigma"] = 0.0

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_in = torch.cat([x_cond, un_x], dim=0)
                    att_in = torch.cat(
                        [attention_mask, attention_mask], dim=0) if attention_mask is not None else None
                    logits = model(x_in, attention_mask=att_in).logits
                    logits_c, logits_u = torch.chunk(logits, 2, dim=0)
                    logits = logits_u + \
                        float(cfg_scale) * (logits_c - logits_u)
                else:
                    logits = model(
                        x_cond, attention_mask=attention_mask).logits

                if ban_ids:
                    _apply_ban_ids_to_logits(logits, ban_ids)

                logits[:, :, mask_id] = -torch.inf
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(
                        p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt_len + (num_block + 1) * block_length:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].item())
                    if k <= 0:
                        continue
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        if prompt_leak_guard and eos_id != mask_id:
            sig_len = max(0, min(int(leak_signature_len), raw_prompt_len))
            if sig_len > 0:
                signature = x[:, instr_len: instr_len + sig_len]
                x = _truncate_on_prompt_leak(x, prompt_len, signature, eos_id)

        return x

    finally:
        _remove_embed_hook()
