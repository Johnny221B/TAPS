export HF_HOME=.cache

CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.writingprompts.eval_llada_wp \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --mode embedding \
  --dataset euclaise/writingprompts \
  --num_prompts 50 \
  --num_samples 16 \
  --temperature 0.7 \
  --cfg 0.0 \
  --cond_embed_noise_std 0.35 \
  --cond_noise_start 0.05 \
  --cond_noise_until 0.95 \
  --cond_embed_impl hook \
  --steps 512 \
  --gen_length 512 \
  --block_length 256 \
  --empty_cache_every 20