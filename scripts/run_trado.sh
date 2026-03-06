export HF_HOME=.cache

CUDA_VISIBLE_DEVICES=1 uv run python -m benchmarks.writingprompts.eval_trado_wp \
  --run_name trado_embedding_run \
  --mode embedding \
  --model_path Gen-Verse/TraDo-8B-Instruct \
  --num_prompts 25 \
  --num_samples 16 \
  --gen_length 512 \
  --steps 4 \
  --block_length 4 \
  --temperature 0.8 \
  --seed 1234 \
  --cond_embed_noise_std 0.40 \
  --top_k 0 \
  --top_p 1.0 \
  --min_p 0.0