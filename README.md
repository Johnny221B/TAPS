# Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models

[![Project Website](https://img.shields.io/badge/Website-TAPS-blue?style=flat-square)](https://taps-dlm.github.io) [![Paper](https://img.shields.io/badge/Paper-ArXiv-red?style=flat-square)](https://arxiv.org/abs/2601.22629) [![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)

**T**ime-**A**nnealed **P**erturbation **S**ampling (TAPS) is an inference-time method for improving **diversity** in diffusion language models without sacrificing generation quality.

This repository contains the official implementation of TAPS and the code used to reproduce experiments reported in the paper.

---

## Method Overview

<img src="docs/method.png" width="800"/>

*A conceptual comparison of the inference process between the base Diffusion-LM and our proposed method, TAPS, illustrating different context conditioning behaviors.*

---

## Supported Backbones

This repository supports two diffusion language model backbones:

| Backbone                    | Hugging Face                                                                   | Loader                                |
| --------------------------- | ------------------------------------------------------------------------------ | ------------------------------------- |
| **LLaDA-8B-Instruct** | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)     | `transformers.AutoModel`            |
| **TraDo-8B-Instruct** | [Gen-Verse/TraDo-8B-Instruct](https://huggingface.co/Gen-Verse/TraDo-8B-Instruct) | `transformers.AutoModelForCausalLM` |

---

## Benchmarks

- **GSM8K**
- **WritingPrompts**
- **NoveltyBench**
- **Arena-Hard-Auto**

---

## Reproduce: WritingPrompts

### LLaDA

```bash
cd /mnt/data/wujx/DLM/TAPS
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 -m benchmarks.writingprompts.eval_llada_wp \
  --model_path /path/to/llada \
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
```

### TraDo

```bash
cd /mnt/data/wujx/DLM/TAPS
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.writingprompts.eval_trado_wp \
  --run_name trado_embedding_run \
  --mode embedding \
  --model_path /path/to/trado \
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
```

---

## Citation

```bibtex
@misc{wu2026timeannealedperturbationsamplingdiverse,
      title={Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models}, 
      author={Jingxuan Wu and Zhenglin Wan and Xingrui Yu and Yuzhe Yang and Yiqiao Huang and Ivor Tsang and Yang You},
      year={2026},
      eprint={2601.22629},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.22629}, 
}

```

## License

This project is released under the **MIT License**. See the [LICENSE](./LICENSE) file for the full text.

```
SPDX-License-Identifier: MIT
```
