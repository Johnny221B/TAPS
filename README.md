# TAPS

**T**ime-**A**nnealed **P**erturbation **S**ampling (TAPS) is an inference-time method
for improving **set-level diversity** in diffusion language models without
sacrificing generation quality.

This repository contains the official implementation of TAPS and the code
used to reproduce experiments in the paper.

## Supported Backbones

- **LLaDA**
- **TraDo**

## Benchmarks

- GSM8K
- WritingPrompts
- NoveltyBench (via external repository with patches)
- Arena-Hard-Auto (via external repository with patches)

## Repository Structure


src/            # Core method implementations

benchmarks/     # Benchmark-specific runners and wrappers

configs/        # Experiment configurations

scripts/        # Launch and orchestration scripts

docs/           # Setup and reproduction instructions

## License

This project is released under the MIT License.
