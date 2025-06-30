A basic implementation of [How Alignment Shrinks the Generative Horizon](https://arxiv.org/abs/2506.17871) (Yang & Holtzman, preprint 2025)

Calculates branching factor using a vLLM backend. 

**What is BF?** Branching factor is the `exp` of the negated average per-token NLL of a set of generated sequences. Think of the generation space as traversing a tree. The branching factor (BF) would be the average children considered at each step of the tree. For instance, a BF of 10 indicates the model realistically considers 10 high-probability next tokens at each step. Although BF averages NLL across the full sequence, in practice some tokens have higher BF than others (as explored in [Bigelow et al., 2024](https://arxiv.org/abs/2412.07961)).

### Quick Start

```sh
pip install -r requirements.txt

# (optional) if using B200s, install torch wheels (requires nvcc 12.3+, cuda 12.9+)
uv pip install torch torchvision numpy==2.2 vllm==0.9.1 --torch-backend=cu128 --force-reinstall
```

Let's try it with the default config:

```sh
# quickly check everything is working
python src/branching.py --model=Qwen/Qwen3-0.6B --num_samples=2 --max_tokens=128
```

### Demo: Llama 3 8B

We can now try compute BF on a pair of base/instruction tuned models:

```sh
# demo: computes branching factor over Minerva MATH Algebra for Llama 3 8B
# (< 20 min on 1 A100)
python src/branching.py --model=meta-llama/Meta-Llama-3-8B
python src/branching.py --model=meta-llama/Meta-Llama-3-8B-Instruct
```

This should output:

```sh
# Result for Llama 3 base
Results summary (meta-llama/Meta-Llama-3-8B)
============================================================
Branching Factor (mean): 3.6661
Branching Factor (median): 1.7372
Branching Factor (std):  8.6085
Number of sequences: 1187 prompts * 16 samples = 18992

# Result for Llama 3 instruct
Results summary (meta-llama/Meta-Llama-3-8B-Instruct)
============================================================
Branching Factor (mean): 2.5833
Branching Factor (median): 1.0001
Branching Factor (std):  3.9720
Number of sequences: 1187 prompts * 16 samples = 18992
```

Notice how the instruction-tuned Llama has a lower branching factor. This agrees with the core finding of Yang & Holtzman, 2025. Nice!

### Custom Args

```sh
# pass custom args like this
python branching.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --num_samples 5 \
    --max_tokens 256 \
    --temperature 0.8 \
    --top_p 0.95 \
    --output_dir "./my_results"
```

### Forking Paths

Also working on a reproduction of [Forking Paths in Neural Text Generation](https://arxiv.org/abs/2412.07961) (Bigelow et al., 2024).

```sh
python src/forking_paths.py --model=Qwen/Qwen3-0.6B
```