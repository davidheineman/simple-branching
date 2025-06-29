A basic implementation of [How Alignment Shrinks the Generative Horizon](https://arxiv.org/abs/2506.17871) (Yang & Holtzman, preprint 2025)

Calculates branching factor using vLLM.

### Quick Start

```sh
pip install -r requirements.txt
```

Let's try it with the default config:

```sh
# demo: computes branching factor over Minerva MATH Algebra for Llama 3 8B
# (< 4 min on 1 A100)
python src/branching.py --model=meta-llama/Meta-Llama-3-8B
python src/branching.py --model=meta-llama/Meta-Llama-3-8B-Instruct
```

This should output:

```sh
# Result for Llama 3 base
Results summary (meta-llama/Meta-Llama-3-8B)
============================================================
Branching Factor (mean): 3.5983
Branching Factor (std):  5.1210
Branching Factor (median): 1.3975
Number of sequences: 3561
Prompts: 1187
Samples per prompt: 3

# Result for Llama 3 instruct
Results summary (meta-llama/Meta-Llama-3-8B-Instruct)
============================================================
Branching Factor (mean): 2.0251
Branching Factor (std):  2.9968
Branching Factor (median): 1.0016
Number of sequences: 3561
Prompts: 1187
Samples per prompt: 3
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