A basic implementation of [How Alignment Shrinks the Generative Horizon](https://arxiv.org/abs/2506.17871) (Yang & Holtzman, preprint 2025)

Calculates branching factor using vLLM.

### Quick Start

```sh
pip install -r requirements.txt
```

```sh
# demo: computes branching factor over Minerva MATH for Llama 3 8B (< 4 min on 1 A100)
python branching.py

# use custom args like this
python branching.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --num_samples 5 \
    --max_tokens 256 \
    --temperature 0.8 \
    --top_p 0.95 \
    --output_dir "./my_results"
```