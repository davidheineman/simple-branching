from dataclasses import dataclass
import gc
import os, json
import atexit
from pathlib import Path
import numpy as np
from typing import List, Dict
import torch
from transformers import AutoTokenizer
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams, TokensPrompt
from vllm.sequence import Logprob
from utils import apply_overrides, quiet_vllm_logger
from rich.pretty import pprint
from dataset.simple_metric import Instance
from dataset.simple_data import MinervaMath
from dataset.math_extract import extract_answer, is_equiv
from pretty.ridgelines import to_discrete_distributions, ridgeline_plot

RESULTS_DIR = Path(
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
)


@dataclass
class RunConfig:
    model: str = "Qwen/Qwen3-0.6B"

    # forking paths config
    num_samples: int = 16  # Resamples per token

    # vllm config
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    min_p: float = 0.1
    logprobs: int = 20
    gpu_memory_utilization: float = 0.7
    output_dir: str = RESULTS_DIR / "results"
    quiet_vllm: bool = True


class Pipeline:
    def __init__(self, config: RunConfig):
        self.config: RunConfig = config

        atexit.register(self.cleanup)  # add cleanup

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.config.model)

        # run on all available gpus
        num_gpus = torch.cuda.device_count()

        # set cuda architecutre (avoids warning)
        capability = torch.cuda.get_device_capability()
        arch_str = f"{capability[0]}.{capability[1]}"
        os.environ["TORCH_CUDA_ARCH_List"] = arch_str

        self.llm: LLM = LLM(
            model=self.config.model,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
            enable_chunked_prefill=True,
        )

    def cleanup(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def generate(self, prompts, n) -> List[RequestOutput]:
        sampling_params = SamplingParams(
            n=n,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            min_p=self.config.min_p,
            max_tokens=self.config.max_tokens,
            logprobs=self.config.logprobs,
            prompt_logprobs=self.config.logprobs,
        )

        outputs: List[RequestOutput] = self.llm.generate(prompts, sampling_params)

        return outputs

    def sample_completions(
        self, prompts: List[str], num_samples: int
    ) -> List[List[str]]:
        outputs: List[RequestOutput] = self.generate(prompts, n=num_samples)

        # Get all completions
        all_completions = []
        for output in outputs:
            completions: List[CompletionOutput] = output.outputs
            completion_texts = [completion.text for completion in completions]
            all_completions += [completion_texts]

        return all_completions

    def extract_answers(self, completions: List[str]) -> List[str]:
        answers = []
        for completion in completions:
            gen_answers: List[str] = extract_answer(completion)
            answers += [
                gen_answers[0]
            ]  # keep most plausible (first) answer from each generation
        return answers


class ForkingPathsPipeline(Pipeline):
    def sample_single_completion(self, prompt: str) -> List[List[str]]:
        output = self.generate([prompt], n=1)

        # Get the only completion
        assert len(output) == 1
        output: RequestOutput = output[0]
        completion = output.outputs
        assert len(completion) == 1
        completion: CompletionOutput = completion[0]

        token_ids: List[str] = completion.token_ids
        logprobs: List[dict[int, Logprob]] = completion.logprobs

        # Convert Logprob to float
        logprobs: List[dict[int, float]] = [
            {id: lp.logprob for id, lp in lps.items()} for lps in logprobs
        ]

        return {"tokens": token_ids, "logprobs": logprobs}

    def run_forking_analysis(self, prompt: str):
        base_completion = self.sample_single_completion(prompt)
        base_toks = base_completion["tokens"]
        base_logprobs = base_completion["logprobs"]

        # Get all prefix in the completion
        all_prefixes = []
        for t in range(1, len(base_toks)):
            prefix = base_toks[:t]
            all_prefixes.append(TokensPrompt(prompt_token_ids=prefix))

        # Sample completions for each prefix
        all_outputs: List[List[str]] = self.sample_completions(
            prompts=all_prefixes, num_samples=self.config.num_samples
        )

        # Get outcomes
        all_prefix_answers: List[List[str]] = []
        for prefix_output in all_outputs:
            prefix_answers: List[str] = self.extract_answers(prefix_output)
            all_prefix_answers += [prefix_answers]

        # Print distribution of outcomes over time
        distributions, x_order = to_discrete_distributions(all_prefix_answers, top_k=20)
        ridgeline_plot(distributions, x_order)

        return all_prefix_answers


class SimpleDistributionPipeline(Pipeline):
    def run_simple_dist_analysis(self, prompts: List[str]) -> List[str]:
        completions = self.sample_completions(prompts, self.config.num_samples)
        all_answers: List[List[str]] = []
        for samples in completions:
            answers: List[str] = self.extract_answers(samples)
            all_answers += [answers]
        return all_answers


def create_math_prompts() -> List[str]:
    dataset = MinervaMath("algebra")
    instances: List[Instance] = dataset.requests
    requests: List[str] = [instance.request for instance in instances]
    return requests


def main():
    default_config = RunConfig()
    config: RunConfig = apply_overrides(default_config)
    pprint(config, expand_all=True)

    if config.quiet_vllm:
        quiet_vllm_logger()

    prompts = create_math_prompts()

    # pipeline = ForkingPathsPipeline(config)
    # results = pipeline.run_forking_analysis(prompt=prompts[0])
    # pprint(results)

    #####

    # Get all checkpoints
    base_path = config.model
    ckpt_paths = [os.path.join(base_path, d) for d in os.listdir(base_path)]

    print("Running these checkpoints:")
    pprint(ckpt_paths)

    # Run each checkpoint
    results = {}
    for ckpt_path in ckpt_paths:
        checkpoint_config = RunConfig(**vars(config))
        checkpoint_config.model = ckpt_path
        pprint(checkpoint_config, expand_all=True)

        try:
            # Get distribution
            pipeline = SimpleDistributionPipeline(checkpoint_config)
            results[ckpt_path] = pipeline.run_simple_dist_analysis(prompts=prompts)
        except Exception as e:
            print(f'Failure on ckpt {ckpt_path}: {e}')

        # Clear VLLM resources and memory
        del pipeline.llm
        torch.cuda.empty_cache()
        gc.collect()

    pprint(results)

    # Save results
    base_dir_name = os.path.basename(base_path)
    output_path = config.output_dir / f"{base_dir_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
