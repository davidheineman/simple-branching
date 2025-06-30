from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
import numpy as np
from simple_data import MinervaMath, Instance
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import json
from datetime import datetime
from omegaconf import OmegaConf
from rich import print as rprint
from rich.pretty import pprint

from vllm import LLM, RequestOutput, SamplingParams
from transformers import AutoTokenizer
from uncertainty_computation import compute_bf_values
from loglik_computation import get_tokenwise_entropy_from_vllm_outputs


RESULTS_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))


@dataclass
class RunConfig:
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_samples: int = 16
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    min_p: float = 0.1
    logprobs: int = 20
    gpu_memory_utilization: float = 0.8
    output_dir: str = RESULTS_DIR / "results"
    quiet_vllm: bool = True


class BranchingFactorPipeline:
    def __init__(self, config: RunConfig):
        self.config: RunConfig = config
        self.model_name = config.model
        self.gpu_memory_utilization = config.gpu_memory_utilization
        self.tokenizer = None
        self.llm = None

    def setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # run on all available gpus
        num_gpus = torch.cuda.device_count()

        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
            enable_chunked_prefill=True,
        )

    def generate_responses(
        self,
        prompts: List[str],
    ) -> list[RequestOutput]:
        sampling_params = SamplingParams(
            n=self.config.num_samples,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            min_p=self.config.min_p,
            max_tokens=self.config.max_tokens,
            logprobs=self.config.logprobs,
            prompt_logprobs=self.config.logprobs,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs

    def compute_entropy_profiles(self, outputs: list[RequestOutput], top_p: float) -> Dict[str, List]:
        """ Convert vLLM outputs to token-level entropy scores """
        
        generation_info = []
        entropy_profiles = []
        output_per_token_logprob_truncated = []
        for output in tqdm(outputs, desc="Calculating token-level entropy"):
            prompt = output.prompt
            all_outputs = output.outputs
            all_output_texts = [x.text.strip() for x in all_outputs]

            # Step 1
            entropy_profile = get_tokenwise_entropy_from_vllm_outputs(
                all_outputs, p=top_p, top_p_mode=True
            )

            entropies = [tok_entropy for tok_entropy, tok_id in entropy_profile]
            token_ids = [tok_id for tok_entropy, tok_id in entropy_profile]
            token_texts = [self.tokenizer.convert_ids_to_tokens(x) for x in token_ids]

            # Step 2
            per_output_logprobs = []
            per_output_logprobs_truncated = []
            per_output_entropies = []            
            for single_output in all_outputs:
                if not single_output.logprobs:
                    raise AssertionError("logprobs must exist!")
                
                logprobs = []
                logprobs_truncated = []
                entropy_vals = []

                for token_logprob in single_output.logprobs:
                    if token_logprob:
                        top_token_logprob = list(token_logprob.values())[0].logprob
                        logprobs.append(top_token_logprob)

                        if top_token_logprob != 0.0:
                            logprobs_truncated.append(top_token_logprob)

                        token_probs = [
                            (token_id, logprob_info.logprob)
                            for token_id, logprob_info in token_logprob.items()
                        ]
                        entropy = -sum(
                            np.exp(logprob) * logprob for _, logprob in token_probs
                        )
                        entropy_vals.append(entropy)

                per_output_logprobs.append(logprobs)
                per_output_logprobs_truncated.append(logprobs_truncated)
                per_output_entropies.append(entropy_vals)

            entropy_profiles.append(entropies)
            output_per_token_logprob_truncated.append(per_output_entropies)
            generation_info.append({
                "prompt": prompt,
                "output_text": all_output_texts,
                "token_text": token_texts,
                "logprobs": per_output_logprobs,
                "logprobs_truncated": per_output_logprobs_truncated
            })
        
        return {
            "entropy_profiles": entropy_profiles, 
            "output_per_token_logprob_truncated": output_per_token_logprob_truncated,
            "generation_info": generation_info
        }
    

    def compute_branching_factors(
        self,
        entropy_profiles,
        output_per_token_logprob_truncated,
        asymptotic_limit: int = 50,
    ) -> Dict[str, Any]:
        # Process entropy values (cumulative)
        output_token_entropies = []
        for entropies_per_output in entropy_profiles:
            for entropy_list in entropies_per_output:
                if len(entropy_list) > 0:
                    cumulative_entropies = np.cumsum(entropy_list)
                    output_token_entropies.append(cumulative_entropies)

        # Process loglik values
        output_token_logprobs = []
        for instance in output_per_token_logprob_truncated:
            for output_logprobs in instance:
                if len(output_logprobs) > 0:
                    logprob_values = [
                        float(x) for x in output_logprobs if float(x) != 0.0
                    ]
                    if len(logprob_values) > 0:
                        output_token_logprobs.append(np.array(logprob_values))

        print(f"   Total entropy sequences: {len(output_token_entropies)}")
        print(f"   Total loglik sequences: {len(output_token_logprobs)}")

        if len(output_token_entropies) == 0 or len(output_token_logprobs) == 0:
            raise RuntimeError("No valid sequences found for BF computation")
        
        min_sequences = min(len(output_token_entropies), len(output_token_logprobs))
        entropy_seqs = output_token_entropies[:min_sequences]
        loglik_seqs = output_token_logprobs[:min_sequences]

        print(f"   Using {min_sequences} matched sequences")

        # Compute branching factors using the core algorithm
        bf_values = compute_bf_values(
            entropy_seqs, loglik_seqs, asymptotic_limit=asymptotic_limit
        )

        exp_bf_values = np.exp(bf_values)
        mean_bf = np.mean(exp_bf_values)
        std_bf = np.std(exp_bf_values)
        median_bf = np.median(exp_bf_values)

        entropy_lengths = [len(seq) for seq in entropy_seqs]
        loglik_lengths = [len(seq) for seq in loglik_seqs]

        results = {
            "branching_factor_mean": float(mean_bf),
            "branching_factor_std": float(std_bf),
            "branching_factor_median": float(median_bf),
            "branching_factor_values": exp_bf_values.tolist(),
            "log_branching_factor_values": bf_values,
            "num_sequences": min_sequences,
            "entropy_seq_stats": {
                "min_length": min(entropy_lengths),
                "max_length": max(entropy_lengths),
                "mean_length": float(np.mean(entropy_lengths)),
            },
            "loglik_seq_stats": {
                "min_length": min(loglik_lengths),
                "max_length": max(loglik_lengths),
                "mean_length": float(np.mean(loglik_lengths)),
            },
            "asymptotic_limit": asymptotic_limit,
        }

        return results


    def run_generation_and_bf(self, prompts: List[str]) -> Dict[str, Any]:
        if self.llm is None:
            self.setup_model()

        # Generate with vLLM
        outputs: list[RequestOutput] = self.generate_responses(
            prompts=prompts,
        )

        # Extract per-token entropy using logprobs
        entropy_data = self.compute_entropy_profiles(
            outputs, 
            top_p=self.config.top_p
        )

        # Compute branching factor
        bf_results = self.compute_branching_factors(
            entropy_profiles=entropy_data["entropy_profiles"], 
            output_per_token_logprob_truncated=entropy_data["output_per_token_logprob_truncated"]
        )

        complete_results = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "config": OmegaConf.to_container(OmegaConf.structured(self.config), resolve=True),
            "branching_factor_results": bf_results,
            "raw_data": {
                "prompts": prompts,
                "entropy_data": entropy_data,
            },
        }

        if self.config.output_dir:
            self.save_results(
                results=complete_results, 
                output_dir=self.config.output_dir
            )

        return complete_results
    
    def save_results(self, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = self.model_name.split("/")[-1]

        results_file = os.path.join(
            output_dir, f"bf_results_{model_short}_{timestamp}.json"
        )
        with open(results_file, "w") as f:
            json_results = results.copy()
            json_results["raw_data"] = {
                "note": "Raw data excluded from JSON for size"
            }
            json.dump(json_results, f, indent=2)

        raw_data_file = os.path.join(
            output_dir, f"bf_raw_data_{model_short}_{timestamp}.pt"
        )
        torch.save(results["raw_data"], raw_data_file)

        print(f"Results saved to: {results_file}")
        print(f"Raw data saved to: {raw_data_file}")


def create_math_prompts() -> List[str]:
    dataset = MinervaMath("algebra")
    instances: List[Instance] = dataset.requests
    requests: List[str] = [instance.request for instance in instances]
    return requests


def apply_overrides(config):
    base = OmegaConf.structured(config)
    
    # Get CLI args up to '--' if present, otherwise all args
    args = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
    cli_args = [arg.lstrip("-") for arg in args]
    
    # Merge overrides
    overrides = OmegaConf.from_cli(cli_args)
    merged = OmegaConf.merge(base, overrides)
    return OmegaConf.to_object(merged)


def quiet_vllm_logger(level=logging.WARNING):
    for name, logger in logging.root.manager.loggerDict.items():
        if name.startswith("vllm"):
            logging.getLogger(name).setLevel(level)


def main():
    # initalize config
    default_config = RunConfig()
    config: RunConfig = apply_overrides(default_config)
    pprint(config, expand_all=True)

    if config.quiet_vllm:
        quiet_vllm_logger()

    pipeline = BranchingFactorPipeline(config)

    # load prompts
    prompts = create_math_prompts()
    rprint(f"[green]Loaded {len(prompts)} MATH prompts by default[/green]")

    # Generate and calculate BF!
    results = pipeline.run_generation_and_bf(prompts=prompts)

    # Print results
    rprint(f"\n[bold blue]Results summary ({results['model_name']})[/bold blue]")
    rprint("=" * 60)
    bf_results = results["branching_factor_results"]
    rprint(f"[green]Branching Factor (mean): {bf_results['branching_factor_mean']:.4f}[/green]")
    rprint(f"[blue]Branching Factor (median): {bf_results['branching_factor_median']:.4f}[/blue]")
    rprint(f"[yellow]Branching Factor (std):  {bf_results['branching_factor_std']:.4f}[/yellow]")
    rprint(f"[cyan]Number of sequences: {len(prompts)} prompts * {config.num_samples} rollouts = {bf_results['num_sequences']}[/cyan]")


if __name__ == "__main__":
    main()
