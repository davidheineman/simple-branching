from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
from simple_data import MinervaMath, Instance
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import json
from datetime import datetime
from omegaconf import OmegaConf
from rich import print as rprint
from rich.panel import Panel

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from uncertainty_computation import compute_bf_values
from loglik_computation import get_tokenwise_entropy_from_vllm_outputs


RESULTS_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


@dataclass
class RunConfig:
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_samples: int = 3
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    min_p: float = 0.1
    logprobs: int = 20
    gpu_memory_utilization: float = 0.3
    output_dir: str = RESULTS_DIR / "results"
    custom_prompts: Optional[str] = None


class BranchingFactorPipeline:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.3):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tokenizer = None
        self.llm = None

    def setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_batched_tokens=512,
            enable_chunked_prefill=True,
        )

    def generate_responses(
        self,
        prompts: List[str],
        num_samples: int = 3,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        min_p: float = 0.1,
        logprobs: int = 20,
    ) -> List[Any]:
        sampling_params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            max_tokens=max_tokens,
            logprobs=logprobs,
            prompt_logprobs=logprobs,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs

    def compute_entropy_profiles(
        self, outputs: List[Any], top_p: float = 0.9
    ) -> Dict[str, List]:
        entropy_profiles = []

        for output in tqdm(outputs, desc="Processing outputs"):
            prompt = output.prompt
            all_outputs = output.outputs
            all_output_texts = [x.text.strip() for x in all_outputs]

            entropy_profile = get_tokenwise_entropy_from_vllm_outputs(
                all_outputs, p=top_p, top_p_mode=True
            )

            entropies = [x[0] for x in entropy_profile]
            token_ids = [x[1] for x in entropy_profile]
            token_texts = [self.tokenizer.convert_ids_to_tokens(x) for x in token_ids]

            entropy_profiles.append([prompt, all_output_texts, token_texts, entropies])
        return {"entropy_profiles": entropy_profiles, "top_p": top_p}

    def compute_loglik_analysis(self, outputs: List[Any]) -> Dict[str, List]:
        loglik_data = {
            "prompt": [],
            "output": [],
            "output_per_token_logprob": [],
            "output_per_token_logprob_truncated": [],
            "entropy": [],
        }

        for output in tqdm(outputs, desc="Processing loglik data"):
            prompt = output.prompt
            all_outputs = output.outputs

            output_texts = []
            output_logprobs = []
            output_logprobs_truncated = []
            entropies = []

            for single_output in all_outputs:
                output_texts.append(single_output.text)

                if single_output.logprobs:
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

                    output_logprobs.append(logprobs)
                    output_logprobs_truncated.append(logprobs_truncated)
                    entropies.append(entropy_vals)
                else:
                    output_logprobs.append([])
                    output_logprobs_truncated.append([])
                    entropies.append([])

            loglik_data["prompt"].append(prompt)
            loglik_data["output"].append(output_texts)
            loglik_data["output_per_token_logprob"].append(output_logprobs)
            loglik_data["output_per_token_logprob_truncated"].append(
                output_logprobs_truncated
            )
            loglik_data["entropy"].append(entropies)

        return loglik_data

    def compute_branching_factors(
        self,
        entropy_data: Dict[str, Any],
        loglik_data: Dict[str, Any],
        asymptotic_limit: int = 50,
    ) -> Dict[str, Any]:
        entropy_profiles = entropy_data["entropy_profiles"]

        # Process entropy values (cumulative)
        output_token_entropies = []
        for instance in entropy_profiles:
            entropies_per_output = instance[3]
            for entropy_list in entropies_per_output:
                if len(entropy_list) > 0:
                    cumulative_entropies = np.cumsum(entropy_list)
                    output_token_entropies.append(cumulative_entropies)

        # Process loglik values
        output_token_logprobs = []
        for instance in loglik_data["output_per_token_logprob_truncated"]:
            for output_logprobs in instance:
                if len(output_logprobs) > 0:
                    logprob_values = [
                        float(x) for x in output_logprobs if float(x) != 0.0
                    ]
                    if len(logprob_values) > 0:
                        output_token_logprobs.append(np.array(logprob_values))

        print(f"   Total entropy sequences: {len(output_token_entropies)}")
        print(f"   Total loglik sequences: {len(output_token_logprobs)}")

        if len(output_token_entropies) > 0 and len(output_token_logprobs) > 0:
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
        else:
            print("   No valid sequences found for BF computation")
            return {"error": "No valid sequences found"}

    def run_pipeline(
        self,
        prompts: List[str],
        num_samples: int = 3,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        min_p: float = 0.1,
        logprobs: int = 20,
        output_dir: str = None,
    ) -> Dict[str, Any]:
        if self.llm is None:
            self.setup_model()

        outputs = self.generate_responses(
            prompts=prompts,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            logprobs=logprobs,
        )

        entropy_data = self.compute_entropy_profiles(outputs, top_p=top_p)
        loglik_data  = self.compute_loglik_analysis(outputs)
        bf_results   = self.compute_branching_factors(entropy_data, loglik_data)

        complete_results = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "num_prompts": len(prompts),
                "num_samples": num_samples,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "min_p": min_p,
                "logprobs": logprobs,
            },
            "branching_factor_results": bf_results,
            "raw_data": {
                "prompts": prompts,
                "entropy_data": entropy_data,
                "loglik_data": loglik_data,
            },
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = self.model_name.split("/")[-1]

            results_file = os.path.join(
                output_dir, f"bf_results_{model_short}_{timestamp}.json"
            )
            with open(results_file, "w") as f:
                json_results = complete_results.copy()
                json_results["raw_data"] = {
                    "note": "Raw data excluded from JSON for size"
                }
                json.dump(json_results, f, indent=2)

            raw_data_file = os.path.join(
                output_dir, f"bf_raw_data_{model_short}_{timestamp}.pt"
            )
            torch.save(complete_results["raw_data"], raw_data_file)

            print(f"Results saved to: {results_file}")
            print(f"Raw data saved to: {raw_data_file}")

        return complete_results


def create_math_prompts() -> List[str]:
    dataset = MinervaMath("algebra")
    instances: List[Instance] = dataset.requests
    requests: List[str] = [instance.request for instance in instances]
    return requests


def main():
    """Main function to run the pipeline"""

    # initalize config
    config: RunConfig = OmegaConf.structured(RunConfig)
    # cli_config = OmegaConf.from_cli()
    # config = OmegaConf.merge(config, cli_config)

    rprint(config)

    # Load prompts
    if config.custom_prompts and os.path.exists(config.custom_prompts):
        with open(config.custom_prompts, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        rprint(f"[green]Loaded {len(prompts)} custom prompts from {config.custom_prompts}[/green]")
    else:
        prompts = create_math_prompts()
        rprint(f"[green]Using {len(prompts)} MATH prompts by default[/green]")

    pipeline = BranchingFactorPipeline(
        model_name=config.model,
        gpu_memory_utilization=config.gpu_memory_utilization
    )

    results = pipeline.run_pipeline(
        prompts=prompts,
        num_samples=config.num_samples,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        min_p=config.min_p,
        logprobs=config.logprobs,
        output_dir=config.output_dir,
    )

    rprint("\n[bold blue]Results summary[/bold blue]")
    rprint("=" * 60)

    if "error" not in results["branching_factor_results"]:
        bf_results = results["branching_factor_results"]
        rprint(f"[green]Branching Factor (mean): {bf_results['branching_factor_mean']:.4f}[/green]")
        rprint(f"[yellow]Branching Factor (std):  {bf_results['branching_factor_std']:.4f}[/yellow]")
        rprint(f"[blue]Branching Factor (median): {bf_results['branching_factor_median']:.4f}[/blue]")
        rprint(f"[cyan]Number of sequences: {bf_results['num_sequences']}[/cyan]")
        rprint(f"[magenta]Model: {results['model_name']}[/magenta]")
        rprint(f"[cyan]Prompts: {results['parameters']['num_prompts']}[/cyan]")
        rprint(f"[cyan]Samples per prompt: {results['parameters']['num_samples']}[/cyan]")
    else:
        rprint("[red]Computation failed:", results["branching_factor_results"]["error"], "[/red]")


if __name__ == "__main__":
    main()
