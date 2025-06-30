from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import List, Dict
from bayesian_changepoint_detection.offline_changepoint_detection import offline_changepoint_detection
import torch
from transformers import AutoTokenizer
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams, TokensPrompt
from vllm.sequence import Logprob
from utils import apply_overrides, quiet_vllm_logger
from rich import print as rprint
from rich.pretty import pprint
from dataset.simple_metric import Instance
from dataset.simple_data import MinervaMath
from dataset.math_extract import extract_answer, is_equiv


L2_NORM = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))

RESULTS_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))


@dataclass
class RunConfig:
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # forking paths config
    num_outcome_samples: int = 128 # num generations to estimate the output set
    num_samples: int = 16 # Resamples per alternative token
    epsilon = 0.3         # Threshold for outcome deviation

    # vllm config
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    min_p: float = 0.1
    logprobs: int = 20
    gpu_memory_utilization: float = 0.8
    output_dir: str = RESULTS_DIR / "results"
    quiet_vllm: bool = True


class ForkingPathsPipeline:
    def __init__(self, config: RunConfig):
        self.config: RunConfig = config

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.config.model)
        
        # run on all available gpus
        num_gpus = torch.cuda.device_count()

        # set cuda architecutre (avoids warning)
        capability = torch.cuda.get_device_capability()
        arch_str = f"{capability[0]}.{capability[1]}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_str

        self.llm: LLM = LLM(
            model=self.config.model,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
            enable_chunked_prefill=True,
        )
        

    def generate(self, prompts, n):
        sampling_params = SamplingParams(
            n=n,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            min_p=self.config.min_p,
            max_tokens=self.config.max_tokens,
            logprobs=self.config.logprobs,
            prompt_logprobs=self.config.logprobs,
        )
        
        outputs: list[RequestOutput] = self.llm.generate(
            prompts, 
            sampling_params
        )

        return outputs


    def get_base_path_with_logprobs(self, prompt: str) -> Dict:
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
        logprobs: List[dict[int, float]] = [{id: lp.logprob for id, lp in lps.items()} for lps in logprobs]

        return {
            'tokens': token_ids,
            'logprobs': logprobs
        }


    def sample_completions(self, prefix: List[str], num_samples: int) -> List[str]:
        output = self.generate([prefix], n=num_samples)

        # Get all completions
        assert len(output) == 1
        output: RequestOutput = output[0]
        completions: List[CompletionOutput] = output.outputs

        completion_texts = [completion.text for completion in completions]

        return completion_texts
    

    def get_outcomes(self, prompt: str) -> str:
        """run generation once to estimate the set of possible outcomes."""
        
        generations = self.sample_completions(
            prefix = prompt,
            num_samples = self.config.num_outcome_samples
        )

        answers = []
        for generation in generations:
            gen_answers: List[str] = extract_answer(generation)

            # keep most plausible (first) answer from each generation
            answers += [gen_answers[0]] 

        return answers


    def outcome_vector(self, ans: str, choices: List[str]) -> np.ndarray:
        """ return a 1-hot corresponding to the matching answer """
        vec = np.zeros(len(choices) + 1)

        gen_answers: List[str] = extract_answer(ans)

        idx = -1 # if nothing matches, use last entry
        for i, choice in enumerate(choices):
            for gen in gen_answers:
                if is_equiv(gen, choice):
                    idx = i
                    break

        vec[idx] = 1
        return vec


    def run_forking_analysis(self, prompt: str):
        outcome_choices = self.get_outcomes(prompt)
        
        base = self.get_base_path_with_logprobs(prompt)
        x_star = base['tokens']
        logprobs = base['logprobs']

        o_t = []
        o_t_w = defaultdict(dict)
        token_probs = []

        # TODO: Collapse the double for loop into 1 vLLM call

        for t in range(len(x_star)):
            token_prob_dict = logprobs[t]
            token_probs.append(token_prob_dict)

            for w, p_w in token_prob_dict.items():
                prefix = x_star[:t] + [w]
                prompt = TokensPrompt(prompt_token_ids=prefix)
                samples = self.sample_completions(prompt, self.config.num_samples)
                outcomes = [self.outcome_vector(sample, outcome_choices) for sample in samples]
                avg_outcome = np.mean(outcomes, axis=0)
                o_t_w[t][w] = avg_outcome

            base_token = x_star[t]
            dist_w = [token_prob_dict[w] * o_t_w[t][w] for w in token_prob_dict]
            o_t.append(np.sum(dist_w, axis=0))

        # Semantic drift vector y_t
        o_0 = o_t[0]
        y_t = [L2_NORM(o, o_0) for o in o_t]

        # CPD
        def prior(length):
            return 1 / length

        _, _, P = offline_changepoint_detection(np.array(y_t), prior, truncate=-40)
        cp_prob = 1 - P.sum(0)

        # Survival Analysis
        survival = []
        S_t = 1.0

        for t in range(len(x_star)):
            hazard = 0
            base_token = x_star[t]
            base_vec = o_t_w[t].get(base_token)

            if base_vec is None:
                survival.append(S_t)
                continue

            for w, vec in o_t_w[t].items():
                if L2_NORM(vec, base_vec) > self.config.epsilon:
                    hazard += token_probs[t].get(w, 0)
            S_t *= (1 - hazard)
            survival.append(S_t)

        return {
            'tokens': x_star,
            'o_t': o_t,
            'cp_prob': cp_prob,
            'survival': survival
        }


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

    pipeline = ForkingPathsPipeline(config)

    prompts = create_math_prompts()

    results = pipeline.run_forking_analysis(prompt=prompts[0])

    pprint(results)


if __name__ == "__main__":
    main()