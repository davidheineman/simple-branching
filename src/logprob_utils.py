from typing import List
import torch
import numpy as np

from vllm import CompletionOutput
from vllm.sequence import Logprob

TOLERANCE_INF = 1e-12


def get_truncated_dist_top_p(
    sorted_dist: dict[int, Logprob], top_p: float, tolerance_inf: float = TOLERANCE_INF
):
    # 2025-01-10: this is a new implementation take numerical instability into consideration
    # esp. for entropy contribution, when a logprob is sufficiently small, we can ignore it
    _cumulative_probs = 0.0
    truncated_dist = dict()
    for key, value in sorted_dist:
        _exp_logprob = np.exp(value)
        if np.isinf(value) or _exp_logprob < tolerance_inf:
            # too small to contribute meaningfully, so we can save some consideration for numerical stability
            continue
        _cumulative_probs += _exp_logprob
        truncated_dist[key] = value
        if _cumulative_probs >= top_p:
            break
    return truncated_dist


def _get_token_level_entropy_truncated(
    logprobs: List[dict[int, Logprob]], 
    token_ids: List[int],
    p: float, 
    max_length=None, 
    top_p_mode=False
):
    gen_seq_len = len(logprobs)
    per_output_entropies = []
    _max_length = (
        min(gen_seq_len, max_length) if max_length is not None else gen_seq_len
    )
    for length_i in range(_max_length):
        unnormalized_dist = logprobs[length_i]
        if unnormalized_dist is None:
            break
        
        _, normalized_dist_values = get_token_truncated_dist(
            unnormalized_dist=unnormalized_dist, 
            token_ids=token_ids, 
            length_i=length_i, 
            p=p, 
            top_p_mode=top_p_mode
        )
        entropy = -torch.sum(
            normalized_dist_values * torch.log(normalized_dist_values)
        )
        per_output_entropies.append(entropy.item())
    return per_output_entropies


def get_token_truncated_dist(
    unnormalized_dist: dict[int, Logprob],
    token_ids: List[int],
    length_i: int,
    p: float,
    top_p_mode: bool = False,
):
    if not top_p_mode:
        # min_p mode
        truncated_dist = dict()
        for key in unnormalized_dist:
            _value = unnormalized_dist[key].logprob
            if np.exp(_value) >= p or key == token_ids[length_i]:
                # "or" part: allow minor error
                # delay all normalization in the end to avoid numerical instability
                truncated_dist[key] = _value
    else:
        _unnormalized_dist = {
            key: unnormalized_dist[key].logprob for key in unnormalized_dist
        }
        _sorted_dist = sorted(
            _unnormalized_dist.items(), key=lambda x: x[1], reverse=True
        )
        truncated_dist = get_truncated_dist_top_p(_sorted_dist, p)
        # "or" part: allow minor error
        # delay all normalization in the end to avoid numerical instability
        if token_ids[length_i] not in truncated_dist:
            truncated_dist[token_ids[length_i]] = unnormalized_dist[
                token_ids[length_i]
            ].logprob
    keys = list(truncated_dist.keys())
    assert (
        token_ids[length_i] in keys
    ), "Token not in the distribution: length-i:{}, token: {}, prob: {}".format(
        length_i,
        token_ids[length_i],
        np.exp(unnormalized_dist[token_ids[length_i]].logprob),
    )
    values = [truncated_dist[key] for key in keys]
    normalized_dist_values = torch.softmax(torch.tensor(values), dim=0)
    return keys, normalized_dist_values


def token_entropy(token_logprob: dict[int, Logprob]):
    """Compute entropy of "next-token" logprob distribution"""
    top_token_logprob = list(token_logprob.values())[0].logprob

    token_probs = [
        (token_id, logprob_info.logprob)
        for token_id, logprob_info in token_logprob.items()
    ]

    entropy = -sum(np.exp(logprob) * logprob for _, logprob in token_probs)

    return top_token_logprob, entropy


def get_token_level_entropy_truncated(
    outputs: list[CompletionOutput], p: float, max_length=None, top_p_mode=False
):
    """Given a list of vLLM outputs, returns truncated token-level entropy"""
    entropies = []
    all_tok_ids = []
    for output in outputs:
        token_ids = output.token_ids

        per_output_entropies = _get_token_level_entropy_truncated(
            logprobs=output.logprobs,
            token_ids=token_ids,
            p=p,
            max_length=max_length,
            top_p_mode=top_p_mode
        )
        
        entropies.append(per_output_entropies)
        all_tok_ids.append(token_ids)
    return entropies, all_tok_ids


def get_token_level_entropy(outputs: list[CompletionOutput]):
    """Given a list of vLLM outputs, returns token-level entropy"""
    logprobs: List[List[dict[int, Logprob]]] = [single_output.logprobs for single_output in outputs]

    per_output_logprobs = []
    per_output_logprobs_truncated = []
    per_output_entropies = []
    for sequence_logprobs in logprobs:
        logprobs = []
        logprobs_truncated = []
        entropy_vals = []

        for token_logprob in sequence_logprobs:
            if token_logprob:
                top_token_logprob, entropy = token_entropy(token_logprob)
                entropy_vals.append(entropy)
                logprobs.append(top_token_logprob)
                if top_token_logprob != 0.0:
                    logprobs_truncated.append(top_token_logprob)

        per_output_logprobs.append(logprobs)
        per_output_logprobs_truncated.append(logprobs_truncated)
        per_output_entropies.append(entropy_vals)

    return per_output_logprobs, per_output_logprobs_truncated, per_output_entropies
