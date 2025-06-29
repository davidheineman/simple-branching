import torch
import numpy as np

TOLERANCE_INF = 1e-12


def get_truncated_dist_top_p(sorted_dist, top_p, tolerance_inf=TOLERANCE_INF):
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


def get_logprob_per_token_from_vllm_outputs(vllm_output_token):
    # vllm updated implementation, for backward compatibility
    return vllm_output_token if isinstance(vllm_output_token, float) else vllm_output_token.logprob


def get_tokenwise_entropy_from_vllm_outputs(
    outputs, p, max_length=None, top_p_mode=False
):
    buf = []
    for output in outputs:
        _per_output_buf = []
        gen_seq_len = len(output.logprobs)
        token_ids = output.token_ids
        _max_length = (
            min(gen_seq_len, max_length) if max_length is not None else gen_seq_len
        )
        for length_i in range(_max_length):
            unnormalized_dist = output.logprobs[length_i]
            if unnormalized_dist is None:
                break
            keys, normalized_dist_values = get_token_truncated_dist_from_vllm_outputs(
                unnormalized_dist, token_ids, length_i, p, top_p_mode
            )
            entropy = -torch.sum(
                normalized_dist_values * torch.log(normalized_dist_values)
            )
            _per_output_buf.append(entropy.item())
        buf.append([_per_output_buf, token_ids])
    return buf


def get_token_truncated_dist_from_vllm_outputs(
    unnormalized_dist, token_ids, length_i, p, top_p_mode=False
):
    if not top_p_mode:
        # min_p mode
        truncated_dist = dict()
        for key in unnormalized_dist:
            _value = get_logprob_per_token_from_vllm_outputs(unnormalized_dist[key])
            if np.exp(_value) >= p or key == token_ids[length_i]:
                # "or" part: allow minor error
                # delay all normalization in the end to avoid numerical instability
                truncated_dist[key] = _value
    else:
        _unnormalized_dist = {
            key: get_logprob_per_token_from_vllm_outputs(unnormalized_dist[key])
            for key in unnormalized_dist
        }
        _sorted_dist = sorted(
            _unnormalized_dist.items(), key=lambda x: x[1], reverse=True
        )
        truncated_dist = get_truncated_dist_top_p(_sorted_dist, p)
        # "or" part: allow minor error
        # delay all normalization in the end to avoid numerical instability
        if token_ids[length_i] not in truncated_dist:
            truncated_dist[token_ids[length_i]] = (
                get_logprob_per_token_from_vllm_outputs(
                    unnormalized_dist[token_ids[length_i]]
                )
            )
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
