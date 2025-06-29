DEFAULT_ASYMPTOTIC_LIMIT = 50


def compute_bf_values(entropies, logliks, asymptotic_limit=DEFAULT_ASYMPTOTIC_LIMIT):
    # Important Note: here entropies and logliks both have to be "Accumulated"!!
    ret = []
    assert len(entropies) == len(
        logliks
    ), "different instance numbers for: {} vs {}".format(len(entropies), len(logliks))
    for entropy, loglik in zip(entropies, logliks):
        if len(entropy) > asymptotic_limit:
            ret.append(-loglik[-1])
        else:
            ret.append(entropy[-1] / len(entropy))
    return ret
