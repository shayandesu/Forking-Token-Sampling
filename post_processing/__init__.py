from math import comb

def evaluate_pass_at_k(num_correct, num_samples, k):
    """
    Unbiased estimator for pass@k: probability that at least one of k samples is correct.
    Given n samples with c correct: pass@k = 1 - C(n - c, k) / C(n, k).
    """
    n, c = num_samples, num_correct
    if k <= 0 or n <= 0:
        return 0.0
    if n < k:
        # Fewer than k samples: count as pass if any correct
        return 1.0 if c > 0 else 0.0
    if c <= 0:
        return 0.0
    
    if n - c < k:
        return 1.0

    return 1.0 - comb(n - c, k) / comb(n, k)
