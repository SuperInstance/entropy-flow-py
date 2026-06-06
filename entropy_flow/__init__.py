"""entropy-flow: Information-theoretic measures for distributions and time series."""
import numpy as np
from scipy import stats

def shannon_entropy(p, base=2):
    """Compute Shannon entropy of a probability distribution."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if base == 2:
        return -np.sum(p * np.log2(p))
    elif base == np.e:
        return -np.sum(p * np.log(p))
    else:
        return -np.sum(p * np.log(p) / np.log(base))

def kl_divergence(p, q):
    """KL divergence D(P || Q)."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def js_divergence(p, q):
    """Jensen-Shannon divergence (symmetric)."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def mutual_information(x, y, bins=10):
    """Estimate mutual information I(X; Y) via histogram binning."""
    x, y = np.asarray(x), np.asarray(y)
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_xy / hist_xy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mask = pxy > 0
    px_expand = px[:, np.newaxis]
    py_expand = py[np.newaxis, :]
    return np.sum(pxy[mask] * np.log2(pxy[mask] / (px_expand * py_expand)[mask]))

def entropy_rate(series, k=1, base=2):
    """Estimate entropy rate of a time series using k-history."""
    series = np.asarray(series, dtype=np.int64)
    if k == 0:
        _, counts = np.unique(series, return_counts=True)
        p = counts / counts.sum()
        return shannon_entropy(p, base)
    
    # Build (k+1)-grams as proper tuples
    n = len(series)
    
    # Conditional entropy H(X_t | X_{t-1}, ..., X_{t-k})
    from collections import Counter
    context_counts = Counter()
    joint_counts = Counter()
    for i in range(n - k):
        context = tuple(series[i:i+k])
        joint = tuple(series[i:i+k+1])
        context_counts[context] += 1
        joint_counts[joint] += 1
    
    h = 0.0
    total = n - k
    for joint, count in joint_counts.items():
        context = joint[:k]
        p_joint = count / total
        p_context = context_counts[context] / total
        if base == 2:
            h -= p_joint * np.log2(p_joint / p_context)
        else:
            h -= p_joint * np.log(p_joint / p_context) / np.log(base)
    return h

def transfer_entropy(source, target, k=1, bins=10):
    """Transfer entropy from source to target."""
    source, target = np.asarray(source), np.asarray(target)
    n = min(len(source), len(target))
    source, target = source[:n], target[:n]
    
    # Discretize if continuous
    if not np.issubdtype(source.dtype, np.integer):
        source = np.digitize(source, np.linspace(source.min(), source.max(), bins))
    if not np.issubdtype(target.dtype, np.integer):
        target = np.digitize(target, np.linspace(target.min(), target.max(), bins))
    
    from collections import Counter
    counts_3 = Counter()  # (target_past, source_past, target_now)
    counts_2a = Counter()  # (target_past, source_past)
    counts_2b = Counter()  # (target_past, target_now)
    counts_1 = Counter()   # (target_past)
    
    for i in range(k, n):
        tp = tuple(target[i-k:i])
        sp = tuple(source[i-k:i])
        tn = target[i]
        key3 = (tp, sp, tn)
        key2a = (tp, sp)
        key2b = (tp, tn)
        key1 = tp
        counts_3[key3] += 1
        counts_2a[key2a] += 1
        counts_2b[key2b] += 1
        counts_1[key1] += 1
    
    total = n - k
    te = 0.0
    for (tp, sp, tn), c3 in counts_3.items():
        p3 = c3 / total
        p2a = counts_2a[(tp, sp)] / total
        p2b = counts_2b[(tp, tn)] / total
        p1 = counts_1[tp] / total
        te += p3 * np.log2((p3 * p1) / (p2a * p2b))
    return te

def permutation_entropy(series, order=3, delay=1, normalize=True):
    """Permutation entropy of a time series."""
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    from itertools import permutations
    patterns = list(permutations(range(order)))
    pattern_counts = {p: 0 for p in patterns}
    
    for i in range(n - (order - 1) * delay):
        window = [series[i + j * delay] for j in range(order)]
        ranks = tuple(np.argsort(window))
        # Map to canonical permutation
        pattern_counts[ranks] = pattern_counts.get(ranks, 0) + 1
    
    total = sum(pattern_counts.values())
    probs = np.array([c / total for c in pattern_counts.values() if c > 0])
    h = shannon_entropy(probs, base=2)
    
    if normalize:
        h /= np.log2(len(patterns))
    return h

def sample_entropy(series, m=2, r=None):
    """Sample entropy of a time series."""
    series = np.asarray(series, dtype=np.float64)
    if r is None:
        r = 0.2 * np.std(series)
    n = len(series)
    
    def _count_templates(m_val):
        count = 0
        for i in range(n - m_val):
            for j in range(i + 1, n - m_val):
                if np.max(np.abs(series[i:i+m_val] - series[j:j+m_val])) <= r:
                    count += 1
        return count
    
    b = _count_templates(m)
    a = _count_templates(m + 1)
    
    if b == 0:
        return np.inf
    return -np.log(a / b)

def approximate_entropy(series, m=2, r=None):
    """Approximate entropy of a time series."""
    series = np.asarray(series, dtype=np.float64)
    if r is None:
        r = 0.2 * np.std(series)
    n = len(series)
    
    def _phi(m_val):
        counts = []
        for i in range(n - m_val + 1):
            template = series[i:i+m_val]
            count = 0
            for j in range(n - m_val + 1):
                if np.max(np.abs(template - series[j:j+m_val])) <= r:
                    count += 1
            counts.append(count / (n - m_val + 1))
        return -np.sum(np.log(counts)) / (n - m_val + 1)
    
    return _phi(m) - _phi(m + 1)
