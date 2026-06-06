# entropy-flow

Information-theoretic measures for probability distributions and time series. Shannon entropy, divergences, mutual information, permutation entropy, sample entropy, and transfer entropy.

## Features

- **Shannon Entropy** — H(X) for arbitrary distributions, configurable base
- **KL Divergence** — D(P‖Q) with automatic zero-handling
- **JS Divergence** — Symmetric, bounded [0, 1] in base 2
- **Mutual Information** — I(X; Y) via histogram binning
- **Entropy Rate** — Conditional entropy H(X_t | X_{t-1}...X_{t-k})
- **Transfer Entropy** — Directional information flow between time series
- **Permutation Entropy** — Ordinal pattern entropy, normalized [0, 1]
- **Sample Entropy** — Regularity measure for time series
- **Approximate Entropy** — Complexity measure for short sequences

## Installation

```bash
pip install entropy-flow
```

## Usage

```python
from entropy_flow import (
    shannon_entropy, kl_divergence, js_divergence,
    mutual_information, entropy_rate, transfer_entropy,
    permutation_entropy, sample_entropy, approximate_entropy
)
import numpy as np

# Shannon entropy of a fair coin
H = shannon_entropy([0.5, 0.5])  # 1.0 bit

# KL divergence
kl = kl_divergence([0.3, 0.7], [0.5, 0.5])

# Mutual information between correlated variables
np.random.seed(42)
x = np.random.randn(10000)
mi = mutual_information(x, x, bins=20)  # high MI for identical

# Permutation entropy of a periodic signal
signal = np.sin(np.linspace(0, 4*np.pi, 1000))
pe = permutation_entropy(signal, order=3)  # low for periodic

# Sample entropy of random vs regular
random_se = sample_entropy(np.random.randn(200), m=2)
regular_se = sample_entropy(np.sin(np.linspace(0, 10*np.pi, 200)), m=2)
# random_se > regular_se (more irregular = higher entropy)
```

## Testing

```bash
pytest tests/ -v    # 22 tests
```

## License

MIT
