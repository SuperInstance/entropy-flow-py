# Entropy Flow (Python)

**Entropy Flow** is a Python library implementing information-theoretic measures for probability distributions and time series — including Shannon entropy, KL divergence, Jensen-Shannon divergence, mutual information, transfer entropy, permutation entropy, and sample entropy.

## Why It Matters

Information theory provides the mathematical language for quantifying uncertainty, correlation, and complexity. These measures are fundamental to machine learning (cross-entropy loss, information gain in decision trees), neuroscience (neural coding efficiency), physics (thermodynamic entropy), and signal processing (complexity analysis). This library brings together the most important entropy-based measures in a single, consistent API built on NumPy and SciPy. Unlike specialized libraries that focus on one measure, Entropy Flow provides the full toolkit: static measures (Shannon, KL, JS), dynamic measures (transfer entropy, entropy rate), and complexity measures (permutation entropy, sample entropy, approximate entropy).

## How It Works

**Shannon Entropy:**
```
H(p) = −Σ pᵢ log₂(pᵢ)
```
Maximum: log₂(n) for uniform distribution over n elements. Computed in O(n).

**KL Divergence:**
```
D(P‖Q) = Σ pᵢ log(pᵢ/qᵢ)
```
Asymmetric (D(P‖Q) ≠ D(Q‖P)), non-negative, zero iff P = Q. Measures information lost when Q approximates P.

**Jensen-Shannon Divergence:**
```
JS(P‖Q) = ½ D(P‖M) + ½ D(Q‖M),  where M = ½(P + Q)
```
Symmetric, bounded [0, 1], always finite. The square root of JS divergence is a metric (satisfies triangle inequality).

**Mutual Information:**
```
I(X; Y) = ΣΣ p(x,y) log₂(p(x,y) / (p(x)p(y)))
```
Estimated via 2D histogram binning. O(n × bins) computation.

**Transfer Entropy:**
```
TE(S→T) = Σ p(tₙ₊₁, tₙ, sₙ) log₂(p(tₙ₊₁|tₙ,sₙ) / p(tₙ₊₁|tₙ))
```
Measures directed information flow from source S to target T, beyond what T's own history predicts. Computed via k-history context counting. O(n × k) for context construction.

**Permutation Entropy:**
```
PE = H(π) / log₂(m!)
```
Where π ranges over m! permutation patterns of order m. Normalized to [0, 1]. Measures complexity of temporal ordering.

**Sample Entropy:**
```
SampEn = −ln(A/B)
```
Where B = matching templates of length m, A = matching templates of length m+1, within tolerance r (typically 0.2σ). O(n²) computation.

## Quick Start

```python
import numpy as np
from entropy_flow import shannon_entropy, kl_divergence, mutual_information

p = [0.5, 0.3, 0.2]
print(f"H(p) = {shannon_entropy(p):.4f}")  # 1.485 bits

q = [0.4, 0.4, 0.2]
print(f"D(p||q) = {kl_divergence(p, q):.4f}")

x = np.random.randn(1000)
y = 0.5 * x + np.random.randn(1000) * 0.5
print(f"I(X;Y) = {mutual_information(x, y):.4f} bits")
```

## API

| Function | Description |
|----------|-------------|
| `shannon_entropy(p, base)` | Shannon entropy H(p) |
| `kl_divergence(p, q)` | KL divergence D(P‖Q) |
| `js_divergence(p, q)` | Jensen-Shannon divergence |
| `mutual_information(x, y, bins)` | I(X; Y) via histogram |
| `entropy_rate(series, k)` | Conditional entropy H(X_t \| history) |
| `transfer_entropy(source, target, k)` | Directed information flow |
| `permutation_entropy(series, order, delay)` | Complexity of ordering patterns |
| `sample_entropy(series, m, r)` | Template-matching complexity |
| `approximate_entropy(series, m, r)` | Approximate complexity |

## Architecture Notes

Entropy Flow (Python) provides the **analytical toolkit** for measuring γ + η = C conservation. The avoidance ratio's conservation (Law 5, σ ≈ 0.001) is quantified using Shannon entropy across population scales. Transfer entropy measures the directed information flow between γ-layer (actions) and η-layer (models), and mutual information quantifies their statistical coupling.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27.
2. Schreiber, T. (2000). "Measuring Information Transfer." *Physical Review Letters*, 85(2), 461–464.
3. Richman, J.S. & Moorman, J.R. (2000). "Physiological Time-Series Analysis Using Approximate Entropy and Sample Entropy." *American Journal of Physiology*.

## License

MIT
