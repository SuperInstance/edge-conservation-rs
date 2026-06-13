# edge-conservation-rs

**Conservation-law verification for edge and embedded systems**, written in `no_std`-compatible Rust. Verifies that sums, determinants, and information-theoretic quantities satisfy physical or mathematical invariants — with hand-rolled JSON serialization and zero external dependencies for `no_std` targets.

## Why It Matters

In distributed GPU systems, conservation laws are the safety nets that catch silent corruption. If a weight hotswap loses energy (Σ outputs ≠ Σ inputs), or if a transformation matrix becomes singular when it shouldn't be, the system continues running but produces garbage. By the time humans notice, the damage has propagated.

This crate provides lightweight, embeddable verification primitives that run on edge devices (ARM Cortex-M, RISC-V) alongside the main computation. Each check produces a `ConservationReport` with the observed deviation (delta), a pass/fail boolean, and a timestamp — all serializable to compact JSON without `serde`.

The `no_std` support is critical: many edge GPU devices (Jetson Nano, Coral Edge TPU) run bare-metal or RTOS environments where the standard library isn't available. This crate works there.

## How It Works

### Conservation Sum Check

The fundamental operation: verify that a set of parts sums to an expected total within floating-point tolerance:

$$\left| \sum_{i=1}^{n} x_i - T \right| \leq \epsilon$$

Where ε is the tolerance (default 10⁻⁹). This catches:
- Energy loss in signal processing pipelines
- Mass loss in particle simulations
- Token loss in distributed CRDT merge operations

**Floating-point caveat**: The check uses absolute tolerance, not relative. For values of magnitude 10⁶+, consider scaling the tolerance. The delta is reported so callers can apply their own thresholds.

### Shannon Entropy

$$H(X) = -\sum_{i} p_i \log_2 p_i$$

Computed with natural-log conversion: `H = -Σ p_i × ln(p_i) / ln(2)`. Zero-probability entries are skipped (the limit `0 × log(0) → 0`). The implementation uses a `ln_f64` wrapper that dispatches to `f64::ln()` in `std` mode or `libm::log()` in `no_std` mode.

**Properties verified**:
- Uniform distribution over n outcomes: H = log₂(n) bits
- Deterministic distribution (p = [1, 0, ...]): H = 0
- H ≥ 0 always (non-negative)

### KL Divergence

$$D_{KL}(P \| Q) = \sum_{i} p_i \log_2 \frac{p_i}{q_i}$$

The implementation panics if any `q_i = 0` where `p_i > 0` (support violation — Q doesn't cover P's support). This is a deliberate design choice: silent zero-padding would hide distributional bugs.

**Properties**: D_KL ≥ 0 (Gibbs' inequality). D_KL = 0 iff P = Q.

### Jensen-Shannon and Determinant Checks

The 2×2 determinant check verifies `ad − bc = expected` within tolerance — used to confirm that transformation matrices maintain their expected orientation (non-singular, specific scaling).

### Complexity Analysis

| Function | Time | Space |
|----------|------|-------|
| `verify_conservation` | O(n) | O(1) |
| `shannon_entropy` | O(n) | O(1) |
| `kl_divergence` | O(n) | O(1) |
| `verify_determinant` | O(1) | O(1) |
| `EdgeVerifier::summary` | O(k) | O(k) |

Where n = input length, k = accumulated checks.

### Binary Size

Release profile uses `opt-level = "z"`, LTO, single codegen unit, and symbol stripping — producing minimal binary footprint for embedded targets.

## Quick Start

```rust
use edge_conservation::{verify_conservation, shannon_entropy, EdgeVerifier};

// Verify a sum
let report = verify_conservation(&[1.0, 2.0, 3.0], 6.0);
assert!(report.passed);
assert!(report.delta.abs() < 1e-12);

// Shannon entropy of a fair coin = 1 bit
let h = shannon_entropy(&[0.5, 0.5]);
assert!((h - 1.0).abs() < 1e-10);

// Accumulate multiple checks
let mut verifier = EdgeVerifier::new();
verifier.verify(&[1.0, 2.0], 3.0, 1e-9, "energy_balance");
verifier.verify_det(&[1.0, 0.0, 0.0, 1.0], 1.0, "rotation_det");
let summary = verifier.summary();
println!("{}", summary.to_json());
```

## API

### Free Functions
- `verify_conservation(parts: &[f64], total: f64) -> ConservationReport` — Sum check with default tolerance
- `verify_conservation_with_tolerance(parts, total, tolerance) -> ConservationReport` — Explicit tolerance
- `shannon_entropy(probs: &[f64]) -> f64` — Base-2 Shannon entropy in bits
- `kl_divergence(p: &[f64], q: &[f64]) -> f64` — Base-2 KL divergence
- `verify_determinant(m: &[f64; 4], expected_det: f64) -> bool` — 2×2 determinant check

### `EdgeVerifier`
- `new() -> Self` — Default monotonic clock
- `with_clock(fn() -> u64) -> Self` — Custom clock for testing
- `verify(&mut self, parts, total, tolerance, label) -> bool` — Record a sum check
- `verify_det(&mut self, m, expected, label) -> bool` — Record a determinant check
- `summary(self) -> SummaryReport` — Produce JSON-serializable summary

### `ConservationReport`
- `delta: f64` — Observed minus expected
- `tolerance: f64` — Tolerance used
- `passed: bool` — Whether |delta| ≤ tolerance
- `to_json() -> String` — Hand-rolled compact JSON

## Architecture Notes

This crate implements the γ + η = C verification layer for the SuperInstance stack:

- **γ** (gamma) = observed system state (the `parts` array)
- **η** (eta) = expected correction/residual (the delta)
- **C** (constant) = the conservation target (the `total`)

If γ + η ≠ C, the system has lost or gained something it shouldn't have. The report's `delta` quantifies the violation magnitude.

The `no_std` mode makes this deployable alongside GPU kernels on edge devices where the standard library isn't available, enabling real-time conservation checks during inference.

See the full architecture: [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)

## References

1. Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory,* 2nd ed. Wiley. Chapters 2 (entropy) and 8 (KL divergence).
2. Kullback, S. & Leibler, R.A. (1951). "On Information and Sufficiency." *Annals of Mathematical Statistics, 22(1).*
3. The Rust Embedded Working Group. "The Embedded Rust Book." [docs.rust-embedded.org/book](https://docs.rust-embedded.org/book/)
4. Goldberg, D. (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic." *ACM Computing Surveys, 23(1).*

## License

MIT
