# edge-conservation

Conservation-law verification for edge and embedded deployment. Small binary, `no_std` compatible, zero dependencies.

## Features

- **`verify_conservation`** — check that `sum(parts) == total` within configurable tolerance
- **`shannon_entropy`** — compute Shannon entropy in base-2
- **`kl_divergence`** — compute KL divergence D(p‖q) in base-2
- **`verify_determinant`** — verify 2×2 matrix determinant conservation
- **`EdgeVerifier`** — accumulate multiple checks and produce a summary report
- **Hand-rolled JSON serialization** — no serde, no macros, just compact JSON strings

## Usage

```rust
use edge_conservation::*;

// Simple conservation check
let report = verify_conservation(&[0.3, 0.5, 0.2], 1.0);
assert!(report.passed);

// Custom tolerance
let report = verify_conservation_with_tolerance(&[0.1, 0.2, 0.3], 0.6, 1e-6);
assert!(report.passed);

// Shannon entropy
let h = shannon_entropy(&[0.5, 0.5]); // = 1.0 bit (fair coin)

// KL divergence
let d = kl_divergence(&[1.0, 0.0], &[0.5, 0.5]); // = 1.0

// 2×2 determinant verification
assert!(verify_determinant(&[2.0, 3.0, 1.0, 4.0], 5.0)); // ad - bc = 5

// Accumulate multiple checks
let mut verifier = EdgeVerifier::new();
verifier.verify(&[1.0, 2.0, 3.0], 6.0, 1e-9, "mass balance");
verifier.verify_det(&[1.0, 0.0, 0.0, 1.0], 1.0, "identity det");
let summary = verifier.summary();
println!("{}", summary.to_json());
```

## `no_std` Usage

Disable the default `std` feature and add `libm` for floating-point math:

```toml
[dependencies]
edge-conservation = { version = "0.1", default-features = false }
libm = "0.2"
```

Supply your own clock via `EdgeVerifier::with_clock(your_fn)` for timestamping.

## Binary Size

Built with `--release` and the provided profile (opt-level=z, LTO, strip):

```
$ cargo b --release
$ ls -lh target/release/libedge_conservation.rlib
```

Designed for constrained environments — no allocator required for the core math functions (only JSON serialization and `EdgeVerifier` need `alloc`).

## API

| Function | Description |
|---|---|
| `verify_conservation(parts, total)` | Check sum(parts) ≈ total |
| `verify_conservation_with_tolerance(parts, total, tol)` | Same, custom tolerance |
| `shannon_entropy(probs)` | Shannon entropy H(p) in base-2 |
| `kl_divergence(p, q)` | KL divergence D(p‖q) in base-2 |
| `verify_determinant(m, expected)` | Verify 2×2 determinant |
| `ConservationReport` | Single check result (delta, tolerance, passed, timestamp) |
| `EdgeVerifier` | Accumulate checks → `SummaryReport` |
| `SummaryReport::to_json()` | Compact JSON output |

## License

MIT
