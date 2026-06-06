#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::string::ToString;
#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(feature = "std")]
use std::format;

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::string::ToString;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Default tolerance for floating-point comparisons.
pub const DEFAULT_TOLERANCE: f64 = 1e-9;

/// Natural log of 2, used for base-2 entropy.
const LN_2: f64 = 0.6931471805599453;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// Result of a single conservation-law check.
#[derive(Debug, Clone)]
pub struct ConservationReport {
    /// Observed sum minus expected total.
    pub delta: f64,
    /// Tolerance used for the check.
    pub tolerance: f64,
    /// Whether the check passed.
    pub passed: bool,
    /// Monotonic timestamp (milliseconds since some epoch; caller-defined).
    pub timestamp_ms: u64,
    /// Optional label for the check.
    pub label: Option<String>,
}

impl ConservationReport {
    /// Create a new report from parts and a total.
    pub fn new(parts: &[f64], total: f64, tolerance: f64, timestamp_ms: u64) -> Self {
        let sum: f64 = parts.iter().copied().sum();
        let delta = sum - total;
        let passed = delta.abs() <= tolerance;
        Self {
            delta,
            tolerance,
            passed,
            timestamp_ms,
            label: None,
        }
    }

    /// Attach a label.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Serialize to compact JSON (hand-rolled, no serde).
    pub fn to_json(&self) -> String {
        let label_json = match &self.label {
            Some(l) => format!(",\"label\":\"{}\"", l),
            None => String::new(),
        };
        format!(
            "{{\"delta\":{},{},\"tolerance\":{},{},\"ts\":{}}}",
            fmt_f64(self.delta),
            if self.passed { "\"passed\":true" } else { "\"passed\":false" },
            fmt_f64(self.tolerance),
            label_json,
            self.timestamp_ms,
        )
    }
}

/// Summary report from [`EdgeVerifier`].
#[derive(Debug, Clone)]
pub struct SummaryReport {
    pub checks: Vec<ConservationReport>,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub timestamp_ms: u64,
}

impl SummaryReport {
    pub fn to_json(&self) -> String {
        let items: Vec<String> = self.checks.iter().map(|c| c.to_json()).collect();
        format!(
            "{{\"total\":{},\"passed\":{},\"failed\":{},\"ts\":{},\"checks\":[{}]}}",
            self.total,
            self.passed,
            self.failed,
            self.timestamp_ms,
            items.join(","),
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Verify that the sum of `parts` equals `total` within `tolerance`.
pub fn verify_conservation(parts: &[f64], total: f64) -> ConservationReport {
    verify_conservation_with_tolerance(parts, total, DEFAULT_TOLERANCE)
}

/// Same as [`verify_conservation`] but with an explicit tolerance.
pub fn verify_conservation_with_tolerance(
    parts: &[f64],
    total: f64,
    tolerance: f64,
) -> ConservationReport {
    let ts = raw_timestamp_ms();
    ConservationReport::new(parts, total, tolerance, ts)
}

/// Compute Shannon entropy in base-2 for a probability distribution.
///
/// H(p) = -Σ p_i * log2(p_i)
///
/// Elements that are zero are skipped (0 * log(0) → 0).
pub fn shannon_entropy(probs: &[f64]) -> f64 {
    let mut h = 0.0f64;
    for &p in probs {
        if p > 0.0 {
            h -= p * (ln_f64(p) / LN_2);
        }
    }
    h
}

/// Compute KL divergence D(p || q) in base-2.
///
/// D(p||q) = Σ p_i * log2(p_i / q_i)
///
/// Panics if `p` and `q` have different lengths or if any `q_i == 0` where `p_i > 0`.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "p and q must have equal length");
    let mut d = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            assert!(qi > 0.0, "q_i must be > 0 where p_i > 0");
            d += pi * (ln_f64(pi / qi) / LN_2);
        }
    }
    d
}

/// Verify the determinant of a 2×2 matrix against an expected value.
///
/// `m` is row-major: `[a, b, c, d]` → `|a b; c d| = ad - bc`.
pub fn verify_determinant(m: &[f64; 4], expected_det: f64) -> bool {
    let det = m[0] * m[3] - m[1] * m[2];
    (det - expected_det).abs() <= DEFAULT_TOLERANCE
}

// ---------------------------------------------------------------------------
// EdgeVerifier — accumulator
// ---------------------------------------------------------------------------

/// Accumulates multiple conservation checks and produces a summary report.
pub struct EdgeVerifier {
    checks: Vec<ConservationReport>,
    now_fn: fn() -> u64,
}

impl EdgeVerifier {
    /// Create a new verifier using the default monotonic timestamp source.
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            now_fn: raw_timestamp_ms,
        }
    }

    /// Create a verifier with a custom clock (useful for testing).
    pub fn with_clock(now_fn: fn() -> u64) -> Self {
        Self {
            checks: Vec::new(),
            now_fn,
        }
    }

    /// Verify sum(parts) == total within tolerance and record the result.
    pub fn verify(&mut self, parts: &[f64], total: f64, tolerance: f64, label: &str) -> bool {
        let ts = (self.now_fn)();
        let report = ConservationReport::new(parts, total, tolerance, ts).with_label(label);
        let passed = report.passed;
        self.checks.push(report);
        passed
    }

    /// Verify a 2×2 determinant and record the result.
    pub fn verify_det(&mut self, m: &[f64; 4], expected: f64, label: &str) -> bool {
        let ts = (self.now_fn)();
        let det = m[0] * m[3] - m[1] * m[2];
        let passed = (det - expected).abs() <= DEFAULT_TOLERANCE;
        let delta = det - expected;
        self.checks.push(ConservationReport {
            delta,
            tolerance: DEFAULT_TOLERANCE,
            passed,
            timestamp_ms: ts,
            label: Some(label.to_string()),
        });
        passed
    }

    /// Consume the verifier and produce a summary.
    pub fn summary(self) -> SummaryReport {
        let ts = (self.now_fn)();
        let total = self.checks.len();
        let passed = self.checks.iter().filter(|c| c.passed).count();
        SummaryReport {
            checks: self.checks,
            total,
            passed,
            failed: total - passed,
            timestamp_ms: ts,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Monotonic counter-based timestamp.
fn raw_timestamp_ms() -> u64 {
    #[cfg(feature = "std")]
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }
    #[cfg(not(feature = "std"))]
    {
        static mut COUNTER: u64 = 0;
        // In no_std embedded, supply your own clock via with_clock().
        unsafe { COUNTER += 1; COUNTER }
    }
}

/// Wrapper for f64 natural log that works in both std and no_std.
#[cfg(feature = "std")]
fn ln_f64(x: f64) -> f64 {
    x.ln()
}

#[cfg(not(feature = "std"))]
fn ln_f64(x: f64) -> f64 {
    libm::log(x)
}

/// Format an f64 as a compact JSON number.
pub(crate) fn fmt_f64(v: f64) -> String {
    if v.is_nan() {
        return "null".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_positive() { "1e308".to_string() } else { "-1e308".to_string() };
    }
    let s = format!("{}", v);
    if s.contains('.') {
        let trimmed = s.trim_end_matches('0');
        if trimmed.ends_with('.') {
            format!("{}0", trimmed)
        } else {
            trimmed.to_string()
        }
    } else {
        s
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
