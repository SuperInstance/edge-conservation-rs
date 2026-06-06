#[cfg(test)]
mod tests {
    use crate::*;

    // ---- verify_conservation ----

    #[test]
    fn conservation_exact_sum() {
        let r = verify_conservation(&[1.0, 2.0, 3.0], 6.0);
        assert!(r.passed);
        assert!((r.delta).abs() < 1e-12);
    }

    #[test]
    fn conservation_fails_on_bad_sum() {
        let r = verify_conservation(&[1.0, 2.0, 3.0], 7.0);
        assert!(!r.passed);
    }

    #[test]
    fn conservation_tolerance_respected() {
        let r = verify_conservation_with_tolerance(&[0.1, 0.2, 0.3], 0.6, 0.01);
        // 0.1+0.2+0.3 ≈ 0.6000000000000001 due to float, so delta ≈ 5.5e-17
        assert!(r.passed);
    }

    #[test]
    fn conservation_empty_parts() {
        let r = verify_conservation(&[], 0.0);
        assert!(r.passed);
    }

    #[test]
    fn conservation_negative_parts() {
        let r = verify_conservation(&[5.0, -2.0, -3.0], 0.0);
        assert!(r.passed);
    }

    // ---- shannon_entropy ----

    #[test]
    fn entropy_uniform_distribution() {
        // Fair coin: H = 1 bit
        let h = shannon_entropy(&[0.5, 0.5]);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn entropy_certain_outcome() {
        // p=[1,0] → H=0
        let h = shannon_entropy(&[1.0, 0.0]);
        assert!((h - 0.0).abs() < 1e-10);
    }

    #[test]
    fn entropy_four_way_uniform() {
        // Four equal outcomes: H = 2 bits
        let h = shannon_entropy(&[0.25, 0.25, 0.25, 0.25]);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn entropy_empty() {
        assert_eq!(shannon_entropy(&[]), 0.0);
    }

    // ---- kl_divergence ----

    #[test]
    fn kl_identical_distributions() {
        let p = [0.25, 0.25, 0.25, 0.25];
        let d = kl_divergence(&p, &p);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn kl_known_value() {
        // p=[1,0], q=[0.5,0.5] → D = 1*log2(1/0.5) = 1
        let d = kl_divergence(&[1.0, 0.0], &[0.5, 0.5]);
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn kl_unequal_lengths_panics() {
        let _ = kl_divergence(&[0.5], &[0.25, 0.75]);
    }

    #[test]
    #[should_panic(expected = "q_i must be > 0")]
    fn kl_zero_q_nonzero_p_panics() {
        let _ = kl_divergence(&[0.5, 0.5], &[1.0, 0.0]);
    }

    // ---- verify_determinant ----

    #[test]
    fn det_identity() {
        assert!(verify_determinant(&[1.0, 0.0, 0.0, 1.0], 1.0));
    }

    #[test]
    fn det_known_2x2() {
        // |2 3; 1 4| = 8-3 = 5
        assert!(verify_determinant(&[2.0, 3.0, 1.0, 4.0], 5.0));
    }

    #[test]
    fn det_singular() {
        // |1 2; 2 4| = 0
        assert!(verify_determinant(&[1.0, 2.0, 2.0, 4.0], 0.0));
    }

    #[test]
    fn det_wrong_expected() {
        assert!(!verify_determinant(&[1.0, 0.0, 0.0, 1.0], 2.0));
    }

    // ---- ConservationReport JSON ----

    #[test]
    fn report_json_has_fields() {
        let r = ConservationReport::new(&[1.0, 2.0], 3.0, 1e-9, 42)
            .with_label("test");
        let j = r.to_json();
        assert!(j.contains("\"passed\":true"));
        assert!(j.contains("\"ts\":42"));
        assert!(j.contains("\"label\":\"test\""));
    }

    #[test]
    fn report_json_without_label() {
        let r = ConservationReport::new(&[1.0], 1.0, 1e-9, 0);
        let j = r.to_json();
        assert!(!j.contains("label"));
    }

    // ---- EdgeVerifier ----

    #[test]
    fn verifier_all_pass() {
        static mut CLOCK: u64 = 100;
        fn clock() -> u64 {
            unsafe { let t = CLOCK; CLOCK += 1; t }
        }
        let mut v = EdgeVerifier::with_clock(clock);
        assert!(v.verify(&[1.0, 2.0, 3.0], 6.0, 1e-9, "sum"));
        assert!(v.verify_det(&[1.0, 0.0, 0.0, 1.0], 1.0, "det"));
        let s = v.summary();
        assert_eq!(s.total, 2);
        assert_eq!(s.passed, 2);
        assert_eq!(s.failed, 0);
    }

    #[test]
    fn verifier_mixed_results() {
        static mut CLOCK: u64 = 200;
        fn clock() -> u64 {
            unsafe { let t = CLOCK; CLOCK += 1; t }
        }
        let mut v = EdgeVerifier::with_clock(clock);
        assert!(v.verify(&[1.0, 2.0], 3.0, 1e-9, "ok"));
        assert!(!v.verify(&[1.0, 2.0], 4.0, 1e-9, "bad"));
        let s = v.summary();
        assert_eq!(s.passed, 1);
        assert_eq!(s.failed, 1);
        // Check summary JSON is valid-ish
        let j = s.to_json();
        assert!(j.starts_with('{'));
        assert!(j.contains("\"total\":2"));
    }

    // ---- fmt_f64 edge cases ----

    #[test]
    fn fmt_f64_normal() {
        assert_eq!(fmt_f64(1.234), "1.234");
    }

    #[test]
    fn fmt_f64_nan() {
        assert_eq!(fmt_f64(f64::NAN), "null");
    }
}
