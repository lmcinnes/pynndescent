//! Binary/set-based distances for boolean-like vectors.
//!
//! These distances treat non-zero values as "true" and zero as "false".

use super::traits::Distance;

/// Hamming distance: proportion of differing elements.
///
/// Formula: d(a, b) = (1/n) Σ 𝟙{aᵢ ≠ bᵢ}
#[derive(Clone, Copy, Debug, Default)]
pub struct Hamming;

impl Distance<f32> for Hamming {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        if n == 0 {
            return 0.0;
        }

        let mut count = 0u32;
        for i in 0..n {
            if a[i] != b[i] {
                count += 1;
            }
        }
        count as f32 / n as f32
    }

    fn name(&self) -> &'static str {
        "hamming"
    }
}

/// Jaccard distance: 1 - |intersection| / |union|.
///
/// Non-zero values are treated as set membership.
#[derive(Clone, Copy, Debug, Default)]
pub struct Jaccard;

impl Distance<f32> for Jaccard {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut num_non_zero = 0.0f32;
        let mut num_equal = 0.0f32;
        for i in 0..a.len() {
            let x_true = a[i] != 0.0;
            let y_true = b[i] != 0.0;
            if x_true || y_true {
                num_non_zero += 1.0;
            }
            if x_true && y_true {
                num_equal += 1.0;
            }
        }
        if num_non_zero == 0.0 {
            0.0
        } else {
            (num_non_zero - num_equal) / num_non_zero
        }
    }

    fn name(&self) -> &'static str {
        "jaccard"
    }
}

/// Alternative Jaccard using log transform for bounded-radius search.
///
/// Formula: d_alt(a, b) = -log₂(|intersection| / |union|)
/// Correction: 1 - 2^(-d_alt)
#[derive(Clone, Copy, Debug, Default)]
pub struct AlternativeJaccard;

impl Distance<f32> for AlternativeJaccard {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut num_non_zero = 0.0f32;
        let mut num_equal = 0.0f32;
        for i in 0..a.len() {
            let x_true = a[i] != 0.0;
            let y_true = b[i] != 0.0;
            if x_true || y_true {
                num_non_zero += 1.0;
            }
            if x_true && y_true {
                num_equal += 1.0;
            }
        }
        if num_non_zero == 0.0 {
            0.0
        } else {
            -(num_equal / num_non_zero).log2()
        }
    }

    fn name(&self) -> &'static str {
        "alternative_jaccard"
    }
}

/// Matching distance: proportion of boolean-differing elements.
///
/// Formula: d(a, b) = (1/n) Σ 𝟙{(aᵢ≠0) ≠ (bᵢ≠0)}
#[derive(Clone, Copy, Debug, Default)]
pub struct Matching;

impl Distance<f32> for Matching {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        if n == 0 {
            return 0.0;
        }

        let mut count = 0u32;
        for i in 0..n {
            if (a[i] != 0.0) != (b[i] != 0.0) {
                count += 1;
            }
        }
        count as f32 / n as f32
    }

    fn name(&self) -> &'static str {
        "matching"
    }
}

/// Dice (Sørensen-Dice) distance.
///
/// Formula: d(a, b) = |a⊕b| / (2|a∩b| + |a⊕b|)
#[derive(Clone, Copy, Debug, Default)]
pub struct Dice;

impl Distance<f32> for Dice {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut num_true_true = 0.0f32;
        let mut num_not_equal = 0.0f32;
        for i in 0..a.len() {
            let x_true = a[i] != 0.0;
            let y_true = b[i] != 0.0;
            if x_true && y_true {
                num_true_true += 1.0;
            }
            if x_true != y_true {
                num_not_equal += 1.0;
            }
        }
        if num_not_equal == 0.0 {
            0.0
        } else {
            num_not_equal / (2.0 * num_true_true + num_not_equal)
        }
    }

    fn name(&self) -> &'static str {
        "dice"
    }
}

/// Kulsinski distance.
///
/// Formula: d(a, b) = (|a⊕b| - |a∩b| + n) / (|a⊕b| + n)
#[derive(Clone, Copy, Debug, Default)]
pub struct Kulsinski;

impl Distance<f32> for Kulsinski {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len() as f32;

        let mut num_true_true = 0.0f32;
        let mut num_not_equal = 0.0f32;
        for i in 0..a.len() {
            let x_true = a[i] != 0.0;
            let y_true = b[i] != 0.0;
            if x_true && y_true {
                num_true_true += 1.0;
            }
            if x_true != y_true {
                num_not_equal += 1.0;
            }
        }
        if num_not_equal == 0.0 {
            0.0
        } else {
            (num_not_equal - num_true_true + n) / (num_not_equal + n)
        }
    }

    fn name(&self) -> &'static str {
        "kulsinski"
    }
}

/// Rogers-Tanimoto distance.
///
/// Formula: d(a, b) = 2|a⊕b| / (n + |a⊕b|)
#[derive(Clone, Copy, Debug, Default)]
pub struct RogersTanimoto;

impl Distance<f32> for RogersTanimoto {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len() as f32;

        let mut num_not_equal = 0.0f32;
        for i in 0..a.len() {
            if (a[i] != 0.0) != (b[i] != 0.0) {
                num_not_equal += 1.0;
            }
        }
        (2.0 * num_not_equal) / (n + num_not_equal)
    }

    fn name(&self) -> &'static str {
        "rogerstanimoto"
    }
}

/// Russell-Rao distance.
///
/// Formula: d(a, b) = (n - |a∩b|) / n
#[derive(Clone, Copy, Debug, Default)]
pub struct RussellRao;

impl Distance<f32> for RussellRao {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len() as f32;
        if n == 0.0 {
            return 0.0;
        }

        let mut num_true_true = 0.0f32;
        for i in 0..a.len() {
            if a[i] != 0.0 && b[i] != 0.0 {
                num_true_true += 1.0;
            }
        }
        (n - num_true_true) / n
    }

    fn name(&self) -> &'static str {
        "russellrao"
    }
}

/// Sokal-Michener distance (equivalent to Rogers-Tanimoto).
///
/// Formula: d(a, b) = 2|a⊕b| / (n + |a⊕b|)
#[derive(Clone, Copy, Debug, Default)]
pub struct SokalMichener;

impl Distance<f32> for SokalMichener {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // Identical to Rogers-Tanimoto
        RogersTanimoto.distance(a, b)
    }

    fn name(&self) -> &'static str {
        "sokalmichener"
    }
}

/// Sokal-Sneath distance.
///
/// Formula: d(a, b) = |a⊕b| / (0.5|a∩b| + |a⊕b|)
#[derive(Clone, Copy, Debug, Default)]
pub struct SokalSneath;

impl Distance<f32> for SokalSneath {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut num_true_true = 0.0f32;
        let mut num_not_equal = 0.0f32;
        for i in 0..a.len() {
            let x_true = a[i] != 0.0;
            let y_true = b[i] != 0.0;
            if x_true && y_true {
                num_true_true += 1.0;
            }
            if x_true != y_true {
                num_not_equal += 1.0;
            }
        }
        if num_not_equal == 0.0 {
            0.0
        } else {
            num_not_equal / (0.5 * num_true_true + num_not_equal)
        }
    }

    fn name(&self) -> &'static str {
        "sokalsneath"
    }
}

/// Yule distance.
///
/// Formula: d(a, b) = 2·nTF·nFT / (nTT·nFF + nTF·nFT)
#[derive(Clone, Copy, Debug, Default)]
pub struct Yule;

impl Distance<f32> for Yule {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut num_tt = 0.0f32;
        let mut num_tf = 0.0f32;
        let mut num_ft = 0.0f32;
        for i in 0..a.len() {
            let x_true = a[i] != 0.0;
            let y_true = b[i] != 0.0;
            if x_true && y_true {
                num_tt += 1.0;
            }
            if x_true && !y_true {
                num_tf += 1.0;
            }
            if !x_true && y_true {
                num_ft += 1.0;
            }
        }
        let num_ff = a.len() as f32 - num_tt - num_tf - num_ft;

        if num_tf == 0.0 || num_ft == 0.0 {
            0.0
        } else {
            (2.0 * num_tf * num_ft) / (num_tt * num_ff + num_tf * num_ft)
        }
    }

    fn name(&self) -> &'static str {
        "yule"
    }
}

/// Bit-packed Hamming distance for u8 arrays.
///
/// Each byte contains 8 packed binary features.
/// Returns the total number of differing bits (not normalized).
#[derive(Clone, Copy, Debug, Default)]
pub struct BitHamming;

impl Distance<u8> for BitHamming {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut result = 0u32;
        for i in 0..a.len() {
            result += (a[i] ^ b[i]).count_ones();
        }
        result as f32
    }

    fn name(&self) -> &'static str {
        "bit_hamming"
    }
}

/// Bit-packed Jaccard distance for u8 arrays.
///
/// Uses negative log transform: -ln(popcount(a&b) / popcount(a|b))
#[derive(Clone, Copy, Debug, Default)]
pub struct BitJaccard;

impl Distance<u8> for BitJaccard {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut and_count = 0u32;
        let mut or_count = 0u32;
        for i in 0..a.len() {
            and_count += (a[i] & b[i]).count_ones();
            or_count += (a[i] | b[i]).count_ones();
        }
        if or_count == 0 {
            0.0
        } else {
            -(and_count as f32 / or_count as f32).ln()
        }
    }

    fn name(&self) -> &'static str {
        "bit_jaccard"
    }
}

/// Correction function for alternative Jaccard: 1 - 2^(-d)
#[inline]
pub fn correct_alternative_jaccard(d: f32) -> f32 {
    1.0 - 2.0f32.powf(-d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];
        // 2 differences out of 4
        let d = Hamming.distance(&a, &b);
        assert!((d - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard() {
        let a = vec![1.0, 0.0, 1.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 1.0, 0.0];
        // union = 4 (indices 0,1,2,3), intersection = 2 (indices 0,3)
        // 1 - 2/4 = 0.5
        let d = Jaccard.distance(&a, &b);
        assert!((d - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dice() {
        let a = vec![1.0, 0.0, 1.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 1.0, 0.0];
        // tt=2, not_equal=2
        // 2 / (2*2 + 2) = 2/6 = 1/3
        let d = Dice.distance(&a, &b);
        assert!((d - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_yule() {
        let a = vec![1.0, 1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 1.0, 0.0];
        // tt=1, tf=1, ft=1, ff=1
        // 2*1*1/(1*1+1*1) = 2/2 = 1
        let d = Yule.distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bit_hamming() {
        let a = vec![0b11001100u8];
        let b = vec![0b11110000u8];
        // XOR = 0b00111100 → 4 bits
        let d = BitHamming.distance(&a, &b);
        assert!((d - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_bit_jaccard() {
        let a = vec![0b11111111u8];
        let b = vec![0b11111111u8];
        // AND = 8, OR = 8 → -ln(1) = 0
        let d = BitJaccard.distance(&a, &b);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_rogers_tanimoto() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];
        // not_equal = 2, n = 4
        // 2*2/(4+2) = 4/6 = 2/3
        let d = RogersTanimoto.distance(&a, &b);
        assert!((d - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_matching() {
        let a = vec![1.0, 0.0, 3.0, 0.0];
        let b = vec![0.0, 0.0, 5.0, 2.0];
        // boolean: [T,F,T,F] vs [F,F,T,T] → differ at 0,3 → 2/4 = 0.5
        let d = Matching.distance(&a, &b);
        assert!((d - 0.5).abs() < 1e-6);
    }
}
