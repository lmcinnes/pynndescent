//! Minkowski-family distances: Manhattan, Chebyshev, Minkowski, Canberra, Bray-Curtis.

use super::traits::Distance;

/// Manhattan (L1 / taxicab) distance.
///
/// Formula: d(a, b) = Σ|aᵢ - bᵢ|
#[derive(Clone, Copy, Debug, Default)]
pub struct Manhattan;

impl Distance<f32> for Manhattan {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { manhattan_avx2(a, b) };
            }
        }

        scalar_manhattan(a, b)
    }

    fn name(&self) -> &'static str {
        "manhattan"
    }
}

#[inline]
fn scalar_manhattan(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let idx = i * 4;
        sum += (a[idx] - b[idx]).abs()
            + (a[idx + 1] - b[idx + 1]).abs()
            + (a[idx + 2] - b[idx + 2]).abs()
            + (a[idx + 3] - b[idx + 3]).abs();
    }

    let start = chunks * 4;
    for i in 0..remainder {
        sum += (a[start + i] - b[start + i]).abs();
    }

    sum
}

/// AVX2 Manhattan distance: abs(diff) via clearing sign bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn manhattan_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    // Sign mask: all bits except bit 31 set → AND clears sign bit
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
    let mut vsum = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let aa = _mm256_loadu_ps(a.as_ptr().add(idx));
        let bb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm256_sub_ps(aa, bb);
        let abs_diff = _mm256_and_ps(diff, sign_mask);
        vsum = _mm256_add_ps(vsum, abs_diff);
    }

    let mut result = hsum256_ps(vsum);

    let start = chunks * 8;
    for i in start..n {
        result += (a[i] - b[i]).abs();
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

/// Chebyshev (L∞) distance.
///
/// Formula: d(a, b) = max_i |aᵢ - bᵢ|
#[derive(Clone, Copy, Debug, Default)]
pub struct Chebyshev;

impl Distance<f32> for Chebyshev {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut result = 0.0f32;
        for i in 0..a.len() {
            let d = (a[i] - b[i]).abs();
            if d > result {
                result = d;
            }
        }
        result
    }

    fn name(&self) -> &'static str {
        "chebyshev"
    }
}

/// Minkowski (Lp) distance.
///
/// Formula: d(a, b) = (Σ|aᵢ - bᵢ|^p)^(1/p)
///
/// Stores the `p` parameter. For p=1 use Manhattan, for p=2 use Euclidean.
#[derive(Clone, Copy, Debug)]
pub struct Minkowski {
    pub p: f32,
}

impl Minkowski {
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl Distance<f32> for Minkowski {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += (a[i] - b[i]).abs().powf(self.p);
        }
        sum.powf(1.0 / self.p)
    }

    fn name(&self) -> &'static str {
        "minkowski"
    }
}

/// Canberra distance.
///
/// Formula: d(a, b) = Σ |aᵢ - bᵢ| / (|aᵢ| + |bᵢ|)
#[derive(Clone, Copy, Debug, Default)]
pub struct Canberra;

impl Distance<f32> for Canberra {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut sum = 0.0f32;
        for i in 0..a.len() {
            let denom = a[i].abs() + b[i].abs();
            if denom > 0.0 {
                sum += (a[i] - b[i]).abs() / denom;
            }
        }
        sum
    }

    fn name(&self) -> &'static str {
        "canberra"
    }
}

/// Bray-Curtis distance.
///
/// Formula: d(a, b) = Σ|aᵢ - bᵢ| / Σ|aᵢ + bᵢ|
#[derive(Clone, Copy, Debug, Default)]
pub struct BrayCurtis;

impl Distance<f32> for BrayCurtis {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut numer = 0.0f32;
        let mut denom = 0.0f32;
        for i in 0..a.len() {
            numer += (a[i] - b[i]).abs();
            denom += (a[i] + b[i]).abs();
        }
        if denom > 0.0 {
            numer / denom
        } else {
            0.0
        }
    }

    fn name(&self) -> &'static str {
        "braycurtis"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manhattan_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 0.0, -1.0];
        // |1-4| + |2-0| + |3-(-1)| = 3 + 2 + 4 = 9
        let d = Manhattan.distance(&a, &b);
        assert!((d - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_chebyshev_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 0.0, -1.0];
        // max(3, 2, 4) = 4
        let d = Chebyshev.distance(&a, &b);
        assert!((d - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_minkowski_p2_is_euclidean() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = Minkowski::new(2.0).distance(&a, &b);
        assert!((d - std::f32::consts::SQRT_2).abs() < 1e-5);
    }

    #[test]
    fn test_minkowski_p1_is_manhattan() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 0.0, -1.0];
        let d = Minkowski::new(1.0).distance(&a, &b);
        assert!((d - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_canberra_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        // |1-2|/(1+2) + |2-4|/(2+4) + |3-6|/(3+6) = 1/3 + 2/6 + 3/9 = 1/3 + 1/3 + 1/3 = 1
        let d = Canberra.distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_braycurtis_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        // (1+2+3)/(3+6+9) = 6/18 = 1/3
        let d = BrayCurtis.distance(&a, &b);
        assert!((d - 1.0 / 3.0).abs() < 1e-6);
    }
}
