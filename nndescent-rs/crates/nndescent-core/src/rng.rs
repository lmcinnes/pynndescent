//! Fast pseudo-random number generators.
//!
//! Provides both:
//! - `FastRng`: High-performance Xoshiro256++ based RNG (recommended)
//! - `TauRand`: PyNNDescent-compatible RNG for reproducibility

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// High-performance thread-safe RNG using Xoshiro256++.
///
/// This is significantly faster than TauRand and suitable for parallel code.
/// Use this for new code; use TauRand only when exact PyNNDescent reproducibility is needed.
#[derive(Clone, Debug)]
pub struct FastRng {
    inner: Xoshiro256PlusPlus,
}

impl FastRng {
    /// Create a new FastRng from a seed.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            inner: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// Generate a random u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.inner.gen()
    }

    /// Generate a random i64.
    #[inline]
    pub fn next_int(&mut self) -> i64 {
        self.inner.gen::<i64>()
    }

    /// Generate a random float in [0, 1).
    #[inline]
    pub fn next_float(&mut self) -> f32 {
        self.inner.gen()
    }

    /// Generate a random index in [0, n).
    #[inline]
    pub fn next_index(&mut self, n: usize) -> usize {
        self.inner.gen_range(0..n)
    }

    /// Generate a random boolean.
    #[inline]
    pub fn next_bool(&mut self) -> bool {
        self.inner.gen()
    }

    /// Jump the RNG state forward (useful for parallel streams).
    #[inline]
    pub fn jump(&mut self) {
        self.inner.jump();
    }

    /// Create a new independent RNG derived from this one.
    #[inline]
    pub fn fork(&mut self) -> Self {
        let seed = self.next_u64();
        Self::new(seed)
    }

    /// Offset the RNG by consuming n values.
    /// Useful for creating deterministically different streams.
    #[inline]
    pub fn offset(mut self, n: u64) -> Self {
        for _ in 0..n {
            self.next_u64();
        }
        self
    }
}

impl Default for FastRng {
    fn default() -> Self {
        Self::new(42)
    }
}

/// Fast pseudo-random number generator using xorshift.
///
/// This matches PyNNDescent's `tau_rand_int` function exactly for reproducibility.
/// For new code, prefer `FastRng` which is faster and thread-safe.
#[derive(Clone, Debug)]
pub struct TauRand {
    state: [i64; 3],
}

impl TauRand {
    /// Create a new TauRand from a seed.
    pub fn new(seed: u64) -> Self {
        // Initialize state similar to PyNNDescent
        let seed = seed as i64;
        Self {
            state: [
                seed.wrapping_mul(1103515245).wrapping_add(12345),
                seed.wrapping_mul(1103515245).wrapping_add(12345).wrapping_mul(1103515245),
                seed.wrapping_mul(1103515245).wrapping_add(12345).wrapping_mul(1103515245).wrapping_mul(1103515245),
            ],
        }
    }

    /// Create from explicit state (for matching PyNNDescent exactly).
    pub fn from_state(state: [i64; 3]) -> Self {
        Self { state }
    }

    /// Get the current state (for serialization).
    pub fn state(&self) -> [i64; 3] {
        self.state
    }

    /// Generate a random i64, matching PyNNDescent's tau_rand_int.
    #[inline]
    pub fn next_int(&mut self) -> i64 {
        const MASK32: i64 = 0xFFFFFFFF;

        self.state[0] = (((self.state[0] & 4294967294) << 12) & MASK32)
            ^ ((((self.state[0] << 13) & MASK32) ^ self.state[0]) >> 19);

        self.state[1] = (((self.state[1] & 4294967288) << 4) & MASK32)
            ^ ((((self.state[1] << 2) & MASK32) ^ self.state[1]) >> 25);

        self.state[2] = (((self.state[2] & 4294967280) << 17) & MASK32)
            ^ ((((self.state[2] << 3) & MASK32) ^ self.state[2]) >> 11);

        self.state[0] ^ self.state[1] ^ self.state[2]
    }

    /// Generate a random float in [0, 1).
    #[inline]
    pub fn next_float(&mut self) -> f32 {
        (self.next_int().abs() as f32) / (i64::MAX as f32)
    }

    /// Generate a random index in [0, n).
    #[inline]
    pub fn next_index(&mut self, n: usize) -> usize {
        (self.next_int().abs() as usize) % n
    }

    /// Generate a random boolean.
    #[inline]
    pub fn next_bool(&mut self) -> bool {
        (self.next_int() & 1) != 0
    }

    /// Warm up the RNG state (recommended before use).
    pub fn warm_up(&mut self, iterations: usize) {
        for _ in 0..iterations {
            let _ = self.next_int();
        }
    }
}

impl Default for TauRand {
    fn default() -> Self {
        Self::new(42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tau_rand_deterministic() {
        let mut rng1 = TauRand::new(12345);
        let mut rng2 = TauRand::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_int(), rng2.next_int());
        }
    }

    #[test]
    fn test_tau_rand_different_seeds() {
        let mut rng1 = TauRand::new(12345);
        let mut rng2 = TauRand::new(54321);

        // Should produce different sequences
        let seq1: Vec<i64> = (0..10).map(|_| rng1.next_int()).collect();
        let seq2: Vec<i64> = (0..10).map(|_| rng2.next_int()).collect();

        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_next_index_bounds() {
        let mut rng = TauRand::new(42);
        
        for _ in 0..1000 {
            let idx = rng.next_index(100);
            assert!(idx < 100);
        }
    }

    #[test]
    fn test_next_float_bounds() {
        let mut rng = TauRand::new(42);
        
        for _ in 0..1000 {
            let f = rng.next_float();
            assert!(f >= 0.0 && f < 1.0);
        }
    }
}
