//! Deterministic, dependency-free PRNG (SplitMix64).
//!
//! Using our own generator keeps randomized search reproducible across
//! machines and thread counts: every random decision is derived from an
//! explicit `u64` seed.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    /// Derive a fresh, independent generator from a parent seed and a key.
    /// Used to give each trial/fold its own reproducible stream.
    pub fn derive(seed: u64, key: u64) -> Self {
        let mut s = Rng::new(seed ^ key.wrapping_mul(0x9E3779B97F4A7C15));
        // burn one value so nearby keys diverge quickly
        s.next_u64();
        s
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform float in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Uniform integer in [lo, hi] inclusive.
    pub fn int_inclusive(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        let span = (hi - lo) as u64 + 1;
        lo + (self.next_u64() % span) as i64
    }

    /// Uniform index in [0, n).
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Log-uniform float in [lo, hi].
    pub fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let (a, b) = (lo.ln(), hi.ln());
        (a + (b - a) * self.next_f64()).exp()
    }

    /// In-place Fisher-Yates shuffle.
    pub fn shuffle(&mut self, items: &mut [usize]) {
        let n = items.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}
