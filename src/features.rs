//! Bloom-style hashing features.
//!
//! Each n-gram is hashed with `n_hashes` independent FNV-1a hashes into a
//! shared table of `n_features` buckets. With more than one hash the feature
//! behaves like a Bloom filter, spreading each term across several buckets to
//! reduce the impact of individual collisions.
//!
//! * `feature_value = "binary"` — every touched bucket has value `1.0`.
//! * `feature_value = "count"`  — buckets accumulate occurrence counts.

use std::collections::HashMap;

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// FNV-1a hash of `bytes` mixed with a per-hash seed.
#[inline]
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = FNV_OFFSET ^ seed.wrapping_mul(FNV_PRIME);
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A sparse feature row: sorted unique indices with their values.
///
/// NOTE on storage width (deliberately not optimized yet):
/// - `values` are `f32` for simplicity, but are never truly fractional — in
///   `binary` mode every value is `1.0` (the array could be dropped entirely),
///   and in `count` mode they are small non-negative integers (`u16`, or `u8`
///   with a saturating add, would suffice). `index` is `u32` because
///   `n_features` can exceed `u16`'s 64k ceiling (the default tops at 100k).
/// - Payback: at ~200k rows the feature payload is ~74 MB of a ~157 MB peak;
///   dropping binary values + a CSR layout would trim it to ~115 MB (~27%).
///   Below ~25k rows it is <8 MB — noise. So this only matters at 100k+ rows
///   and is left as a future optimization.
pub struct SparseRow {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Hash a document's n-grams into a sparse row.
pub fn hash_row(grams: &[String], n_features: usize, n_hashes: usize, binary: bool) -> SparseRow {
    let mut acc: HashMap<u32, f32> = HashMap::new();
    for gram in grams {
        let bytes = gram.as_bytes();
        for h in 0..n_hashes {
            // Distinct, fixed seeds per hash function keep this reproducible.
            let seed = (h as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let idx = (fnv1a(bytes, seed) % n_features as u64) as u32;
            if binary {
                acc.insert(idx, 1.0);
            } else {
                *acc.entry(idx).or_insert(0.0) += 1.0;
            }
        }
    }
    let mut indices: Vec<u32> = acc.keys().copied().collect();
    indices.sort_unstable();
    let values: Vec<f32> = indices.iter().map(|i| acc[i]).collect();
    SparseRow { indices, values }
}
