//! Bloom-style hashing features.
//!
//! Each n-gram is spread across `n_hashes` buckets of a shared table of
//! `n_features` buckets. With more than one hash the feature behaves like a
//! Bloom filter, reducing the impact of individual collisions.
//!
//! * `feature_value = "binary"` — every touched bucket has value `1.0`.
//! * `feature_value = "count"`  — buckets accumulate occurrence counts.
//!
//! Two bucket schemes exist, selected by the model artifact version:
//!
//! * **Fused** (v3+, used for all new training): one FNV-1a pass per gram,
//!   then Kirsch–Mitzenmacher double hashing derives the k bucket ids as
//!   `h1 + i*h2` — statistically equivalent to k independent hashes at one
//!   byte-pass of cost, and it reduces each gram to a single cacheable `u64`.
//! * **Legacy** (v2 artifacts): `n_hashes` full FNV-1a passes with distinct
//!   seeds. Kept so old artifacts predict identically forever.

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// FNV-1a hash of `bytes` mixed with a per-hash seed (legacy scheme).
#[inline]
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = FNV_OFFSET ^ seed.wrapping_mul(FNV_PRIME);
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Single seedless FNV-1a pass (fused scheme).
#[inline]
fn fnv1a_plain(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// SplitMix64 finalizer: derives the second (stride) hash from the first so
/// Kirsch–Mitzenmacher double hashing needs only one byte pass per gram.
#[inline]
fn mix(mut h: u64) -> u64 {
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D049BB133111EB);
    h ^ (h >> 31)
}

/// Push the `n_hashes` bucket ids for one already-hashed gram (fused scheme).
#[inline]
pub fn push_buckets_fused(gram_hash: u64, n_features: usize, n_hashes: usize, out: &mut Vec<u32>) {
    let h1 = gram_hash;
    let h2 = mix(gram_hash);
    for i in 0..n_hashes {
        let idx = h1.wrapping_add((i as u64).wrapping_mul(h2)) % n_features as u64;
        out.push(idx as u32);
    }
}

/// One FNV-1a pass over a gram's bytes (fused scheme's cacheable per-gram key).
#[inline]
pub fn gram_hash(bytes: &[u8]) -> u64 {
    fnv1a_plain(bytes)
}

/// Incremental FNV-1a: fold more bytes into a running hash. Feeding a gram's
/// tokens separated by a `b' '` byte yields exactly `gram_hash` of the joined
/// string, so n-gram hashes never need the joined `String`.
#[inline]
pub fn fnv_init() -> u64 {
    FNV_OFFSET
}

#[inline]
pub fn fnv_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Emit the fused gram hash of every word n-gram (n in `min..=max`) of a
/// tokenized document, without materializing gram strings.
pub fn gram_hashes_into(buf: &crate::tokenize::TokenBuf, min: u32, max: u32, out: &mut Vec<u64>) {
    let n_tokens = buf.n_tokens();
    let min = min.max(1) as usize;
    let max = (max.max(min as u32)) as usize;
    for n in min..=max {
        if n_tokens < n {
            continue;
        }
        for start in 0..=(n_tokens - n) {
            let mut h = fnv_init();
            for j in start..start + n {
                if j > start {
                    h = fnv_update(h, b" ");
                }
                h = fnv_update(h, buf.token_bytes(j));
            }
            out.push(h);
        }
    }
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
///
/// `fused` selects the bucket scheme: true for v3+ (Kirsch–Mitzenmacher),
/// false for legacy v2 artifacts.
pub fn hash_row(
    grams: &[String],
    n_features: usize,
    n_hashes: usize,
    binary: bool,
    fused: bool,
) -> SparseRow {
    let mut scratch = Vec::new();
    hash_row_with_scratch(grams, n_features, n_hashes, binary, fused, &mut scratch)
}

/// Like [`hash_row`], but reuses a caller-provided scratch buffer so bulk
/// featurization (one call per document) does no transient allocation.
pub fn hash_row_with_scratch(
    grams: &[String],
    n_features: usize,
    n_hashes: usize,
    binary: bool,
    fused: bool,
    scratch: &mut Vec<u32>,
) -> SparseRow {
    // Collect every touched bucket id, then sort + run-length dedup. This
    // beats a HashMap here: rows are small, and sort/dedup on a flat Vec is
    // allocation-free and branch-predictable. Counts via run length are
    // bit-identical to repeated `+= 1.0` (exact in f32 far past any real
    // document length).
    scratch.clear();
    for gram in grams {
        let bytes = gram.as_bytes();
        if fused {
            push_buckets_fused(gram_hash(bytes), n_features, n_hashes, scratch);
        } else {
            for h in 0..n_hashes {
                // Distinct, fixed seeds per hash function keep this reproducible.
                let seed = (h as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
                scratch.push((fnv1a(bytes, seed) % n_features as u64) as u32);
            }
        }
    }
    rows_from_sorted_buckets(scratch, binary)
}

/// Sort + run-length dedup a scratch bucket list into a `SparseRow`.
pub fn rows_from_sorted_buckets(scratch: &mut Vec<u32>, binary: bool) -> SparseRow {
    scratch.sort_unstable();
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    let mut i = 0;
    while i < scratch.len() {
        let idx = scratch[i];
        let mut run = 1usize;
        while i + run < scratch.len() && scratch[i + run] == idx {
            run += 1;
        }
        indices.push(idx);
        values.push(if binary { 1.0 } else { run as f32 });
        i += run;
    }
    SparseRow { indices, values }
}
