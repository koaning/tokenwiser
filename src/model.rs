//! Sparse multilabel logistic regression.
//!
//! One independent binary logistic regressor is trained per label, all sharing
//! the same hashed feature space. Weights are stored densely as `f32` in a
//! single `n_labels * n_features` buffer, **feature-major**
//! (`weights[feature * n_labels + label]`): the innermost loops run over the
//! labels of one touched feature, so they walk contiguous memory and
//! auto-vectorize. Dense is the right choice for weights (mostly non-zero,
//! O(1) indexed dot products); the sparsity lives on the input side.
//!
//! Training is mini-batch SGD with class weights always "balanced" (per the v1
//! API) and L1/L2 regularization applied to the weights touched by each batch
//! — the standard sparse-update approximation.
//!
//! **Early stopping on tolerance.** `cfg.epochs` is a *maximum* budget; we stop
//! once the (balanced) training log-loss fails to improve by `TOL` for
//! `N_ITER_NO_CHANGE` consecutive epochs (the same idea scikit-learn's
//! `SGDClassifier` uses).

use rayon::prelude::*;

use crate::config::TrialConfig;
use crate::features::{
    gram_hashes_into, push_buckets_fused, rows_from_sorted_buckets, SparseRow,
};
use crate::rng::Rng;
use crate::tokenize::TokenBuf;

/// Minimum log-loss improvement per epoch to count as "still converging".
const TOL: f32 = 1e-4;
/// Consecutive non-improving epochs before we stop early.
const N_ITER_NO_CHANGE: usize = 2;

pub struct LogReg {
    pub n_labels: usize,
    pub n_features: usize,
    /// Feature-major weights: `weights[feature * n_labels + label]`.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// A tokenized corpus reduced to one fused `u64` hash per n-gram, CSR-style.
///
/// This is everything featurization needs that depends only on the
/// *tokenization* knobs (tokenizer, lowercase, ngram range). Deriving a
/// trial's `SparseRow`s from it is pure integer work — `n_features`,
/// `n_hashes` and `feature_value` are applied by remixing the cached hashes —
/// so a randomized search tokenizes the corpus once per distinct tokenization
/// signature instead of once per trial.
pub struct TokenCache {
    /// Fused gram hashes, all rows concatenated.
    pub hashes: Vec<u64>,
    /// Row boundaries into `hashes` (`n_rows + 1` entries).
    pub offsets: Vec<usize>,
}

impl TokenCache {
    /// Tokenize the whole corpus (in parallel) under one tokenization config.
    pub fn build(
        texts: &[String],
        tokenizer: &str,
        lowercase: bool,
        ngram_min: u32,
        ngram_max: u32,
    ) -> TokenCache {
        // Tokenize in parallel per bounded chunk, flattening as we go: the
        // transient per-row Vecs only ever cover one chunk, so peak memory is
        // the flat cache plus O(chunk), not 2x the corpus.
        const CHUNK: usize = 8_192;
        let mut offsets = Vec::with_capacity(texts.len() + 1);
        offsets.push(0);
        let mut hashes: Vec<u64> = Vec::new();
        for chunk in texts.chunks(CHUNK) {
            let per_row: Vec<Vec<u64>> = chunk
                .par_iter()
                .map_init(TokenBuf::default, |buf, text| {
                    buf.fill(text, tokenizer, lowercase);
                    let mut row = Vec::with_capacity(buf.n_tokens());
                    gram_hashes_into(buf, ngram_min, ngram_max, &mut row);
                    row
                })
                .collect();
            for row in per_row {
                hashes.extend_from_slice(&row);
                offsets.push(hashes.len());
            }
        }
        TokenCache { hashes, offsets }
    }

    /// Number of cached rows.
    pub fn n_rows(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Derive one trial's sparse rows: remix the cached gram hashes into
    /// bucket ids (in parallel), no string work at all.
    pub fn rows(&self, n_features: usize, n_hashes: usize, binary: bool) -> Vec<SparseRow> {
        (0..self.n_rows())
            .into_par_iter()
            .map_init(Vec::new, |scratch: &mut Vec<u32>, i| {
                scratch.clear();
                for &h in &self.hashes[self.offsets[i]..self.offsets[i + 1]] {
                    push_buckets_fused(h, n_features, n_hashes, scratch);
                }
                rows_from_sorted_buckets(scratch, binary)
            })
            .collect()
    }
}

/// Build sparse feature rows directly from raw texts under `cfg`.
///
/// Tokenization is transient per row (never cached), so this holds only the
/// compact integer feature rows, not the corpus as strings. For repeated
/// derivations under one tokenization config, prefer [`TokenCache`].
/// New training always uses the fused (v3) bucket scheme.
pub fn build_rows(texts: &[String], cfg: &TrialConfig) -> Vec<SparseRow> {
    let binary = cfg.feature_value == "binary";
    // Parallel within bounded chunks (sequential across them) to keep the
    // allocator's high-water mark close to the final rows' footprint.
    const CHUNK: usize = 8_192;
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(CHUNK) {
        out.par_extend(chunk.par_iter().map_init(
            || (TokenBuf::default(), Vec::new(), Vec::new()),
            |(buf, hashes, scratch): &mut (TokenBuf, Vec<u64>, Vec<u32>), text| {
                buf.fill(text, &cfg.tokenizer, cfg.lowercase);
                hashes.clear();
                gram_hashes_into(buf, cfg.ngram_min, cfg.ngram_max, hashes);
                scratch.clear();
                for &h in hashes.iter() {
                    push_buckets_fused(h, cfg.n_features, cfg.n_hashes, scratch);
                }
                rows_from_sorted_buckets(scratch, binary)
            },
        ));
    }
    out
}

impl LogReg {
    /// Train on a subset of rows (referenced by `idxs`).
    ///
    /// `rows` and `labels` are indexed by the original row id; `idxs` selects
    /// the training rows for this fit.
    pub fn train(
        rows: &[SparseRow],
        labels: &[Vec<u32>],
        idxs: &[usize],
        n_labels: usize,
        n_features: usize,
        cfg: &TrialConfig,
        batch_size: usize,
        seed: u64,
    ) -> LogReg {
        let n = idxs.len();
        let mut weights = vec![0.0f32; n_labels * n_features];
        let mut bias = vec![0.0f32; n_labels];

        // Balanced class weights per label, computed on the training subset:
        //   w_pos = N / (2 * n_pos), w_neg = N / (2 * n_neg).
        let mut pos_count = vec![0usize; n_labels];
        for &i in idxs {
            for &l in &labels[i] {
                pos_count[l as usize] += 1;
            }
        }
        let mut w_pos = vec![0.0f32; n_labels];
        let mut w_neg = vec![0.0f32; n_labels];
        for l in 0..n_labels {
            let np = pos_count[l];
            let nn = n - np;
            w_pos[l] = if np > 0 {
                (n as f32) / (2.0 * np as f32)
            } else {
                0.0
            };
            w_neg[l] = if nn > 0 {
                (n as f32) / (2.0 * nn as f32)
            } else {
                0.0
            };
        }

        let lr = cfg.learning_rate as f32;
        let l1 = cfg.l1 as f32;
        let l2 = cfg.l2 as f32;
        let bs = batch_size.max(1);

        // Reusable per-batch gradient accumulators.
        let mut grad = vec![0.0f32; n_labels * n_features];
        let mut grad_bias = vec![0.0f32; n_labels];
        let mut order: Vec<usize> = idxs.to_vec();

        // Reusable per-sample buffers, hoisted out of the hot loops: the
        // per-label scores, a label-membership mask (replacing a
        // `binary_search` per (sample, label)), and the per-label gradient
        // coefficients for the feature-major scatter below.
        let mut z = vec![0.0f32; n_labels];
        let mut is_pos = vec![false; n_labels];
        let mut coef = vec![0.0f32; n_labels];
        let mut touched: Vec<u32> = Vec::new();

        let mut rng = Rng::new(seed);
        let mut best_loss = f32::INFINITY;
        let mut no_improve = 0usize;
        for _epoch in 0..cfg.epochs {
            rng.shuffle(&mut order);
            let mut epoch_loss = 0.0f32;
            let mut start = 0;
            while start < n {
                let end = (start + bs).min(n);
                let batch = &order[start..end];
                let inv_bs = 1.0 / batch.len() as f32;

                // Track features touched this batch so we only reset those.
                touched.clear();

                for &i in batch {
                    let row = &rows[i];
                    // Scores for all labels: z[l] = bias[l] + w[l]·x. The
                    // label loop reads `n_labels` contiguous floats per
                    // touched feature (feature-major layout).
                    z.copy_from_slice(&bias);
                    for (&idx, &v) in row.indices.iter().zip(&row.values) {
                        let base = idx as usize * n_labels;
                        for l in 0..n_labels {
                            z[l] += weights[base + l] * v;
                        }
                    }
                    // Per-label gradient coefficient: sw * (p - y).
                    let label_set = &labels[i];
                    for &l in label_set {
                        is_pos[l as usize] = true;
                    }
                    for l in 0..n_labels {
                        let pos = is_pos[l];
                        let y = if pos { 1.0 } else { 0.0 };
                        let p = sigmoid(z[l]);
                        let sw = if pos { w_pos[l] } else { w_neg[l] };
                        coef[l] = sw * (p - y);
                        // Balanced log-loss for the convergence check (uses the
                        // probability from the current forward pass). One ln()
                        // suffices: the other term is always multiplied by 0.
                        let pc = p.clamp(1e-7, 1.0 - 1e-7);
                        epoch_loss -= sw * if pos { pc.ln() } else { (1.0 - pc).ln() };
                        grad_bias[l] += coef[l];
                    }
                    // Feature-major gradient scatter: contiguous writes of all
                    // labels' contributions per touched feature.
                    for (&idx, &v) in row.indices.iter().zip(&row.values) {
                        let base = idx as usize * n_labels;
                        for l in 0..n_labels {
                            grad[base + l] += coef[l] * v;
                        }
                    }
                    for &l in label_set {
                        is_pos[l as usize] = false;
                    }
                    for &idx in &row.indices {
                        touched.push(idx);
                    }
                }

                touched.sort_unstable();
                touched.dedup();

                // Apply weight updates (data gradient + regularization) only to
                // touched features, then reset their gradient slots. Each
                // touched feature updates `n_labels` contiguous weights.
                for &idx in &touched {
                    let base = idx as usize * n_labels;
                    for l in 0..n_labels {
                        let wi = base + l;
                        let w = weights[wi];
                        let g = grad[wi] * inv_bs + l2 * w + l1 * w.signum();
                        weights[wi] = w - lr * g;
                        grad[wi] = 0.0;
                    }
                }
                for l in 0..n_labels {
                    bias[l] -= lr * grad_bias[l] * inv_bs;
                    grad_bias[l] = 0.0;
                }

                start = end;
            }

            // Early stop: halt once the loss stops improving by at least TOL.
            epoch_loss /= (n * n_labels.max(1)) as f32;
            if best_loss - epoch_loss < TOL {
                no_improve += 1;
                if no_improve >= N_ITER_NO_CHANGE {
                    break;
                }
            } else {
                no_improve = 0;
            }
            if epoch_loss < best_loss {
                best_loss = epoch_loss;
            }
        }

        LogReg {
            n_labels,
            n_features,
            weights,
            bias,
        }
    }

    /// Probabilities for one sparse row (length `n_labels`).
    pub fn proba_row(&self, row: &SparseRow) -> Vec<f32> {
        let n_labels = self.n_labels;
        let mut z = self.bias.clone();
        for (&idx, &v) in row.indices.iter().zip(&row.values) {
            let base = idx as usize * n_labels;
            for l in 0..n_labels {
                z[l] += self.weights[base + l] * v;
            }
        }
        for p in z.iter_mut() {
            *p = sigmoid(*p);
        }
        z
    }
}
