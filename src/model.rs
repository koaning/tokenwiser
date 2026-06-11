//! Sparse multilabel logistic regression.
//!
//! One independent binary logistic regressor is trained per label, all sharing
//! the same hashed feature space. Weights are stored densely as `f32` in a
//! single `n_labels * n_features` buffer (row-major by label) — dense is the
//! right choice for weights (mostly non-zero, O(1) indexed dot products); the
//! sparsity lives on the input side.
//!
//! Training is mini-batch SGD with class weights always "balanced" (per the v1
//! API) and L1/L2 regularization applied to the weights touched by each batch
//! — the standard sparse-update approximation.
//!
//! **Early stopping on tolerance.** `cfg.epochs` is a *maximum* budget; we stop
//! once the (balanced) training log-loss fails to improve by `TOL` for
//! `N_ITER_NO_CHANGE` consecutive epochs (the same idea scikit-learn's
//! `SGDClassifier` uses).

use crate::config::TrialConfig;
use crate::features::{hash_row, SparseRow};
use crate::rng::Rng;
use crate::tokenize::{base_tokens, ngrams};

/// Minimum log-loss improvement per epoch to count as "still converging".
const TOL: f32 = 1e-4;
/// Consecutive non-improving epochs before we stop early.
const N_ITER_NO_CHANGE: usize = 2;

pub struct LogReg {
    pub n_labels: usize,
    pub n_features: usize,
    /// Row-major weights: `weights[label * n_features + feature]`.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Build sparse feature rows directly from raw texts under `cfg`.
///
/// Tokenization is transient per row (never cached), so this holds only the
/// compact integer feature rows, not the corpus as strings.
pub fn build_rows(texts: &[String], cfg: &TrialConfig) -> Vec<SparseRow> {
    let binary = cfg.feature_value == "binary";
    texts
        .iter()
        .map(|text| {
            let tokens = base_tokens(text, &cfg.tokenizer, cfg.lowercase);
            let grams = ngrams(&tokens, cfg.ngram_min, cfg.ngram_max);
            hash_row(&grams, cfg.n_features, cfg.n_hashes, binary)
        })
        .collect()
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
                let mut touched: Vec<u32> = Vec::new();

                for &i in batch {
                    let row = &rows[i];
                    // Scores for all labels: z[l] = bias[l] + w[l]·x.
                    let mut z = bias.clone();
                    for (k, &idx) in row.indices.iter().enumerate() {
                        let v = row.values[k];
                        let fi = idx as usize;
                        for l in 0..n_labels {
                            z[l] += weights[l * n_features + fi] * v;
                        }
                    }
                    // Per-label gradient coefficient: sw * (p - y).
                    let label_set = &labels[i];
                    for l in 0..n_labels {
                        let y = if label_set.binary_search(&(l as u32)).is_ok() {
                            1.0
                        } else {
                            0.0
                        };
                        let p = sigmoid(z[l]);
                        let sw = if y > 0.5 { w_pos[l] } else { w_neg[l] };
                        let coef = sw * (p - y);
                        // Balanced log-loss for the convergence check (uses the
                        // probability from the current forward pass).
                        let pc = p.clamp(1e-7, 1.0 - 1e-7);
                        epoch_loss -= sw * (y * pc.ln() + (1.0 - y) * (1.0 - pc).ln());
                        grad_bias[l] += coef;
                        let base = l * n_features;
                        for (k, &idx) in row.indices.iter().enumerate() {
                            grad[base + idx as usize] += coef * row.values[k];
                        }
                    }
                    for &idx in &row.indices {
                        touched.push(idx);
                    }
                }

                touched.sort_unstable();
                touched.dedup();

                // Apply weight updates (data gradient + regularization) only to
                // touched features, then reset their gradient slots.
                for l in 0..n_labels {
                    let base = l * n_features;
                    for &idx in &touched {
                        let wi = base + idx as usize;
                        let w = weights[wi];
                        let g = grad[wi] * inv_bs + l2 * w + l1 * w.signum();
                        weights[wi] = w - lr * g;
                        grad[wi] = 0.0;
                    }
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
        let mut z = self.bias.clone();
        for (k, &idx) in row.indices.iter().enumerate() {
            let v = row.values[k];
            let fi = idx as usize;
            for l in 0..self.n_labels {
                z[l] += self.weights[l * self.n_features + fi] * v;
            }
        }
        z.iter().map(|&v| sigmoid(v)).collect()
    }
}
