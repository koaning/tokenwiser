//! Cross-validated randomized search.
//!
//! For each trial a configuration is sampled (seeded, reproducible), evaluated
//! with k-fold CV on macro-F1, and the best configuration is retrained on all
//! rows to produce the final model. A [`Report`] captures every trial.
//!
//! v1 runs single-threaded so results are bit-for-bit reproducible regardless
//! of the host's core count.

use std::time::Instant;

use serde::Serialize;

use crate::config::{SearchSpace, TrialConfig};
use crate::dataset::Dataset;
use crate::features::SparseRow;
use crate::model::{build_rows, LogReg};
use crate::rng::Rng;
use crate::{artifact::Model, config};

#[derive(Serialize, Clone)]
pub struct ConfigJson {
    pub tokenizer: String,
    pub lowercase: bool,
    pub ngram_range: (u32, u32),
    pub n_features: usize,
    pub n_hashes: usize,
    pub feature_value: String,
    pub epochs: usize,
    pub learning_rate: f64,
    pub l1: f64,
    pub l2: f64,
}

impl ConfigJson {
    fn from(cfg: &TrialConfig) -> ConfigJson {
        ConfigJson {
            tokenizer: cfg.tokenizer.clone(),
            lowercase: cfg.lowercase,
            ngram_range: (cfg.ngram_min, cfg.ngram_max),
            n_features: cfg.n_features,
            n_hashes: cfg.n_hashes,
            feature_value: cfg.feature_value.clone(),
            epochs: cfg.epochs,
            learning_rate: cfg.learning_rate,
            l1: cfg.l1,
            l2: cfg.l2,
        }
    }
}

#[derive(Serialize)]
pub struct TrialReport {
    pub trial: usize,
    pub config: ConfigJson,
    pub fold_scores: Vec<f64>,
    pub mean_macro_f1: f64,
}

#[derive(Serialize)]
pub struct Report {
    pub n_rows: usize,
    pub n_labels: usize,
    pub labels: Vec<String>,
    pub label_support: Vec<usize>,
    pub cv: usize,
    pub trials: usize,
    pub seed: u64,
    pub batch_size: usize,
    pub metric: String,
    pub trials_detail: Vec<TrialReport>,
    pub best_trial: usize,
    pub best_config: ConfigJson,
    pub best_macro_f1: f64,
    pub train_seconds: f64,
}

/// Macro-averaged F1 over the evaluation rows, at the fixed 0.5 threshold.
fn macro_f1(
    model: &LogReg,
    rows: &[SparseRow],
    labels: &[Vec<u32>],
    eval_idxs: &[usize],
    n_labels: usize,
) -> f64 {
    let mut tp = vec![0u64; n_labels];
    let mut fp = vec![0u64; n_labels];
    let mut fn_ = vec![0u64; n_labels];

    for &i in eval_idxs {
        let probs = model.proba_row(&rows[i]);
        let truth = &labels[i];
        for l in 0..n_labels {
            let predicted = probs[l] >= 0.5;
            let actual = truth.binary_search(&(l as u32)).is_ok();
            match (predicted, actual) {
                (true, true) => tp[l] += 1,
                (true, false) => fp[l] += 1,
                (false, true) => fn_[l] += 1,
                (false, false) => {}
            }
        }
    }

    let mut sum = 0.0;
    for l in 0..n_labels {
        let denom_p = tp[l] + fp[l];
        let denom_r = tp[l] + fn_[l];
        let precision = if denom_p > 0 {
            tp[l] as f64 / denom_p as f64
        } else {
            0.0
        };
        let recall = if denom_r > 0 {
            tp[l] as f64 / denom_r as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        sum += f1;
    }
    if n_labels == 0 {
        0.0
    } else {
        sum / n_labels as f64
    }
}

/// Assign each row to one of `cv` folds (round-robin over a seeded shuffle, so
/// folds stay balanced in size and reproducible).
fn make_folds(n: usize, cv: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    Rng::new(seed).shuffle(&mut order);
    let mut fold_of = vec![0usize; n];
    for (pos, &row) in order.iter().enumerate() {
        fold_of[row] = pos % cv;
    }
    fold_of
}

/// Run the full search and return the final model plus the report.
pub fn run(space: &SearchSpace, mut data: Dataset) -> Result<(Model, Report), String> {
    let start = Instant::now();
    let n = data.n_rows();
    let n_labels = data.n_labels();
    let cv = space.cv.max(1);
    let folds = make_folds(n, cv, space.seed);

    let mut trials_detail: Vec<TrialReport> = Vec::with_capacity(space.trials);
    let mut best: Option<(usize, f64, TrialConfig)> = None;

    for t in 0..space.trials {
        let mut rng = Rng::derive(space.seed, t as u64 + 1);
        let cfg = space.sample(&mut rng);

        // Base tokens cached per (tokenizer, lowercase); build hashed rows once
        // for the whole dataset and reuse them across folds.
        let base = data.base(&cfg.tokenizer, cfg.lowercase).to_vec();
        let rows = build_rows(&base, &cfg);

        let mut fold_scores = Vec::with_capacity(cv);
        for f in 0..cv {
            let train_idx: Vec<usize> = (0..n).filter(|&i| folds[i] != f).collect();
            let val_idx: Vec<usize> = (0..n).filter(|&i| folds[i] == f).collect();
            if train_idx.is_empty() || val_idx.is_empty() {
                continue;
            }
            // Fold fit seed is reproducible from the trial/fold position.
            let fit_seed = space
                .seed
                .wrapping_add((t as u64 + 1).wrapping_mul(1009))
                .wrapping_add((f as u64 + 1).wrapping_mul(9176));
            let fit = LogReg::train(
                &rows,
                &data.labels,
                &train_idx,
                n_labels,
                cfg.n_features,
                &cfg,
                space.batch_size,
                fit_seed,
            );
            fold_scores.push(macro_f1(&fit, &rows, &data.labels, &val_idx, n_labels));
        }

        let mean = if fold_scores.is_empty() {
            0.0
        } else {
            fold_scores.iter().sum::<f64>() / fold_scores.len() as f64
        };
        trials_detail.push(TrialReport {
            trial: t,
            config: ConfigJson::from(&cfg),
            fold_scores: fold_scores.clone(),
            mean_macro_f1: mean,
        });

        // Strictly-greater keeps the earliest trial on ties (reproducible).
        if best.as_ref().map_or(true, |(_, s, _)| mean > *s) {
            best = Some((t, mean, cfg));
        }
    }

    let (best_trial, best_score, best_cfg) =
        best.ok_or_else(|| "no trials were run (trials must be >= 1)".to_string())?;

    // Retrain the winning configuration on all rows.
    let base = data.base(&best_cfg.tokenizer, best_cfg.lowercase).to_vec();
    let rows = build_rows(&base, &best_cfg);
    let all_idx: Vec<usize> = (0..n).collect();
    let final_seed = space.seed.wrapping_add(0xF1A1);
    let fit = LogReg::train(
        &rows,
        &data.labels,
        &all_idx,
        n_labels,
        best_cfg.n_features,
        &best_cfg,
        space.batch_size,
        final_seed,
    );
    let model = Model::from_fit(data.vocab.clone(), &best_cfg, fit);

    let report = Report {
        n_rows: n,
        n_labels,
        labels: data.vocab.clone(),
        label_support: data.label_support(),
        cv,
        trials: space.trials,
        seed: space.seed,
        batch_size: space.batch_size,
        metric: "macro_f1".to_string(),
        trials_detail,
        best_trial,
        best_config: ConfigJson::from(&best_cfg),
        best_macro_f1: best_score,
        train_seconds: start.elapsed().as_secs_f64(),
    };

    Ok((model, report))
}

/// Parse the JSON search-space spec produced by the Python layer.
pub fn parse_space(spec_json: &str) -> Result<SearchSpace, String> {
    let space: config::SearchSpace =
        serde_json::from_str(spec_json).map_err(|e| format!("invalid search spec: {e}"))?;
    if space.trials < 1 {
        return Err("trials must be >= 1".to_string());
    }
    if space.cv < 2 {
        return Err("cv must be >= 2".to_string());
    }
    Ok(space)
}
