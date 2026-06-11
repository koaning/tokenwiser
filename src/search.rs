//! Cross-validated randomized search.
//!
//! For each trial a configuration is sampled (seeded, reproducible), evaluated
//! with k-fold CV on macro-F1, and the best configuration is retrained on all
//! rows to produce the final model. A [`Report`] captures every trial.
//!
//! Trials run in parallel across all cores (rayon), but results are bit-for-bit
//! reproducible regardless of thread count: configs and fits are seeded purely
//! from (trial, fold), and per-trial scores are collected by index, so only the
//! completion order varies. Set `RAYON_NUM_THREADS` to pin the core count.

use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;

use crate::config::{SearchSpace, TrialConfig};
use crate::dataset::Dataset;
use crate::features::SparseRow;
use crate::model::{build_rows, LogReg, TokenCache};
use crate::rng::Rng;
use crate::{artifact::Model, config};

/// The slice of a config that determines tokenization output. Trials sharing
/// a signature share one [`TokenCache`]; everything downstream (`n_features`,
/// `n_hashes`, `feature_value`) is integer remixing per trial.
#[derive(PartialEq)]
struct TokenSig {
    tokenizer: String,
    lowercase: bool,
    ngram: (u32, u32),
}

fn token_sig(cfg: &TrialConfig) -> TokenSig {
    TokenSig {
        tokenizer: cfg.tokenizer.clone(),
        lowercase: cfg.lowercase,
        ngram: (cfg.ngram_min, cfg.ngram_max),
    }
}

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
    /// Whether per-label thresholds were tuned (else fixed 0.5).
    pub tune_thresholds: bool,
    /// Tuned per-label decision thresholds (aligned to `labels`).
    pub thresholds: Vec<f32>,
    /// Out-of-fold macro-F1 of the winning config at the tuned thresholds.
    pub oof_macro_f1: f64,
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
pub fn run(space: &SearchSpace, data: Dataset) -> Result<(Model, Report), String> {
    let start = Instant::now();
    let n = data.n_rows();
    let n_labels = data.n_labels();
    let cv = space.cv.max(1);
    let folds = make_folds(n, cv, space.seed);

    // Precompute fold membership lists once (shared across all trials).
    let fold_idx: Vec<(Vec<usize>, Vec<usize>)> = (0..cv)
        .map(|f| {
            let train: Vec<usize> = (0..n).filter(|&i| folds[i] != f).collect();
            let val: Vec<usize> = (0..n).filter(|&i| folds[i] == f).collect();
            (train, val)
        })
        .collect();

    // Configs are deterministic from the seed; enumerate them up front and
    // group trials by tokenization signature, preserving first-appearance
    // order. Each group tokenizes the corpus ONCE into a compact u64
    // gram-hash cache; its trials then derive their feature rows by integer
    // remixing. Groups run sequentially so at most one cache is alive.
    let configs: Vec<TrialConfig> = (0..space.trials)
        .map(|t| space.sample(&mut Rng::derive(space.seed, t as u64 + 1)))
        .collect();
    let mut groups: Vec<(TokenSig, Vec<usize>)> = Vec::new();
    for (t, cfg) in configs.iter().enumerate() {
        let sig = token_sig(cfg);
        match groups.iter_mut().find(|(s, _)| *s == sig) {
            Some((_, ts)) => ts.push(t),
            None => groups.push((sig, vec![t])),
        }
    }

    // Trials are independent, and every fit is seeded deterministically from
    // (trial, fold), so evaluating them in parallel is bit-identical to a
    // serial run — only completion order differs, and reports are collected
    // by trial index. rayon uses all cores by default (set RAYON_NUM_THREADS
    // to pin it).
    let mut trials_slots: Vec<Option<TrialReport>> = (0..space.trials).map(|_| None).collect();
    for (sig, trial_ids) in &groups {
        // A single-trial group gets nothing back for the cache's memory
        // (one u64 per gram corpus-wide); featurize it directly instead.
        // This keeps fixed-config runs (e.g. trials=1 production fits) at
        // the no-cache memory profile.
        let cache = if trial_ids.len() > 1 {
            Some(TokenCache::build(
                &data.texts,
                &sig.tokenizer,
                sig.lowercase,
                sig.ngram.0,
                sig.ngram.1,
            ))
        } else {
            None
        };
        let reports: Vec<TrialReport> = trial_ids
            .par_iter()
            .map(|&t| {
                let cfg = &configs[t];

                // Derive this trial's compact feature rows from the shared
                // cache; reused across folds, dropped at the end of the trial.
                let rows = match &cache {
                    Some(c) => c.rows(
                        cfg.n_features,
                        cfg.n_hashes,
                        cfg.feature_value == "binary",
                    ),
                    None => build_rows(&data.texts, cfg),
                };

                // Folds are independent too; nested rayon parallelism is cheap
                // (work-stealing only splits when cores are idle) and fills the
                // machine when `trials < cores` — notably trials=1 runs.
                // Collected by fold index, so the score order matches the
                // serial loop.
                let fold_scores: Vec<f64> = (0..cv)
                    .into_par_iter()
                    .map(|f| {
                        let (train_idx, val_idx) = &fold_idx[f];
                        if train_idx.is_empty() || val_idx.is_empty() {
                            return None;
                        }
                        let fit_seed = space
                            .seed
                            .wrapping_add((t as u64 + 1).wrapping_mul(1009))
                            .wrapping_add((f as u64 + 1).wrapping_mul(9176));
                        let fit = LogReg::train(
                            &rows,
                            &data.labels,
                            train_idx,
                            n_labels,
                            cfg.n_features,
                            cfg,
                            space.batch_size,
                            fit_seed,
                        );
                        Some(macro_f1(&fit, &rows, &data.labels, val_idx, n_labels))
                    })
                    .collect::<Vec<Option<f64>>>()
                    .into_iter()
                    .flatten()
                    .collect();

                let mean = if fold_scores.is_empty() {
                    0.0
                } else {
                    fold_scores.iter().sum::<f64>() / fold_scores.len() as f64
                };
                TrialReport {
                    trial: t,
                    config: ConfigJson::from(cfg),
                    fold_scores,
                    mean_macro_f1: mean,
                }
            })
            .collect();
        for r in reports {
            let t = r.trial;
            trials_slots[t] = Some(r);
        }
    }
    let trials_detail: Vec<TrialReport> = trials_slots
        .into_iter()
        .map(|r| r.expect("every trial produced a report"))
        .collect();

    // Winner selection is serial; strictly-greater keeps the earliest trial on
    // ties (reproducible regardless of thread count).
    let mut best: Option<(usize, f64)> = None;
    for tr in &trials_detail {
        if best.as_ref().map_or(true, |(_, s)| tr.mean_macro_f1 > *s) {
            best = Some((tr.trial, tr.mean_macro_f1));
        }
    }
    let (best_trial, best_score) =
        best.ok_or_else(|| "no trials were run (trials must be >= 1)".to_string())?;
    let best_cfg = configs[best_trial].clone();

    let rows = build_rows(&data.texts, &best_cfg);
    let all_idx: Vec<usize> = (0..n).collect();

    // Two independent tail jobs run concurrently (rayon::join):
    //
    // * Per-label threshold tuning on the winning config's out-of-fold
    //   predictions: re-run its CV (same per-fold seeds as the search, so the
    //   fold models are identical) and collect each row's probability from the
    //   fold where it was held out. Those unbiased holdout probabilities pick
    //   the threshold that maximizes each label's F1.
    // * The final retrain of the winning configuration on all rows.
    //
    // Neither depends on the other's output, so this is deterministic.
    let final_seed = space.seed.wrapping_add(0xF1A1);
    let ((thresholds, oof_macro_f1), fit) = rayon::join(
        || {
            if space.tune_thresholds {
                let oof =
                    oof_probabilities(space, &data, &best_cfg, &rows, &fold_idx, best_trial, n_labels);
                let thr = tune_thresholds(&oof, &data.labels, n_labels);
                let f1 = macro_f1_at(&oof, &data.labels, &thr, n_labels);
                (thr, f1)
            } else {
                (vec![0.5f32; n_labels], best_score)
            }
        },
        || {
            LogReg::train(
                &rows,
                &data.labels,
                &all_idx,
                n_labels,
                best_cfg.n_features,
                &best_cfg,
                space.batch_size,
                final_seed,
            )
        },
    );
    let model = Model::from_fit(data.vocab.clone(), &best_cfg, fit, thresholds.clone());

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
        tune_thresholds: space.tune_thresholds,
        thresholds,
        oof_macro_f1,
        train_seconds: start.elapsed().as_secs_f64(),
    };

    Ok((model, report))
}

/// Out-of-fold probabilities for one config: `oof[row][label]` is the row's
/// probability from the CV model that did not train on it.
fn oof_probabilities(
    space: &SearchSpace,
    data: &Dataset,
    cfg: &TrialConfig,
    rows: &[SparseRow],
    fold_idx: &[(Vec<usize>, Vec<usize>)],
    trial: usize,
    n_labels: usize,
) -> Vec<Vec<f32>> {
    // Folds run in parallel; each writes a disjoint set of rows, so the merge
    // below is order-independent and the result deterministic.
    let fold_preds: Vec<Vec<(usize, Vec<f32>)>> = fold_idx
        .par_iter()
        .enumerate()
        .map(|(f, (train_idx, val_idx))| {
            if train_idx.is_empty() || val_idx.is_empty() {
                return Vec::new();
            }
            // Same seed formula as the search, so this fold model is identical.
            let fit_seed = space
                .seed
                .wrapping_add((trial as u64 + 1).wrapping_mul(1009))
                .wrapping_add((f as u64 + 1).wrapping_mul(9176));
            let fit = LogReg::train(
                rows,
                &data.labels,
                train_idx,
                n_labels,
                cfg.n_features,
                cfg,
                space.batch_size,
                fit_seed,
            );
            val_idx
                .iter()
                .map(|&i| (i, fit.proba_row(&rows[i])))
                .collect()
        })
        .collect();

    let mut oof = vec![vec![0.5f32; n_labels]; rows.len()];
    for preds in fold_preds {
        for (i, p) in preds {
            oof[i] = p;
        }
    }
    oof
}

/// Pick each label's threshold (grid over [0.05, 0.95]) to maximize its F1 on
/// the out-of-fold predictions. Labels with no positives keep 0.5.
fn tune_thresholds(oof: &[Vec<f32>], labels: &[Vec<u32>], n_labels: usize) -> Vec<f32> {
    let n = oof.len();
    let mut thresholds = vec![0.5f32; n_labels];
    for l in 0..n_labels {
        let has_pos = labels.iter().any(|ls| ls.binary_search(&(l as u32)).is_ok());
        if !has_pos {
            continue;
        }
        let mut best_f1 = -1.0;
        let mut best_t = 0.5f32;
        let mut step = 0;
        while step <= 90 {
            let t = 0.05 + 0.01 * step as f32;
            let (mut tp, mut fp, mut fn_) = (0u64, 0u64, 0u64);
            for i in 0..n {
                let pred = oof[i][l] >= t;
                let actual = labels[i].binary_search(&(l as u32)).is_ok();
                match (pred, actual) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_ += 1,
                    (false, false) => {}
                }
            }
            let p = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let r = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
            let f1 = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
            if f1 > best_f1 {
                best_f1 = f1;
                best_t = t;
            }
            step += 1;
        }
        thresholds[l] = best_t;
    }
    thresholds
}

/// Macro-F1 over OOF predictions at the given per-label thresholds.
fn macro_f1_at(
    oof: &[Vec<f32>],
    labels: &[Vec<u32>],
    thresholds: &[f32],
    n_labels: usize,
) -> f64 {
    if n_labels == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for l in 0..n_labels {
        let (mut tp, mut fp, mut fn_) = (0u64, 0u64, 0u64);
        for (i, probs) in oof.iter().enumerate() {
            let pred = probs[l] >= thresholds[l];
            let actual = labels[i].binary_search(&(l as u32)).is_ok();
            match (pred, actual) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
        let p = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let r = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        sum += if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
    }
    sum / n_labels as f64
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
