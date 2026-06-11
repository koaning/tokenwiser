//! Criterion micro-benchmarks for the three hot paths: featurization,
//! a single SGD fit, and single-text prediction.
//!
//! Run with: `cargo bench --no-default-features`
//! (drops the PyO3 bindings so no CPython linking is needed).

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use _core::artifact::Model;
use _core::config::TrialConfig;
use _core::model::{build_rows, LogReg};
use _core::rng::Rng;

/// Deterministic synthetic corpus: `n_docs` documents of `words_per_doc`
/// tokens drawn Zipf-ishly from a `vocab`-word vocabulary.
fn corpus(n_docs: usize, words_per_doc: usize, vocab: usize, seed: u64) -> Vec<String> {
    let mut rng = Rng::new(seed);
    (0..n_docs)
        .map(|_| {
            let words: Vec<String> = (0..words_per_doc)
                // Squaring a uniform skews toward low word ids, which gives a
                // crude but deterministic Zipf-like frequency profile.
                .map(|_| {
                    let u = rng.next_f64();
                    format!("w{}", (u * u * vocab as f64) as usize)
                })
                .collect();
            words.join(" ")
        })
        .collect()
}

fn config(ngram_max: u32, n_hashes: usize, binary: bool) -> TrialConfig {
    TrialConfig {
        tokenizer: "unicode".to_string(),
        lowercase: true,
        ngram_min: 1,
        ngram_max,
        n_features: 20_000,
        n_hashes,
        feature_value: if binary { "binary" } else { "count" }.to_string(),
        epochs: 3,
        learning_rate: 0.1,
        l1: 0.0,
        l2: 1e-6,
    }
}

/// Deterministic multilabel targets: 6 labels with descending prevalence.
fn labels(n_docs: usize, n_labels: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    (0..n_docs)
        .map(|_| {
            (0..n_labels as u32)
                .filter(|&l| rng.next_f64() < 0.5 / (l + 1) as f64)
                .collect()
        })
        .collect()
}

fn bench_featurize(c: &mut Criterion) {
    let texts = corpus(2_000, 40, 5_000, 11);
    let mut g = c.benchmark_group("featurize");
    g.sample_size(20);
    g.bench_function("build_rows/uni_h2_binary", |b| {
        let cfg = config(1, 2, true);
        b.iter(|| build_rows(black_box(&texts), &cfg))
    });
    g.bench_function("build_rows/bi_h3_count", |b| {
        let cfg = config(2, 3, false);
        b.iter(|| build_rows(black_box(&texts), &cfg))
    });
    g.finish();
}

fn bench_sgd(c: &mut Criterion) {
    let texts = corpus(2_000, 40, 5_000, 11);
    let n_labels = 6;
    let y = labels(texts.len(), n_labels, 13);
    let cfg = config(1, 2, true);
    let rows = build_rows(&texts, &cfg);
    let idxs: Vec<usize> = (0..texts.len()).collect();
    let mut g = c.benchmark_group("sgd");
    g.sample_size(20);
    g.bench_function("train/2k_rows_6_labels_3_epochs", |b| {
        b.iter(|| {
            LogReg::train(
                black_box(&rows),
                &y,
                &idxs,
                n_labels,
                cfg.n_features,
                &cfg,
                256,
                42,
            )
        })
    });
    g.finish();
}

fn bench_predict(c: &mut Criterion) {
    let texts = corpus(2_000, 40, 5_000, 11);
    let n_labels = 6;
    let y = labels(texts.len(), n_labels, 13);
    let cfg = config(1, 2, true);
    let rows = build_rows(&texts, &cfg);
    let idxs: Vec<usize> = (0..texts.len()).collect();
    let fit = LogReg::train(&rows, &y, &idxs, n_labels, cfg.n_features, &cfg, 256, 42);
    let vocab: Vec<String> = (0..n_labels).map(|l| format!("label{l}")).collect();
    let model = Model::from_fit(vocab, &cfg, fit, vec![0.5; n_labels]);

    let short = "please summarize this email for me quickly".to_string();
    let long = corpus(1, 300, 5_000, 17).pop().unwrap();

    let mut g = c.benchmark_group("predict");
    g.bench_function("proba/short_text", |b| b.iter(|| model.proba(black_box(&short))));
    g.bench_function("proba/long_text", |b| b.iter(|| model.proba(black_box(&long))));
    g.finish();
}

criterion_group!(benches, bench_featurize, bench_sgd, bench_predict);
criterion_main!(benches);
