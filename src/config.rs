//! Search-space specification and per-trial sampling.
//!
//! The Python layer is responsible for type-aware parsing of the public
//! `train(...)` keyword arguments into the JSON described here:
//!
//! * scalar  -> `{"type": "fixed", ...}`
//! * list    -> `{"type": "choice", ...}`
//! * tuple   -> `{"type": "int_range"|"log_range", ...}` (scalar numeric only)
//!
//! Each `*Spec` knows how to sample a concrete value from a [`Rng`].

use serde::Deserialize;

use crate::rng::Rng;

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum StrSpec {
    #[serde(rename = "fixed")]
    Fixed { value: String },
    #[serde(rename = "choice")]
    Choice { values: Vec<String> },
}

impl StrSpec {
    pub fn sample(&self, rng: &mut Rng) -> String {
        match self {
            StrSpec::Fixed { value } => value.clone(),
            StrSpec::Choice { values } => values[rng.index(values.len())].clone(),
        }
    }
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum BoolSpec {
    #[serde(rename = "fixed")]
    Fixed { value: bool },
    #[serde(rename = "choice")]
    Choice { values: Vec<bool> },
}

impl BoolSpec {
    pub fn sample(&self, rng: &mut Rng) -> bool {
        match self {
            BoolSpec::Fixed { value } => *value,
            BoolSpec::Choice { values } => values[rng.index(values.len())],
        }
    }
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum IntSpec {
    #[serde(rename = "fixed")]
    Fixed { value: i64 },
    #[serde(rename = "choice")]
    Choice { values: Vec<i64> },
    #[serde(rename = "int_range")]
    Range { lo: i64, hi: i64 },
}

impl IntSpec {
    pub fn sample(&self, rng: &mut Rng) -> i64 {
        match self {
            IntSpec::Fixed { value } => *value,
            IntSpec::Choice { values } => values[rng.index(values.len())],
            IntSpec::Range { lo, hi } => rng.int_inclusive(*lo, *hi),
        }
    }
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum FloatSpec {
    #[serde(rename = "fixed")]
    Fixed { value: f64 },
    #[serde(rename = "choice")]
    Choice { values: Vec<f64> },
    #[serde(rename = "log_range")]
    LogRange { lo: f64, hi: f64 },
}

impl FloatSpec {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            FloatSpec::Fixed { value } => *value,
            FloatSpec::Choice { values } => values[rng.index(values.len())],
            FloatSpec::LogRange { lo, hi } => rng.log_uniform(*lo, *hi),
        }
    }
}

/// `ngram_range` is a structured tuple: `(min, max)` is *fixed*, a list of
/// such tuples is a *choice*. It is never interpreted as a numeric range.
#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum NgramSpec {
    #[serde(rename = "fixed")]
    Fixed { value: (u32, u32) },
    #[serde(rename = "choice")]
    Choice { values: Vec<(u32, u32)> },
}

impl NgramSpec {
    pub fn sample(&self, rng: &mut Rng) -> (u32, u32) {
        match self {
            NgramSpec::Fixed { value } => *value,
            NgramSpec::Choice { values } => values[rng.index(values.len())],
        }
    }
}

/// The full search space, deserialized from the Python layer.
#[derive(Deserialize, Clone)]
pub struct SearchSpace {
    pub text_field: String,
    pub labels_field: String,
    pub trials: usize,
    pub cv: usize,
    pub seed: u64,
    pub batch_size: usize,

    pub tokenizer: StrSpec,
    pub lowercase: BoolSpec,
    pub ngram_range: NgramSpec,
    pub n_features: IntSpec,
    pub n_hashes: IntSpec,
    pub feature_value: StrSpec,
    pub epochs: IntSpec,
    pub learning_rate: FloatSpec,
    pub l1: FloatSpec,
    pub l2: FloatSpec,
}

/// A concrete configuration sampled for a single trial.
#[derive(Clone, Debug)]
pub struct TrialConfig {
    pub tokenizer: String,
    pub lowercase: bool,
    pub ngram_min: u32,
    pub ngram_max: u32,
    pub n_features: usize,
    pub n_hashes: usize,
    pub feature_value: String,
    pub epochs: usize,
    pub learning_rate: f64,
    pub l1: f64,
    pub l2: f64,
}

impl SearchSpace {
    pub fn sample(&self, rng: &mut Rng) -> TrialConfig {
        let (ngram_min, ngram_max) = self.ngram_range.sample(rng);
        TrialConfig {
            tokenizer: self.tokenizer.sample(rng),
            lowercase: self.lowercase.sample(rng),
            ngram_min,
            ngram_max,
            n_features: self.n_features.sample(rng).max(1) as usize,
            n_hashes: self.n_hashes.sample(rng).max(1) as usize,
            feature_value: self.feature_value.sample(rng),
            epochs: self.epochs.sample(rng).max(1) as usize,
            learning_rate: self.learning_rate.sample(rng),
            l1: self.l1.sample(rng),
            l2: self.l2.sample(rng),
        }
    }
}
