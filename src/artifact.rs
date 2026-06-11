//! The trained `Model` and its versioned `.tw` on-disk format.
//!
//! Layout:
//!   magic   : 4 bytes  "TKW1"
//!   hdr_len : u32 LE    length of the JSON header
//!   header  : UTF-8 JSON (labels + full preprocessing config)
//!   weights : n_labels * n_features  f32 LE (row-major by label)
//!   bias    : n_labels               f32 LE
//!
//! Keeping the header as JSON makes the artifact self-describing while the
//! bulk weights stay compact as raw little-endian floats.

use serde::{Deserialize, Serialize};

use crate::config::TrialConfig;
use crate::features::hash_row;
use crate::model::LogReg;
use crate::tokenize::{base_tokens, ngrams};

const MAGIC: &[u8; 4] = b"TKW1";

#[derive(Serialize, Deserialize, Clone)]
pub struct Header {
    pub version: u32,
    pub labels: Vec<String>,
    pub tokenizer: String,
    pub lowercase: bool,
    pub ngram_min: u32,
    pub ngram_max: u32,
    pub n_features: usize,
    pub n_hashes: usize,
    pub feature_value: String,
    /// Per-label decision thresholds (aligned to `labels`). Empty/missing in
    /// older artifacts, in which case a fixed 0.5 is used.
    #[serde(default)]
    pub thresholds: Vec<f32>,
}

pub struct Model {
    pub header: Header,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Model {
    pub fn from_fit(
        vocab: Vec<String>,
        cfg: &TrialConfig,
        fit: LogReg,
        thresholds: Vec<f32>,
    ) -> Model {
        Model {
            header: Header {
                version: 2,
                labels: vocab,
                tokenizer: cfg.tokenizer.clone(),
                lowercase: cfg.lowercase,
                ngram_min: cfg.ngram_min,
                ngram_max: cfg.ngram_max,
                n_features: cfg.n_features,
                n_hashes: cfg.n_hashes,
                feature_value: cfg.feature_value.clone(),
                thresholds,
            },
            weights: fit.weights,
            bias: fit.bias,
        }
    }

    /// Decision threshold for label `l` (0.5 if none was stored).
    #[inline]
    fn threshold(&self, l: usize) -> f32 {
        self.header.thresholds.get(l).copied().unwrap_or(0.5)
    }

    fn n_labels(&self) -> usize {
        self.header.labels.len()
    }

    /// Probabilities (per label, in vocabulary order) for one text.
    pub fn proba(&self, text: &str) -> Vec<f32> {
        let tokens = base_tokens(text, &self.header.tokenizer, self.header.lowercase);
        let grams = ngrams(&tokens, self.header.ngram_min, self.header.ngram_max);
        let binary = self.header.feature_value == "binary";
        let row = hash_row(&grams, self.header.n_features, self.header.n_hashes, binary);

        let n_features = self.header.n_features;
        let n_labels = self.n_labels();
        let mut z = self.bias.clone();
        for (k, &idx) in row.indices.iter().enumerate() {
            let v = row.values[k];
            let fi = idx as usize;
            for l in 0..n_labels {
                z[l] += self.weights[l * n_features + fi] * v;
            }
        }
        z.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
    }

    /// Predicted labels for one text, using each label's stored threshold
    /// (0.5 when thresholds were not tuned).
    pub fn predict(&self, text: &str) -> Vec<String> {
        let probs = self.proba(text);
        probs
            .iter()
            .enumerate()
            .filter(|(l, &p)| p >= self.threshold(*l))
            .map(|(l, _)| self.header.labels[l].clone())
            .collect()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let header_json = serde_json::to_vec(&self.header).expect("header serializes");
        let mut out = Vec::with_capacity(
            8 + header_json.len() + (self.weights.len() + self.bias.len()) * 4,
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&header_json);
        for &w in &self.weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for &b in &self.bias {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Model, String> {
        if bytes.len() < 8 || &bytes[0..4] != MAGIC {
            return Err("not a tokenwiser model (bad magic)".to_string());
        }
        let hdr_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let hdr_end = 8 + hdr_len;
        if bytes.len() < hdr_end {
            return Err("truncated model header".to_string());
        }
        let header: Header = serde_json::from_slice(&bytes[8..hdr_end])
            .map_err(|e| format!("invalid model header: {e}"))?;

        let n_labels = header.labels.len();
        let n_weights = n_labels * header.n_features;
        let expected = hdr_end + (n_weights + n_labels) * 4;
        if bytes.len() != expected {
            return Err(format!(
                "model payload size mismatch (expected {expected}, got {})",
                bytes.len()
            ));
        }

        let mut cursor = hdr_end;
        let read_f32 = |cursor: &mut usize| -> f32 {
            let v = f32::from_le_bytes([
                bytes[*cursor],
                bytes[*cursor + 1],
                bytes[*cursor + 2],
                bytes[*cursor + 3],
            ]);
            *cursor += 4;
            v
        };
        let weights: Vec<f32> = (0..n_weights).map(|_| read_f32(&mut cursor)).collect();
        let bias: Vec<f32> = (0..n_labels).map(|_| read_f32(&mut cursor)).collect();

        Ok(Model {
            header,
            weights,
            bias,
        })
    }
}
