//! The trained `Model` and its versioned `.tw` on-disk format.
//!
//! Layout:
//!   magic   : 4 bytes  "TKW1"
//!   hdr_len : u32 LE    length of the JSON header
//!   header  : UTF-8 JSON (labels + full preprocessing config)
//!   weights : encoded per the header's `weight_encoding` (see below)
//!   bias    : n_labels  f32 LE
//!
//! Keeping the header as JSON makes the artifact self-describing while the
//! bulk weights stay compact binary.
//!
//! Weight encodings (`weight_encoding` header field):
//!
//! * `"f32_dense"`  — n_labels * n_features f32 LE, row-major by label. The
//!   v2 format; decoded for backward compatibility, never written anymore.
//! * `"f16_dense"`  — same layout, 2-byte IEEE binary16 per weight.
//! * `"f16_sparse"` — per label: u32 nnz, then nnz × (LEB128 index-delta,
//!   u16 f16 bits). Chosen at save time when most weights are zero (hash
//!   buckets never touched by training stay exactly 0.0).
//!
//! Weights are rounded to f16 once, when the artifact is BUILT (`from_fit`):
//! the in-memory model and the bytes on disk hold the same rounded values,
//! so save/load round-trips are bit-exact.
//!
//! NOTE: on disk the weights are row-major by label, but in memory they are
//! **feature-major** (`weights[feature * n_labels + label]`, matching
//! `model::LogReg`) so the predict inner loop is contiguous. Encode/decode
//! transpose at the boundary — a one-off O(matrix) pass.

use std::cell::RefCell;

use serde::{Deserialize, Serialize};

use crate::config::TrialConfig;
use crate::f16::{f16_bits_to_f32, f32_to_f16_bits, round_to_f16};
use crate::features::{
    gram_hashes_into, hash_row, push_buckets_fused, rows_from_sorted_buckets,
};
use crate::model::LogReg;
use crate::tokenize::{base_tokens, ngrams, TokenBuf};

thread_local! {
    /// Reusable per-thread scratch for the predict hot path: token buffer,
    /// gram hashes, bucket ids. Thread-local (rather than per-model) so
    /// rayon batch prediction stays `&self` and allocation-free per call.
    static PREDICT_SCRATCH: RefCell<(TokenBuf, Vec<u64>, Vec<u32>)> =
        RefCell::new((TokenBuf::default(), Vec::new(), Vec::new()));
}

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
    /// How the weight block is encoded; missing in v2 artifacts (f32 dense).
    #[serde(default = "default_weight_encoding")]
    pub weight_encoding: String,
}

fn default_weight_encoding() -> String {
    "f32_dense".to_string()
}

pub struct Model {
    pub header: Header,
    /// Feature-major: `weights[feature * n_labels + label]`.
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
        // Round weights to f16 here, once: predictions, the saved bytes, and
        // any reload all see the same rounded values (bit-exact round trips).
        // The bias stays f32 — it is n_labels floats and shifts every score.
        let mut weights = fit.weights;
        let mut nnz = 0usize;
        for w in weights.iter_mut() {
            *w = round_to_f16(*w);
            if *w != 0.0 {
                nnz += 1;
            }
        }
        // Untouched hash buckets keep an exact 0.0 weight, so real models are
        // often mostly zeros; the sparse encoding then wins. Break-even is
        // ~3 bytes/nonzero (delta + f16) vs 2 bytes/slot dense.
        let weight_encoding = if nnz * 3 < weights.len() * 2 {
            "f16_sparse"
        } else {
            "f16_dense"
        };
        Model {
            header: Header {
                // v3 = fused (Kirsch–Mitzenmacher) bucket scheme; see features.rs.
                version: 3,
                labels: vocab,
                tokenizer: cfg.tokenizer.clone(),
                lowercase: cfg.lowercase,
                ngram_min: cfg.ngram_min,
                ngram_max: cfg.ngram_max,
                n_features: cfg.n_features,
                n_hashes: cfg.n_hashes,
                feature_value: cfg.feature_value.clone(),
                thresholds,
                weight_encoding: weight_encoding.to_string(),
            },
            weights,
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
        let h = &self.header;
        let binary = h.feature_value == "binary";
        let row = if h.version >= 3 {
            // Fused (v3+) path: tokenize into a reusable thread-local buffer
            // and hash grams straight from byte slices — the only allocation
            // left per call is the returned row/probability vectors.
            PREDICT_SCRATCH.with(|cell| {
                let (buf, hashes, scratch) = &mut *cell.borrow_mut();
                buf.fill(text, &h.tokenizer, h.lowercase);
                hashes.clear();
                gram_hashes_into(buf, h.ngram_min, h.ngram_max, hashes);
                scratch.clear();
                for &gh in hashes.iter() {
                    push_buckets_fused(gh, h.n_features, h.n_hashes, scratch);
                }
                rows_from_sorted_buckets(scratch, binary)
            })
        } else {
            // Legacy (v2) per-seed hashing, so old models predict identically
            // forever.
            let tokens = base_tokens(text, &h.tokenizer, h.lowercase);
            let grams = ngrams(&tokens, h.ngram_min, h.ngram_max);
            hash_row(&grams, h.n_features, h.n_hashes, binary, false)
        };

        let n_labels = self.n_labels();
        let mut z = self.bias.clone();
        for (&idx, &v) in row.indices.iter().zip(&row.values) {
            let base = idx as usize * n_labels;
            for l in 0..n_labels {
                z[l] += self.weights[base + l] * v;
            }
        }
        for p in z.iter_mut() {
            *p = 1.0 / (1.0 + (-*p).exp());
        }
        z
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
            8 + header_json.len() + self.weights.len() * 2 + self.bias.len() * 4,
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&header_json);
        // Iterate label-major (disk order); memory is feature-major.
        let n_labels = self.n_labels();
        let n_features = self.header.n_features;
        match self.header.weight_encoding.as_str() {
            "f16_sparse" => {
                for l in 0..n_labels {
                    let nnz =
                        (0..n_features).filter(|&fi| self.weights[fi * n_labels + l] != 0.0);
                    out.extend_from_slice(&(nnz.count() as u32).to_le_bytes());
                    let mut prev = 0usize;
                    for fi in 0..n_features {
                        let w = self.weights[fi * n_labels + l];
                        if w == 0.0 {
                            continue;
                        }
                        write_leb128((fi - prev) as u64, &mut out);
                        out.extend_from_slice(&f32_to_f16_bits(w).to_le_bytes());
                        prev = fi + 1;
                    }
                }
            }
            "f16_dense" => {
                for l in 0..n_labels {
                    for fi in 0..n_features {
                        let bits = f32_to_f16_bits(self.weights[fi * n_labels + l]);
                        out.extend_from_slice(&bits.to_le_bytes());
                    }
                }
            }
            // "f32_dense": only reachable for a loaded legacy model that is
            // re-saved; keep its lossless representation.
            _ => {
                for l in 0..n_labels {
                    for fi in 0..n_features {
                        out.extend_from_slice(&self.weights[fi * n_labels + l].to_le_bytes());
                    }
                }
            }
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
        let n_features = header.n_features;
        let n_weights = n_labels * n_features;
        let mut cursor = hdr_end;

        let take = |cursor: &mut usize, n: usize| -> Result<&[u8], String> {
            if *cursor + n > bytes.len() {
                return Err("truncated model payload".to_string());
            }
            let s = &bytes[*cursor..*cursor + n];
            *cursor += n;
            Ok(s)
        };

        // Decode to dense f32, feature-major (disk is label-major).
        let mut weights = vec![0.0f32; n_weights];
        match header.weight_encoding.as_str() {
            "f16_sparse" => {
                for l in 0..n_labels {
                    let nnz = u32::from_le_bytes(
                        take(&mut cursor, 4)?.try_into().expect("4 bytes"),
                    ) as usize;
                    let mut fi = 0usize;
                    for _ in 0..nnz {
                        let delta = read_leb128(bytes, &mut cursor)? as usize;
                        fi += delta;
                        if fi >= n_features {
                            return Err("sparse weight index out of range".to_string());
                        }
                        let h = u16::from_le_bytes(
                            take(&mut cursor, 2)?.try_into().expect("2 bytes"),
                        );
                        weights[fi * n_labels + l] = f16_bits_to_f32(h);
                        fi += 1;
                    }
                }
            }
            "f16_dense" => {
                for l in 0..n_labels {
                    for fi in 0..n_features {
                        let h = u16::from_le_bytes(
                            take(&mut cursor, 2)?.try_into().expect("2 bytes"),
                        );
                        weights[fi * n_labels + l] = f16_bits_to_f32(h);
                    }
                }
            }
            "f32_dense" => {
                for l in 0..n_labels {
                    for fi in 0..n_features {
                        let v = f32::from_le_bytes(
                            take(&mut cursor, 4)?.try_into().expect("4 bytes"),
                        );
                        weights[fi * n_labels + l] = v;
                    }
                }
            }
            other => return Err(format!("unknown weight encoding {other:?}")),
        }

        let mut bias = Vec::with_capacity(n_labels);
        for _ in 0..n_labels {
            bias.push(f32::from_le_bytes(
                take(&mut cursor, 4)?.try_into().expect("4 bytes"),
            ));
        }
        if cursor != bytes.len() {
            return Err(format!(
                "model payload size mismatch ({} trailing bytes)",
                bytes.len() - cursor
            ));
        }

        Ok(Model {
            header,
            weights,
            bias,
        })
    }
}

/// Unsigned LEB128 (7 bits per byte, high bit = continuation).
fn write_leb128(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn read_leb128(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
    let mut v = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *bytes
            .get(*cursor)
            .ok_or_else(|| "truncated varint in model payload".to_string())?;
        *cursor += 1;
        v |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint overflow in model payload".to_string());
        }
    }
}
