//! JSONL ingestion.
//!
//! Rows are read from disk and reduced to `(text, label-indices)`. Labels are
//! mapped to a stable, sorted vocabulary so artifacts are reproducible.
//!
//! We deliberately keep *only* the raw texts and labels — no token strings.
//! Tokenizer/lowercase/n-gram choices are searchable, so `search.rs` groups
//! trials by tokenization signature and builds one compact `TokenCache` (a
//! single fused u64 hash per n-gram) per group; hashing knobs are applied by
//! integer remixing per trial. Holding the corpus as token *strings* was the
//! dominant memory cost and bought almost nothing over re-tokenizing.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use serde_json::Value;

pub struct Dataset {
    pub texts: Vec<String>,
    /// Label indices per row (may be empty; sorted, unique).
    pub labels: Vec<Vec<u32>>,
    /// Stable, sorted label vocabulary.
    pub vocab: Vec<String>,
}

impl Dataset {
    pub fn n_rows(&self) -> usize {
        self.texts.len()
    }

    pub fn n_labels(&self) -> usize {
        self.vocab.len()
    }

    /// Number of rows carrying each label (indexed by label id).
    pub fn label_support(&self) -> Vec<usize> {
        let mut support = vec![0usize; self.vocab.len()];
        for row in &self.labels {
            for &l in row {
                support[l as usize] += 1;
            }
        }
        support
    }

    /// Load and parse a JSONL file.
    pub fn load(path: &str, text_field: &str, labels_field: &str) -> Result<Dataset, String> {
        let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let reader = BufReader::new(file);

        let mut texts = Vec::new();
        let mut raw_labels: Vec<Vec<String>> = Vec::new();
        let mut vocab_map: HashMap<String, u32> = HashMap::new();

        for (lineno, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("read error on line {}: {e}", lineno + 1))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("invalid JSON on line {}: {e}", lineno + 1))?;

            let text = match value.get(text_field) {
                Some(Value::String(s)) => s.clone(),
                Some(_) => {
                    return Err(format!(
                        "line {}: field '{text_field}' must be a string",
                        lineno + 1
                    ))
                }
                None => {
                    return Err(format!(
                        "line {}: missing text field '{text_field}'",
                        lineno + 1
                    ))
                }
            };

            // Labels may be a list of strings, a single string, or absent
            // (an empty label set is valid).
            let labels = match value.get(labels_field) {
                None | Some(Value::Null) => Vec::new(),
                Some(Value::String(s)) => vec![s.clone()],
                Some(Value::Array(arr)) => {
                    let mut out = Vec::with_capacity(arr.len());
                    for item in arr {
                        match item {
                            Value::String(s) => out.push(s.clone()),
                            _ => {
                                return Err(format!(
                                    "line {}: labels must be strings",
                                    lineno + 1
                                ))
                            }
                        }
                    }
                    out
                }
                Some(_) => {
                    return Err(format!(
                        "line {}: field '{labels_field}' must be a string or list of strings",
                        lineno + 1
                    ))
                }
            };

            for label in &labels {
                let next = vocab_map.len() as u32;
                vocab_map.entry(label.clone()).or_insert(next);
            }
            texts.push(text);
            raw_labels.push(labels);
        }

        if texts.is_empty() {
            return Err("dataset is empty (no usable JSONL rows)".to_string());
        }

        // Stable sorted vocabulary, then remap label ids.
        let mut vocab: Vec<String> = vocab_map.keys().cloned().collect();
        vocab.sort();
        let id_of: HashMap<&str, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u32))
            .collect();
        let labels: Vec<Vec<u32>> = raw_labels
            .iter()
            .map(|row| {
                let mut ids: Vec<u32> = row.iter().map(|l| id_of[l.as_str()]).collect();
                ids.sort_unstable();
                ids.dedup();
                ids
            })
            .collect();

        Ok(Dataset {
            texts,
            labels,
            vocab,
        })
    }
}
