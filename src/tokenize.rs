//! Tokenization and word n-gram generation.
//!
//! Two tokenizers are supported:
//!
//! * `"whitespace"` — split on Unicode whitespace, keep the runs verbatim.
//! * `"unicode"`    — split into maximal runs of alphanumeric characters,
//!                    discarding punctuation. This is a light, dependency-free
//!                    approximation of Unicode word segmentation.
//!
//! `lowercase` (when set) applies `str::to_lowercase` to every base token.

/// Split a single document into base tokens.
pub fn base_tokens(text: &str, tokenizer: &str, lowercase: bool) -> Vec<String> {
    let raw: Vec<&str> = match tokenizer {
        "whitespace" => text.split_whitespace().collect(),
        // "unicode" (default): runs of alphanumeric characters.
        _ => text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect(),
    };
    if lowercase {
        raw.into_iter().map(|t| t.to_lowercase()).collect()
    } else {
        raw.into_iter().map(|t| t.to_string()).collect()
    }
}

/// Generate word n-grams (for every n in `min..=max`) from base tokens.
///
/// Unigrams are emitted verbatim; higher-order grams join tokens with a
/// single space. The space separator is unambiguous because base tokens
/// never contain whitespace.
pub fn ngrams(tokens: &[String], min: u32, max: u32) -> Vec<String> {
    let min = min.max(1) as usize;
    let max = max.max(min as u32) as usize;
    let mut out = Vec::new();
    for n in min..=max {
        if n == 0 || tokens.len() < n {
            continue;
        }
        for window in tokens.windows(n) {
            if n == 1 {
                out.push(window[0].clone());
            } else {
                out.push(window.join(" "));
            }
        }
    }
    out
}
