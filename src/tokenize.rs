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

/// Reusable tokenized-document buffer: all (optionally lowercased) token
/// bytes packed into one string, with token boundaries as offsets.
///
/// This is the allocation-free counterpart of [`base_tokens`]: refilling a
/// `TokenBuf` reuses its two buffers, so bulk featurization and single-text
/// prediction do no per-token allocation (ASCII tokens; rare non-ASCII
/// tokens fall back to one transient `String` to keep `str::to_lowercase`
/// semantics, e.g. Greek final sigma).
#[derive(Default)]
pub struct TokenBuf {
    bytes: String,
    offsets: Vec<usize>,
}

impl TokenBuf {
    /// Re-tokenize `text` into this buffer (clears previous contents).
    pub fn fill(&mut self, text: &str, tokenizer: &str, lowercase: bool) {
        self.bytes.clear();
        self.offsets.clear();
        self.offsets.push(0);
        let mut push = |tok: &str| {
            if !lowercase {
                self.bytes.push_str(tok);
            } else if tok.is_ascii() {
                // ASCII fast path: char-wise lowercasing == str::to_lowercase.
                for b in tok.bytes() {
                    self.bytes.push(b.to_ascii_lowercase() as char);
                }
            } else {
                // Keep exact str::to_lowercase semantics for non-ASCII.
                self.bytes.push_str(&tok.to_lowercase());
            }
            self.offsets.push(self.bytes.len());
        };
        match tokenizer {
            "whitespace" => text.split_whitespace().for_each(&mut push),
            _ => text
                .split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .for_each(&mut push),
        }
    }

    pub fn n_tokens(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Byte slice of token `i`.
    #[inline]
    pub fn token_bytes(&self, i: usize) -> &[u8] {
        &self.bytes.as_bytes()[self.offsets[i]..self.offsets[i + 1]]
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
