//! Allocation-reusing token output for the single native executable.
//!
//! BPE token boundaries are not UTF-8 boundaries.  This decoder retains only
//! an incomplete suffix and emits the longest valid prefix after each target
//! token, matching the tokenizer's replacement policy for malformed bytes.

use std::str;

use crate::error::Result;
use crate::tokenizer::K3Tokenizer;

#[derive(Debug, Default)]
pub struct IncrementalUtf8Decoder {
    pending: Vec<u8>,
    output: String,
}

impl IncrementalUtf8Decoder {
    pub fn new() -> Self {
        Self {
            // A valid incomplete UTF-8 suffix is at most three bytes. Leave a
            // little room for ordinary multi-byte token payloads too.
            pending: Vec::with_capacity(16),
            output: String::with_capacity(32),
        }
    }

    /// Start a new stream while retaining both reusable allocations.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.output.clear();
    }

    /// Decode one vocabulary item without constructing a temporary byte
    /// vector. The returned view remains valid until the next mutable call.
    pub fn push_token<'a>(&'a mut self, tokenizer: &K3Tokenizer, token_id: u32) -> Result<&'a str> {
        let bytes = tokenizer.token_bytes(token_id)?;
        Ok(self.push_bytes(bytes))
    }

    pub fn push_bytes<'a>(&'a mut self, bytes: &[u8]) -> &'a str {
        self.output.clear();
        self.pending.extend_from_slice(bytes);

        let mut consumed = 0_usize;
        loop {
            match str::from_utf8(&self.pending[consumed..]) {
                Ok(valid) => {
                    self.output.push_str(valid);
                    consumed = self.pending.len();
                    break;
                }
                Err(error) => {
                    let valid_end = consumed + error.valid_up_to();
                    // `valid_up_to` is supplied by `from_utf8` for this exact
                    // slice, so a second checked conversion cannot fail.
                    let valid = str::from_utf8(&self.pending[consumed..valid_end])
                        .expect("validated UTF-8 prefix");
                    self.output.push_str(valid);
                    consumed = valid_end;
                    let Some(invalid_bytes) = error.error_len() else {
                        // Only an incomplete code point remains. Keep it for
                        // the next token rather than printing a false U+FFFD.
                        break;
                    };
                    self.output.push('\u{fffd}');
                    consumed += invalid_bytes;
                    if consumed == self.pending.len() {
                        break;
                    }
                }
            }
        }

        if consumed != 0 {
            let retained = self.pending.len() - consumed;
            self.pending.copy_within(consumed.., 0);
            self.pending.truncate(retained);
        }
        &self.output
    }

    /// Finish a stream. Any incomplete final code point follows Python's
    /// replacement decode policy instead of disappearing silently.
    pub fn finish(&mut self) -> &str {
        self.output.clear();
        if !self.pending.is_empty() {
            self.output
                .push_str(&String::from_utf8_lossy(&self.pending));
            self.pending.clear();
        }
        &self.output
    }

    pub fn pending_bytes(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_complete_text_and_retains_only_a_split_code_point() {
        let mut decoder = IncrementalUtf8Decoder::new();
        assert_eq!(decoder.push_bytes(b"hello "), "hello ");
        assert_eq!(decoder.push_bytes(&[0xf0, 0x9f]), "");
        assert_eq!(decoder.pending_bytes(), 2);
        assert_eq!(decoder.push_bytes(&[0x98, 0x80, b'!']), "😀!");
        assert_eq!(decoder.pending_bytes(), 0);
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn malformed_and_incomplete_final_bytes_match_lossy_decode() {
        let input = [b'a', 0xff, b'b', 0xe2, 0x82];
        let expected = String::from_utf8_lossy(&input).into_owned();
        let mut decoder = IncrementalUtf8Decoder::new();
        let mut actual = decoder.push_bytes(&input).to_owned();
        actual.push_str(decoder.finish());
        assert_eq!(actual, expected);
    }

    #[test]
    fn repeated_ascii_tokens_reuse_the_same_output_allocation() {
        let mut decoder = IncrementalUtf8Decoder::new();
        decoder.push_bytes(b"warmup");
        let capacity = decoder.output.capacity();
        for _ in 0..1_000 {
            assert_eq!(decoder.push_bytes(b"x"), "x");
        }
        assert_eq!(decoder.output.capacity(), capacity);
    }

    #[test]
    fn reset_discards_a_fragment_without_discarding_allocations() {
        let mut decoder = IncrementalUtf8Decoder::new();
        decoder.push_bytes(&[0xf0, 0x9f]);
        let pending_capacity = decoder.pending.capacity();
        decoder.push_bytes(b"warmup");
        let output_capacity = decoder.output.capacity();
        decoder.reset();
        assert_eq!(decoder.pending_bytes(), 0);
        assert_eq!(decoder.finish(), "");
        assert_eq!(decoder.pending.capacity(), pending_capacity);
        assert_eq!(decoder.output.capacity(), output_capacity);
    }
}
