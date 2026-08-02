//! Python-free Kimi K3 tokenizer.
//!
//! K3 uses a tiktoken rank file, a fixed Unicode pre-tokenization expression,
//! and 256 reserved token IDs whose public spellings are overridden by
//! `tokenizer_config.json`.  This module implements that contract directly in
//! the one Deltafin executable.  Chat/XTML rendering intentionally remains a
//! separate concern: callers choose explicitly whether structural special
//! tokens are enabled for each already-rendered segment.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use regex::Regex;
use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::error::{DeltafinError, Result};

type TokenRanks = FxHashMap<Vec<u8>, u32>;
type TokenDecoder = Vec<Vec<u8>>;

const BASE_VOCAB_SIZE: usize = 163_584;
const RESERVED_SPECIAL_TOKENS: usize = 256;
const K3_VOCAB_SIZE: usize = BASE_VOCAB_SIZE + RESERVED_SPECIAL_TOKENS;
const MAX_MODEL_BYTES: u64 = 8 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_ENCODE_CHARS: usize = 400_000;
const MAX_CONSECUTIVE_CLASS_CHARS: usize = 25_000;
// GigaToken 0.10.0's reviewed interface uses stable-order native batch
// encoding and lets independent rows fan out inside Rust. Chat XTML segments
// are likewise independent at their already-classified boundaries. Keep small
// prompts sequential: scoped thread creation pays off only for a substantial
// history, and the decoder itself remains the sole source of token IDs.
const PARALLEL_SEGMENT_MIN_BYTES: usize = 128 * 1024;
const PARALLEL_SEGMENT_MIN_COUNT: usize = 8;
const PARALLEL_SEGMENT_MAX_WORKERS: usize = 8;
// These are the first six (regular-language) alternatives from the pattern in
// k3-meta/tokenization_kimi.py.  Its final two whitespace alternatives are
// handled explicitly below: that preserves their negative-lookahead boundary
// semantics while keeping matching linear-time and non-backtracking.
const K3_REGULAR_PATTERN: &str = concat!(
    r"[\p{Han}]+",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"\p{N}{1,3}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    "|",
    r"\s*[\r\n]+",
);

/// Exact K3 byte-pair encoder and decoder.
///
/// Construction validates the complete K3 vocabulary before publishing the
/// object, so inference cannot silently proceed with a truncated or mismatched
/// tokenizer artifact.
#[derive(Debug)]
pub struct K3Tokenizer {
    ranks: TokenRanks,
    decoder: TokenDecoder,
    specials: SpecialTrie,
    pattern: Regex,
}

impl K3Tokenizer {
    /// Load `tiktoken.model` and `tokenizer_config.json` from either a metadata
    /// directory or a Deltafin/model root containing `k3-meta/`.
    pub fn load_from_root(root: &Path) -> Result<Self> {
        let direct_model = root.join("tiktoken.model");
        let metadata_root = if direct_model.is_file() {
            root.to_path_buf()
        } else {
            root.join("k3-meta")
        };
        Self::load(
            &metadata_root.join("tiktoken.model"),
            &metadata_root.join("tokenizer_config.json"),
        )
    }

    /// Load exact tokenizer artifacts from explicit paths.
    pub fn load(model_path: &Path, config_path: &Path) -> Result<Self> {
        let (ranks, mut decoder) = load_mergeable_ranks(model_path)?;
        let special_tokens = load_special_tokens(config_path)?;
        let mut specials = SpecialTrie::new();
        for (offset, spelling) in special_tokens.into_iter().enumerate() {
            let token = BASE_VOCAB_SIZE + offset;
            specials.insert(&spelling, token as u32)?;
            decoder.push(spelling.into_bytes());
        }
        validate_required_special_tokens(&decoder)?;
        specials.validate_unambiguous()?;
        if decoder.len() != K3_VOCAB_SIZE {
            return Err(DeltafinError::new(format!(
                "tokenizer decoder has {} entries, expected {K3_VOCAB_SIZE}",
                decoder.len()
            )));
        }
        let pattern = Regex::new(K3_REGULAR_PATTERN).map_err(|error| {
            DeltafinError::new(format!("compile the fixed K3 tokenizer pattern: {error}"))
        })?;
        Ok(Self {
            ranks,
            decoder,
            specials,
            pattern,
        })
    }

    pub const fn vocab_size(&self) -> usize {
        K3_VOCAB_SIZE
    }

    /// Encode ordinary user/model text.  Strings that resemble control tokens
    /// remain ordinary BPE input and therefore cannot inject XTML structure.
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        self.encode(text, false)
    }

    /// Encode an already-classified segment.  `allow_special` is intended only
    /// for trusted structural segments produced by the chat renderer.
    pub fn encode(&self, text: &str, allow_special: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(text.len() / 3 + 1);
        self.encode_into(text, allow_special, &mut tokens)?;
        Ok(tokens)
    }

    /// Append encoded IDs to a caller-owned arena. Server/chat paths use this
    /// to avoid allocating one temporary vector for every XTML segment.
    pub fn encode_into(
        &self,
        text: &str,
        allow_special: bool,
        tokens: &mut Vec<u32>,
    ) -> Result<()> {
        let checkpoint = tokens.len();
        let result = (|| {
            for large_slice in split_by_char_count(text, MAX_ENCODE_CHARS) {
                for safe_slice in
                    split_long_character_classes(large_slice, MAX_CONSECUTIVE_CLASS_CHARS)
                {
                    self.encode_slice(safe_slice, allow_special, tokens)?;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            tokens.truncate(checkpoint);
        }
        result
    }

    pub(crate) fn should_parallelize_segments(
        &self,
        segment_count: usize,
        total_bytes: usize,
    ) -> bool {
        segment_count >= PARALLEL_SEGMENT_MIN_COUNT
            && total_bytes >= PARALLEL_SEGMENT_MIN_BYTES
            && std::thread::available_parallelism().is_ok_and(|workers| workers.get() > 1)
    }

    /// Encode independent, already-classified segments in parallel and
    /// concatenate their rows in input order.
    ///
    /// This does not split a BPE stream at new boundaries: callers must pass
    /// the exact same semantic boundaries used by sequential `encode_into`.
    /// Therefore scheduling can change throughput but never token IDs.
    pub(crate) fn encode_segments(&self, segments: &[(&str, bool)]) -> Result<Vec<u32>> {
        let total_bytes = segments.iter().try_fold(0usize, |total, (text, _)| {
            total
                .checked_add(text.len())
                .ok_or_else(|| DeltafinError::new("segment byte length overflows usize"))
        })?;
        if !self.should_parallelize_segments(segments.len(), total_bytes) {
            return self.encode_segments_sequential(segments);
        }

        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(PARALLEL_SEGMENT_MAX_WORKERS)
            .min(segments.len());
        let next = AtomicUsize::new(0);
        let rows = std::thread::scope(|scope| -> Result<Vec<Option<Vec<u32>>>> {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                handles.push(scope.spawn(|| -> Result<Vec<(usize, Vec<u32>)>> {
                    let mut rows = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(text, allow_special)) = segments.get(index) else {
                            break;
                        };
                        rows.push((index, self.encode(text, allow_special)?));
                    }
                    Ok(rows)
                }));
            }

            let mut ordered: Vec<Option<Vec<u32>>> = (0..segments.len()).map(|_| None).collect();
            for handle in handles {
                let completed = handle.join().map_err(|_| {
                    DeltafinError::new("native tokenizer segment worker panicked")
                })??;
                for (index, row) in completed {
                    ordered[index] = Some(row);
                }
            }
            Ok(ordered)
        })?;
        flatten_segment_rows(rows)
    }

    fn encode_segments_sequential(&self, segments: &[(&str, bool)]) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        for &(text, allow_special) in segments {
            self.encode_into(text, allow_special, &mut tokens)?;
        }
        Ok(tokens)
    }

    /// Concatenate the exact byte payload for token IDs.  This is the safe
    /// primitive for incremental UTF-8 streaming because individual tokens may
    /// end in the middle of a code point.
    pub fn decode_bytes(&self, token_ids: &[u32]) -> Result<Vec<u8>> {
        let total = token_ids.iter().try_fold(0usize, |total, &token| {
            let bytes = self.decoder.get(token as usize).ok_or_else(|| {
                DeltafinError::new(format!(
                    "token ID {token} is outside the K3 vocabulary (0..{K3_VOCAB_SIZE})"
                ))
            })?;
            total
                .checked_add(bytes.len())
                .ok_or_else(|| DeltafinError::new("decoded token byte length overflow"))
        })?;
        let mut output = Vec::with_capacity(total);
        for &token in token_ids {
            output.extend_from_slice(&self.decoder[token as usize]);
        }
        Ok(output)
    }

    /// Match tiktoken's default decode policy: malformed token byte sequences
    /// become Unicode replacement characters instead of aborting generation.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(token_ids)?).into_owned())
    }

    /// Borrow one token's exact byte payload without allocating.  The native
    /// streaming decoder uses this for every emitted token; callers that need
    /// a complete string should prefer [`Self::decode`].
    pub fn token_bytes(&self, token_id: u32) -> Result<&[u8]> {
        self.decoder
            .get(token_id as usize)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                DeltafinError::new(format!(
                    "token ID {token_id} is outside the K3 vocabulary (0..{K3_VOCAB_SIZE})"
                ))
            })
    }

    fn encode_slice(&self, text: &str, allow_special: bool, output: &mut Vec<u32>) -> Result<()> {
        if !allow_special {
            return self.encode_ordinary_slice(text, output);
        }

        let mut cursor = 0;
        while let Some(found) = self.specials.find_next(text.as_bytes(), cursor) {
            self.encode_ordinary_slice(&text[cursor..found.start], output)?;
            output.push(found.token);
            cursor = found.end;
        }
        self.encode_ordinary_slice(&text[cursor..], output)
    }

    fn encode_ordinary_slice(&self, text: &str, output: &mut Vec<u32>) -> Result<()> {
        let mut covered = 0;
        for found in self.pattern.find_iter(text) {
            if found.start() != covered {
                let match_starts_with_whitespace = found
                    .as_str()
                    .chars()
                    .next()
                    .is_some_and(char::is_whitespace);
                self.encode_whitespace_gap(
                    &text[covered..found.start()],
                    !match_starts_with_whitespace,
                    output,
                )?;
            }
            self.encode_bpe(found.as_str().as_bytes(), output)?;
            covered = found.end();
        }
        if covered != text.len() {
            self.encode_whitespace_gap(&text[covered..], false, output)?;
        }
        Ok(())
    }

    fn encode_whitespace_gap(
        &self,
        gap: &str,
        followed_by_non_whitespace: bool,
        output: &mut Vec<u32>,
    ) -> Result<()> {
        if gap.is_empty() {
            return Ok(());
        }
        if !gap.chars().all(char::is_whitespace) {
            return Err(DeltafinError::new(
                "K3 tokenizer regular pattern left a non-whitespace byte unmatched",
            ));
        }

        // `\s+(?!\S)|\s+` emits all but the final whitespace as one piece
        // when a non-whitespace token follows, then emits the final whitespace
        // separately (unless an earlier alternative has already attached it
        // to the following word/punctuation token).  At end-of-input it emits
        // the complete run as one piece.
        if followed_by_non_whitespace {
            let last_start = gap.char_indices().next_back().map_or(0, |(index, _)| index);
            if last_start > 0 {
                self.encode_bpe(&gap.as_bytes()[..last_start], output)?;
            }
            self.encode_bpe(&gap.as_bytes()[last_start..], output)
        } else {
            self.encode_bpe(gap.as_bytes(), output)
        }
    }

    fn encode_bpe(&self, piece: &[u8], output: &mut Vec<u32>) -> Result<()> {
        if piece.is_empty() {
            return Ok(());
        }
        if let Some(&rank) = self.ranks.get(piece) {
            output.push(rank);
            return Ok(());
        }
        if piece.len() == 1 {
            return Err(DeltafinError::new(format!(
                "K3 tokenizer lacks the single-byte token 0x{:02x}",
                piece[0]
            )));
        }

        byte_pair_encode(&self.ranks, piece, output)
    }
}

fn flatten_segment_rows(rows: Vec<Option<Vec<u32>>>) -> Result<Vec<u32>> {
    let total = rows.iter().try_fold(0usize, |total, row| {
        let row = row
            .as_ref()
            .ok_or_else(|| DeltafinError::new("native tokenizer segment worker lost a row"))?;
        total
            .checked_add(row.len())
            .ok_or_else(|| DeltafinError::new("segment token count overflows usize"))
    })?;
    let mut tokens = Vec::with_capacity(total);
    for row in rows {
        tokens.extend(row.expect("validated complete segment rows"));
    }
    Ok(tokens)
}

#[derive(Debug, Clone, Copy)]
struct BpeNode {
    start: usize,
    end: usize,
    previous: Option<usize>,
    next: Option<usize>,
    generation: u32,
    alive: bool,
}

type MergeCandidate = Reverse<(u32, usize, usize, usize, u32, u32)>;

fn byte_pair_encode(ranks: &TokenRanks, piece: &[u8], output: &mut Vec<u32>) -> Result<()> {
    let mut nodes = Vec::with_capacity(piece.len());
    for index in 0..piece.len() {
        nodes.push(BpeNode {
            start: index,
            end: index + 1,
            previous: index.checked_sub(1),
            next: (index + 1 < piece.len()).then_some(index + 1),
            generation: 0,
            alive: true,
        });
    }
    let mut candidates = BinaryHeap::with_capacity(piece.len());
    for left in 0..piece.len() - 1 {
        push_candidate(ranks, piece, &nodes, left, &mut candidates);
    }

    while let Some(Reverse((_, _, left, right, left_generation, right_generation))) =
        candidates.pop()
    {
        if !nodes[left].alive
            || !nodes[right].alive
            || nodes[left].next != Some(right)
            || nodes[left].generation != left_generation
            || nodes[right].generation != right_generation
        {
            continue;
        }

        let previous = nodes[left].previous;
        let next = nodes[right].next;
        nodes[left].end = nodes[right].end;
        nodes[left].next = next;
        nodes[left].generation = nodes[left].generation.wrapping_add(1);
        nodes[right].alive = false;
        nodes[right].generation = nodes[right].generation.wrapping_add(1);
        if let Some(next) = next {
            nodes[next].previous = Some(left);
        }
        if let Some(previous) = previous {
            push_candidate(ranks, piece, &nodes, previous, &mut candidates);
        }
        push_candidate(ranks, piece, &nodes, left, &mut candidates);
    }

    let mut current = Some(0);
    while let Some(index) = current {
        debug_assert!(nodes[index].alive);
        let rank = ranks
            .get(&piece[nodes[index].start..nodes[index].end])
            .copied()
            .ok_or_else(|| {
                DeltafinError::new("K3 BPE produced a byte sequence absent from the rank table")
            })?;
        output.push(rank);
        current = nodes[index].next;
    }
    Ok(())
}

#[inline]
fn push_candidate(
    ranks: &TokenRanks,
    piece: &[u8],
    nodes: &[BpeNode],
    left: usize,
    candidates: &mut BinaryHeap<MergeCandidate>,
) {
    let Some(right) = nodes[left].next else {
        return;
    };
    let Some(&rank) = ranks.get(&piece[nodes[left].start..nodes[right].end]) else {
        return;
    };
    candidates.push(Reverse((
        rank,
        nodes[left].start,
        left,
        right,
        nodes[left].generation,
        nodes[right].generation,
    )));
}

fn load_mergeable_ranks(path: &Path) -> Result<(TokenRanks, TokenDecoder)> {
    validate_regular_bounded_file(path, MAX_MODEL_BYTES, "tokenizer model")?;
    let file = File::open(path).map_err(|error| io_error("open", path, error))?;
    let mut ranks = FxHashMap::with_capacity_and_hasher(BASE_VOCAB_SIZE, Default::default());
    let mut decoder = Vec::with_capacity(K3_VOCAB_SIZE);

    for (line_index, line) in BufReader::new(file.take(MAX_MODEL_BYTES + 1))
        .lines()
        .enumerate()
    {
        let line = line.map_err(|error| io_error("read", path, error))?;
        let (encoded, raw_rank) = line.split_once(' ').ok_or_else(|| {
            DeltafinError::new(format!(
                "invalid tokenizer model line {} in {}",
                line_index + 1,
                path.display()
            ))
        })?;
        if encoded.is_empty() || raw_rank.is_empty() || raw_rank.contains(char::is_whitespace) {
            return Err(DeltafinError::new(format!(
                "invalid tokenizer model fields on line {} in {}",
                line_index + 1,
                path.display()
            )));
        }
        let rank = raw_rank.parse::<u32>().map_err(|_| {
            DeltafinError::new(format!(
                "invalid tokenizer rank on line {} in {}",
                line_index + 1,
                path.display()
            ))
        })?;
        if rank as usize != line_index {
            return Err(DeltafinError::new(format!(
                "tokenizer ranks must be contiguous and ordered: line {} contains rank {rank}",
                line_index + 1
            )));
        }
        let bytes = BASE64_STANDARD.decode(encoded).map_err(|error| {
            DeltafinError::new(format!(
                "invalid base64 token on line {} in {}: {error}",
                line_index + 1,
                path.display()
            ))
        })?;
        if bytes.is_empty() {
            return Err(DeltafinError::new(format!(
                "empty mergeable token on line {} in {}",
                line_index + 1,
                path.display()
            )));
        }
        if ranks.insert(bytes.clone(), rank).is_some() {
            return Err(DeltafinError::new(format!(
                "duplicate mergeable token on line {} in {}",
                line_index + 1,
                path.display()
            )));
        }
        decoder.push(bytes);
    }

    if decoder.len() != BASE_VOCAB_SIZE {
        return Err(DeltafinError::new(format!(
            "K3 tokenizer has {} mergeable tokens, expected {BASE_VOCAB_SIZE}: {}",
            decoder.len(),
            path.display()
        )));
    }
    for byte in 0u8..=u8::MAX {
        if !ranks.contains_key(&[byte][..]) {
            return Err(DeltafinError::new(format!(
                "K3 tokenizer is missing single-byte token 0x{byte:02x}"
            )));
        }
    }
    Ok((ranks, decoder))
}

fn validate_required_special_tokens(decoder: &[Vec<u8>]) -> Result<()> {
    const REQUIRED: &[(usize, &str)] = &[
        (163_584, "[BOS]"),
        (163_585, "[EOS]"),
        (163_586, "<|end_of_msg|>"),
        (163_587, "<|open|>"),
        (163_588, "<|close|>"),
        (163_589, "<|sep|>"),
        (163_590, "[start_header_id]"),
        (163_591, "[end_header_id]"),
        (163_593, "[EOT]"),
        (163_602, "<|media_begin|>"),
        (163_603, "<|media_content|>"),
        (163_604, "<|media_end|>"),
        (163_605, "<|media_pad|>"),
        (163_649, "<osagent_mode>"),
        (163_838, "[UNK]"),
        (163_839, "[PAD]"),
    ];
    for &(token, spelling) in REQUIRED {
        if decoder.get(token).map(Vec::as_slice) != Some(spelling.as_bytes()) {
            return Err(DeltafinError::new(format!(
                "K3 tokenizer special token {token} does not spell {spelling:?}"
            )));
        }
    }
    Ok(())
}

fn load_special_tokens(path: &Path) -> Result<Vec<String>> {
    validate_regular_bounded_file(path, MAX_CONFIG_BYTES, "tokenizer config")?;
    let file = File::open(path).map_err(|error| io_error("open", path, error))?;
    let mut reader = BufReader::new(file.take(MAX_CONFIG_BYTES + 1));
    let value: Value = serde_json::from_reader(&mut reader).map_err(|error| {
        DeltafinError::new(format!(
            "parse tokenizer config {}: {error}",
            path.display()
        ))
    })?;
    let overrides = value
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            DeltafinError::new(format!(
                "tokenizer config lacks added_tokens_decoder: {}",
                path.display()
            ))
        })?;

    let mut tokens: Vec<String> = (BASE_VOCAB_SIZE..K3_VOCAB_SIZE)
        .map(|token| format!("<|reserved_token_{token}|>"))
        .collect();
    for (raw_id, entry) in overrides {
        let id = raw_id.parse::<usize>().map_err(|_| {
            DeltafinError::new(format!(
                "invalid added-token ID {raw_id:?} in {}",
                path.display()
            ))
        })?;
        if !(BASE_VOCAB_SIZE..K3_VOCAB_SIZE).contains(&id) {
            return Err(DeltafinError::new(format!(
                "added-token ID {id} is outside K3's reserved range in {}",
                path.display()
            )));
        }
        let content = entry
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                DeltafinError::new(format!(
                    "added-token ID {id} lacks string content in {}",
                    path.display()
                ))
            })?;
        if content.is_empty() || !content.is_ascii() {
            return Err(DeltafinError::new(format!(
                "K3 special token {id} must have non-empty ASCII content"
            )));
        }
        tokens[id - BASE_VOCAB_SIZE] = content.to_owned();
    }
    Ok(tokens)
}

fn validate_regular_bounded_file(path: &Path, maximum: u64, label: &str) -> Result<()> {
    let metadata = path
        .metadata()
        .map_err(|error| io_error("inspect", path, error))?;
    if !metadata.is_file() {
        return Err(DeltafinError::new(format!(
            "{label} is not a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() > maximum {
        return Err(DeltafinError::new(format!(
            "{label} exceeds the {maximum}-byte safety limit: {}",
            path.display()
        )));
    }
    Ok(())
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

fn split_by_char_count(text: &str, maximum: usize) -> Vec<&str> {
    debug_assert!(maximum > 0);
    if text.is_empty() {
        return vec![text];
    }
    // Most prompts fit in one slice.  Avoid a full extra Unicode traversal
    // merely to compute an exact capacity for the rare multi-slice case.
    let mut slices = Vec::new();
    let mut start = 0;
    let mut count = 0;
    for (byte_index, _) in text.char_indices() {
        if count == maximum {
            slices.push(&text[start..byte_index]);
            start = byte_index;
            count = 0;
        }
        count += 1;
    }
    slices.push(&text[start..]);
    slices
}

fn split_long_character_classes(text: &str, maximum: usize) -> Vec<&str> {
    debug_assert!(maximum > 0);
    if text.is_empty() {
        return vec![text];
    }
    let mut slices = Vec::new();
    let mut slice_start = 0;
    let mut current_length = 0;
    let mut current_is_space = text.chars().next().is_some_and(is_python_whitespace);
    for (byte_index, character) in text.char_indices() {
        let is_space = is_python_whitespace(character);
        if current_is_space != is_space {
            current_length = 1;
            current_is_space = is_space;
        } else {
            current_length += 1;
            if current_length > maximum {
                slices.push(&text[slice_start..byte_index]);
                slice_start = byte_index;
                current_length = 1;
            }
        }
    }
    slices.push(&text[slice_start..]);
    slices
}

// Python's str.isspace() includes the four ASCII information separators that
// Rust's Unicode White_Space predicate intentionally excludes.  The safety
// splitter in tokenization_kimi.py uses Python's definition, so preserve it at
// the only boundary where the distinction can affect BPE segmentation.
#[inline]
fn is_python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
}

#[derive(Debug, Default)]
struct TrieNode {
    edges: Vec<(u8, usize)>,
    token: Option<u32>,
}

#[derive(Debug, Default)]
struct SpecialTrie {
    nodes: Vec<TrieNode>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct SpecialMatch {
    start: usize,
    end: usize,
    token: u32,
}

impl SpecialTrie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    fn insert(&mut self, spelling: &str, token: u32) -> Result<()> {
        let mut node = 0;
        for byte in spelling.bytes() {
            let next = match self.nodes[node]
                .edges
                .iter()
                .find(|(candidate, _)| *candidate == byte)
            {
                Some((_, next)) => *next,
                None => {
                    let next = self.nodes.len();
                    self.nodes.push(TrieNode::default());
                    self.nodes[node].edges.push((byte, next));
                    next
                }
            };
            node = next;
        }
        if self.nodes[node].token.replace(token).is_some() {
            return Err(DeltafinError::new(format!(
                "duplicate K3 special-token spelling {spelling:?}"
            )));
        }
        Ok(())
    }

    fn validate_unambiguous(&self) -> Result<()> {
        for node in &self.nodes {
            if node.token.is_some() && !node.edges.is_empty() {
                return Err(DeltafinError::new(
                    "one K3 special token is a prefix of another; matching would be ambiguous",
                ));
            }
        }
        Ok(())
    }

    fn find_next(&self, text: &[u8], cursor: usize) -> Option<SpecialMatch> {
        for start in cursor..text.len() {
            let mut node = 0;
            for (relative, &byte) in text[start..].iter().enumerate() {
                let Some((_, next)) = self.nodes[node]
                    .edges
                    .iter()
                    .find(|(candidate, _)| *candidate == byte)
                else {
                    break;
                };
                node = *next;
                if let Some(token) = self.nodes[node].token {
                    return Some(SpecialMatch {
                        start,
                        end: start + relative + 1,
                        token,
                    });
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repository_root() -> PathBuf {
        if let Some(root) = std::env::var_os("DELTAFIN_TEST_ROOT") {
            return PathBuf::from(root);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn installed_k3_tokenizer_matches_tiktoken_gold_corpus() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        assert_eq!(tokenizer.vocab_size(), 163_840);

        let cases: &[(&str, &[u32])] = &[
            ("", &[]),
            ("The capital of France is", &[1008, 10484, 318, 15383, 387]),
            (
                "The largest planet in our solar system is",
                &[1008, 10604, 19645, 306, 1088, 17860, 2403, 387],
            ),
            ("Hello, world!", &[19180, 11, 2695, 0]),
            ("Hello\n\nworld  ", &[19180, 382, 23617, 256]),
            (
                "I'M we're they'LL 1234567",
                &[40, 103645, 13810, 1129, 6, 5847, 220, 6694, 12972, 22],
            ),
            ("你好，世界！漢字", &[33845, 378, 2243, 856, 1435, 95, 1855]),
            (
                "Привет — café naïve",
                &[159025, 63694, 4275, 70591, 7915, 44187, 367],
            ),
            (
                "🙂🚀👩\u{200d}💻",
                &[5885, 46813, 5885, 40101, 64390, 102, 67963, 66122, 119],
            ),
            (
                "Tabs\tand\r\nlines\u{a0}end",
                &[46154, 58443, 462, 11541, 5406, 517],
            ),
            ("a\u{1c}b", &[64, 216, 65]),
            (" \u{1c} a", &[220, 216, 261]),
            ("a\u{2028}b", &[64, 390, 101, 65]),
            (
                "ＡＢＣ 한국어 हिन्दी العربية",
                &[
                    320, 94, 320, 95, 320, 96, 78560, 54078, 28763, 25042, 48190, 84837, 10722,
                    120413, 12441,
                ],
            ),
        ];
        for &(text, expected) in cases {
            assert_eq!(
                tokenizer.encode_ordinary(text).unwrap(),
                expected,
                "{text:?}"
            );
            assert_eq!(tokenizer.decode(expected).unwrap(), text);
        }
    }

    #[test]
    fn special_tokens_are_explicit_and_user_text_cannot_inject_them() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        let text = "a <|open|> b [BOS] c";
        assert_eq!(
            tokenizer.encode_ordinary(text).unwrap(),
            [64, 22652, 4454, 91, 29, 291, 793, 33, 4110, 60, 275]
        );
        assert_eq!(
            tokenizer.encode(text, true).unwrap(),
            [64, 220, 163587, 291, 220, 163584, 275]
        );
        let structural = [
            163584, 163585, 163586, 163587, 163588, 163589, 163838, 163839,
        ];
        assert_eq!(
            tokenizer.decode(&structural).unwrap(),
            "[BOS][EOS]<|end_of_msg|><|open|><|close|><|sep|>[UNK][PAD]"
        );
    }

    #[test]
    fn decoder_rejects_ids_outside_the_validated_vocabulary() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        assert!(tokenizer.decode(&[163_840]).is_err());
    }

    #[test]
    fn long_input_safety_boundaries_match_tiktoken() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        let cases = [
            ("a".repeat(25_001), 3_126, 0xaeb0_77ec_9c03_b22e),
            (" ".repeat(25_001), 100, 0x73a7_566b_071d_3a87),
            ("🙂".repeat(25_001), 50_002, 0x63d6_5387_3538_8801),
            (
                format!("{}b", "a ".repeat(200_000)),
                200_002,
                0x59c1_be93_4dfc_d242,
            ),
        ];
        for (text, expected_len, expected_hash) in cases {
            let ids = tokenizer.encode_ordinary(&text).unwrap();
            assert_eq!(ids.len(), expected_len);
            assert_eq!(token_id_hash(&ids), expected_hash);
        }
    }

    #[test]
    fn combinatorial_boundary_corpus_matches_tiktoken() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        let fragments = [
            "",
            "a",
            "A",
            "Ab",
            "ABCd",
            "é",
            "e\u{301}",
            "汉",
            "漢字",
            "한",
            "ह",
            "ا",
            "0",
            "12",
            "1234",
            " ",
            "  ",
            "\t",
            "\n",
            "\r\n",
            "\u{a0}",
            "!",
            "...",
            "🙂",
            "👩\u{200d}💻",
            "'s",
            "<|open|>",
            "[BOS]",
        ];
        for (allow_special, expected_tokens, expected_hash) in [
            (false, 98_026usize, 0x11e1_3c50_d8d5_c7e8u64),
            (true, 84_294usize, 0x14d8_50fe_1958_bc7au64),
        ] {
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            let mut token_count = 0;
            for left in fragments {
                for middle in fragments {
                    for right in fragments {
                        let text = format!("{left}{middle}{right}");
                        let ids = tokenizer.encode(&text, allow_special).unwrap();
                        hash_u32(&mut hash, ids.len() as u32);
                        for &token in &ids {
                            hash_u32(&mut hash, token);
                        }
                        token_count += ids.len();
                    }
                }
            }
            assert_eq!(token_count, expected_tokens);
            assert_eq!(hash, expected_hash);
        }
    }

    #[test]
    fn safety_splitting_counts_unicode_scalars_like_python_strings() {
        assert_eq!(split_by_char_count("a🙂bc", 2), ["a🙂", "bc"]);
        assert_eq!(
            split_long_character_classes("abc defgh", 3),
            ["abc def", "gh"]
        );
        assert_eq!(split_long_character_classes("    ", 3), ["   ", " "]);
        assert!(is_python_whitespace('\u{1c}'));
        assert_eq!(split_long_character_classes("", 3), [""]);
    }

    #[test]
    fn tokenizer_can_be_shared_by_server_workers() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<K3Tokenizer>();
    }

    #[test]
    fn native_segment_batch_is_bit_exact_for_large_deterministic_fuzz() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        let fragments = [
            "alpha ",
            "漢字",
            "🙂",
            "\r\n",
            "1234567",
            "<|open|>",
            " e\u{301} ",
            "!?'s",
        ];
        let mut state = 0x9e37_79b9_u32;
        let mut owned = Vec::new();
        for row in 0..768 {
            let mut text = String::with_capacity(512);
            for _ in 0..64 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                text.push_str(fragments[state as usize % fragments.len()]);
            }
            if row % 7 == 0 {
                text.push_str("<|close|><|sep|>");
            }
            owned.push((text, row % 11 == 0));
        }
        let segments = owned
            .iter()
            .map(|(text, allow_special)| (text.as_str(), *allow_special))
            .collect::<Vec<_>>();
        let total_bytes = segments.iter().map(|(text, _)| text.len()).sum();
        if std::thread::available_parallelism().is_ok_and(|workers| workers.get() > 1) {
            assert!(tokenizer.should_parallelize_segments(segments.len(), total_bytes));
        }
        assert_eq!(
            tokenizer.encode_segments(&segments).unwrap(),
            tokenizer.encode_segments_sequential(&segments).unwrap()
        );
    }

    fn token_id_hash(token_ids: &[u32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for token in token_ids {
            hash_u32(&mut hash, *token);
        }
        hash
    }

    fn hash_u32(hash: &mut u64, value: u32) {
        for byte in value.to_le_bytes() {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
}
