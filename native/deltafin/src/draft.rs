//! Native proposal sources for exact full-K3 verification.
//!
//! A proposal source can only return untrusted token IDs. It has no output
//! sink, provider-tail handle, or cache-commit capability; the target engine
//! must compare every ID with full K3 before anything becomes visible.

use crate::error::Result;

pub trait DraftSource {
    fn propose(&mut self, target_history: &[u32], maximum: usize) -> Result<Box<[u32]>>;
}

/// Allocation-bounded continuation lookup from repeated target history.
///
/// This is the compiled equivalent of the mature Python `ngram_draft`: search
/// the longest suffix first, choose the most recent earlier occurrence, append
/// its following token, and repeat against the extended speculative context.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct NgramDraftSource {
    pub minimum_n: usize,
    pub maximum_n: usize,
}

impl Default for NgramDraftSource {
    fn default() -> Self {
        Self {
            minimum_n: 2,
            maximum_n: 6,
        }
    }
}

impl NgramDraftSource {
    pub fn new(minimum_n: usize, maximum_n: usize) -> Result<Self> {
        if minimum_n == 0 || maximum_n < minimum_n {
            return Err(crate::error::DeltafinError::new(
                "n-gram draft bounds must satisfy 1 <= minimum <= maximum",
            ));
        }
        Ok(Self {
            minimum_n,
            maximum_n,
        })
    }

    fn next(&self, history: &[u32]) -> Option<u32> {
        if history.len() <= self.minimum_n {
            return None;
        }
        let largest = self.maximum_n.min(history.len().saturating_sub(1));
        for width in (self.minimum_n..=largest).rev() {
            let suffix_start = history.len() - width;
            let suffix = &history[suffix_start..];
            // `candidate + width` must name a following token and must lie
            // strictly before the suffix itself. Reverse iteration matches the
            // mature path's most-recent-earlier-occurrence policy.
            for candidate in (0..suffix_start).rev() {
                let following = candidate + width;
                if following < history.len() && &history[candidate..following] == suffix {
                    return Some(history[following]);
                }
            }
        }
        None
    }
}

impl DraftSource for NgramDraftSource {
    fn propose(&mut self, target_history: &[u32], maximum: usize) -> Result<Box<[u32]>> {
        if maximum == 0 {
            return Ok(Box::new([]));
        }
        let mut context = Vec::with_capacity(target_history.len().saturating_add(maximum));
        context.extend_from_slice(target_history);
        let mut drafts = Vec::with_capacity(maximum);
        while drafts.len() < maximum {
            let Some(token) = self.next(&context) else {
                break;
            };
            drafts.push(token);
            context.push(token);
        }
        Ok(drafts.into_boxed_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_recent_suffix_matches_the_mature_lookup_order() {
        let source = NgramDraftSource::new(2, 6).unwrap();
        // The final [1,2,3] has two earlier matches. Reverse search chooses the
        // later continuation 8, while longest-first rejects the shorter [2,3]
        // continuation 9.
        let history = [1, 2, 3, 7, 1, 2, 3, 8, 2, 3, 9, 1, 2, 3];
        assert_eq!(source.next(&history), Some(8));
    }

    #[test]
    fn proposal_extends_its_own_untrusted_context_without_authoring_output() {
        let mut source = NgramDraftSource::new(2, 6).unwrap();
        let history = [4, 5, 6, 7, 4, 5];
        assert_eq!(&*source.propose(&history, 4).unwrap(), &[6, 7, 4, 5]);
        assert!(source.propose(&history, 0).unwrap().is_empty());
    }

    #[test]
    fn absent_repeat_yields_no_draft() {
        let mut source = NgramDraftSource::default();
        assert!(source.propose(&[1, 2, 3, 4], 8).unwrap().is_empty());
    }
}
