//! Allocation-reusing exact target/draft arbitration.
//!
//! Draft models are scheduling hints only.  This state machine accepts a
//! candidate token solely when the full target's prediction at that position
//! agrees; the mismatch (or all-accepted bonus) always comes from full K3.

use crate::error::{DeltafinError, Result};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DECODE_ARENA_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum StopReason {
    Eos,
    MaxNew,
    ContextFull,
    /// The CLI observed SIGINT between complete target transactions. Every
    /// token returned before this stop remains full-target authoritative.
    Interrupted,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct VerificationResult<'a> {
    pub emitted: &'a [u32],
    pub accepted_drafts: usize,
    pub stop: Option<StopReason>,
}

/// An immutable, full-target-authoritative decode decision.
///
/// Planning is deliberately separate from applying so a provider transaction
/// can be cancelled or rerun without publishing logical token state.  A plan
/// is bound to the exact arena revision that produced it; applying it after
/// any other successful decode transaction fails before mutation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct VerificationPlan {
    emitted: Box<[u32]>,
    accepted_drafts: usize,
    stop: Option<StopReason>,
    arena_id: u64,
    arena_revision: u64,
    history_len: usize,
}

impl VerificationPlan {
    pub fn emitted(&self) -> &[u32] {
        &self.emitted
    }

    pub const fn accepted_drafts(&self) -> usize {
        self.accepted_drafts
    }

    pub const fn stop(&self) -> Option<StopReason> {
        self.stop
    }
}

#[derive(Debug)]
pub struct DecodeArena {
    history: Vec<u32>,
    emitted: Vec<u32>,
    prompt_tokens: usize,
    maximum_context: usize,
    maximum_new: Option<usize>,
    eos_token: u32,
    stopped: Option<StopReason>,
    arena_id: u64,
    revision: u64,
}

impl DecodeArena {
    pub fn new(
        prompt: &[u32],
        maximum_context: usize,
        maximum_new: Option<usize>,
        eos_token: u32,
        maximum_draft: usize,
    ) -> Result<Self> {
        if prompt.is_empty() {
            return Err(DeltafinError::new("the target prompt cannot be empty"));
        }
        if prompt.len() > maximum_context {
            return Err(DeltafinError::new(format!(
                "prompt has {} tokens but the target context limit is {maximum_context}",
                prompt.len()
            )));
        }
        let remaining = maximum_context - prompt.len();
        let history_capacity = prompt
            .len()
            .checked_add(maximum_new.unwrap_or(remaining).min(remaining))
            .ok_or_else(|| DeltafinError::new("decode history capacity overflows usize"))?;
        let mut history = Vec::with_capacity(history_capacity);
        history.extend_from_slice(prompt);
        Ok(Self {
            history,
            emitted: Vec::with_capacity(maximum_draft.saturating_add(1)),
            prompt_tokens: prompt.len(),
            maximum_context,
            maximum_new,
            eos_token,
            stopped: if remaining == 0 {
                Some(StopReason::ContextFull)
            } else if maximum_new == Some(0) {
                Some(StopReason::MaxNew)
            } else {
                None
            },
            arena_id: NEXT_DECODE_ARENA_ID.fetch_add(1, Ordering::Relaxed),
            revision: 0,
        })
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }

    pub fn generated(&self) -> usize {
        self.history.len() - self.prompt_tokens
    }

    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stopped
    }

    /// Commit one ordinary target prediction.
    pub fn commit_target(&mut self, target_token: u32) -> Result<VerificationResult<'_>> {
        self.commit_verified(&[], &[target_token])
    }

    /// Commit one full-target verification transaction.
    ///
    /// `target_tokens[i]` is the full target prediction after the prefix ending
    /// immediately before `draft_tokens[i]`.  `target_tokens[draft.len()]` is
    /// the target's bonus prediction and is required even when a prior draft
    /// mismatches; this fixed shape lets the provider run one coarse tape.
    pub fn commit_verified(
        &mut self,
        draft_tokens: &[u32],
        target_tokens: &[u32],
    ) -> Result<VerificationResult<'_>> {
        let plan = self.plan_verified(draft_tokens, target_tokens)?;
        self.apply_verified(plan)
    }

    /// Plan one full-target verification transaction without publishing it.
    ///
    /// The returned authoritative IDs can safely be used to validate a
    /// shorter provider rerun. Neither history, stop state, nor the arena's
    /// reusable output storage changes until `apply_verified` succeeds.
    pub fn plan_verified(
        &self,
        draft_tokens: &[u32],
        target_tokens: &[u32],
    ) -> Result<VerificationPlan> {
        if let Some(reason) = self.stopped {
            return Err(DeltafinError::new(format!(
                "decode is already stopped ({reason:?})"
            )));
        }
        if target_tokens.len() != draft_tokens.len().saturating_add(1) {
            return Err(DeltafinError::new(format!(
                "target verification must provide draft+1 predictions; got {} drafts and {} target tokens",
                draft_tokens.len(),
                target_tokens.len()
            )));
        }

        let accepted = draft_tokens
            .iter()
            .zip(target_tokens)
            .take_while(|(draft, target)| draft == target)
            .count();

        // Accepted drafts are target-confirmed. The following token is either
        // the target's mismatch correction or its all-accepted bonus.
        let mut emitted = Vec::with_capacity(accepted.saturating_add(1));
        let mut generated = self.generated();
        let mut history_len = self.history.len();
        let mut stop = None;
        for &token in &draft_tokens[..accepted] {
            if !plan_authoritative(
                &mut emitted,
                &mut generated,
                &mut history_len,
                &mut stop,
                token,
                self.maximum_context,
                self.maximum_new,
                self.eos_token,
            ) {
                break;
            }
        }
        if stop.is_none() {
            plan_authoritative(
                &mut emitted,
                &mut generated,
                &mut history_len,
                &mut stop,
                target_tokens[accepted],
                self.maximum_context,
                self.maximum_new,
                self.eos_token,
            );
        }
        let accepted_drafts = accepted.min(emitted.len());
        Ok(VerificationPlan {
            emitted: emitted.into_boxed_slice(),
            accepted_drafts,
            stop,
            arena_id: self.arena_id,
            arena_revision: self.revision,
            history_len: self.history.len(),
        })
    }

    /// Publish a previously planned verification decision.
    ///
    /// Every fallible validation happens before mutation. Once the stale-plan
    /// guard passes, applying the already-bounded token slice is an infallible
    /// logical state transition.
    pub fn apply_verified(&mut self, plan: VerificationPlan) -> Result<VerificationResult<'_>> {
        if plan.arena_id != self.arena_id
            || plan.arena_revision != self.revision
            || plan.history_len != self.history.len()
            || self.stopped.is_some()
        {
            return Err(DeltafinError::new(
                "verification plan is stale or belongs to a different decode arena",
            ));
        }

        self.emitted.clear();
        self.emitted.extend_from_slice(&plan.emitted);
        self.history.extend_from_slice(&plan.emitted);
        self.stopped = plan.stop;
        self.revision = self.revision.wrapping_add(1);
        Ok(VerificationResult {
            emitted: &self.emitted,
            accepted_drafts: plan.accepted_drafts,
            stop: self.stopped,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn plan_authoritative(
    emitted: &mut Vec<u32>,
    generated: &mut usize,
    history_len: &mut usize,
    stopped: &mut Option<StopReason>,
    token: u32,
    maximum_context: usize,
    maximum_new: Option<usize>,
    eos_token: u32,
) -> bool {
    if *history_len >= maximum_context {
        *stopped = Some(StopReason::ContextFull);
        return false;
    }
    if maximum_new.is_some_and(|maximum| *generated >= maximum) {
        *stopped = Some(StopReason::MaxNew);
        return false;
    }
    emitted.push(token);
    *history_len += 1;
    *generated += 1;
    if token == eos_token {
        *stopped = Some(StopReason::Eos);
    } else if *history_len == maximum_context {
        *stopped = Some(StopReason::ContextFull);
    } else if maximum_new.is_some_and(|maximum| *generated == maximum) {
        *stopped = Some(StopReason::MaxNew);
    }
    stopped.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_can_never_author_a_mismatching_token() {
        let mut arena = DecodeArena::new(&[10, 11], 100, None, 999, 8).unwrap();
        let result = arena
            .commit_verified(&[20, 21, 22], &[20, 71, 72, 73])
            .unwrap();
        assert_eq!(result.accepted_drafts, 1);
        assert_eq!(result.emitted, &[20, 71]);
        assert_eq!(arena.history(), &[10, 11, 20, 71]);
        assert!(!arena.history().contains(&21));
        assert!(!arena.history().contains(&22));
    }

    #[test]
    fn all_accepted_path_emits_the_target_bonus() {
        let mut arena = DecodeArena::new(&[1], 100, None, 999, 4).unwrap();
        let result = arena.commit_verified(&[2, 3], &[2, 3, 4]).unwrap();
        assert_eq!(result.accepted_drafts, 2);
        assert_eq!(result.emitted, &[2, 3, 4]);
        assert_eq!(arena.history(), &[1, 2, 3, 4]);
    }

    #[test]
    fn eos_and_limits_cut_a_wide_verification_transaction_safely() {
        let mut eos = DecodeArena::new(&[1], 100, None, 3, 8).unwrap();
        let result = eos.commit_verified(&[2, 3, 4], &[2, 3, 4, 5]).unwrap();
        assert_eq!(result.emitted, &[2, 3]);
        assert_eq!(result.accepted_drafts, 2);
        assert_eq!(result.stop, Some(StopReason::Eos));

        let mut maximum = DecodeArena::new(&[1], 100, Some(2), 999, 8).unwrap();
        let result = maximum.commit_verified(&[2, 3, 4], &[2, 3, 4, 5]).unwrap();
        assert_eq!(result.emitted, &[2, 3]);
        assert_eq!(result.stop, Some(StopReason::MaxNew));

        let mut context = DecodeArena::new(&[1, 2], 4, None, 999, 8).unwrap();
        let result = context.commit_verified(&[3, 4, 5], &[3, 4, 5, 6]).unwrap();
        assert_eq!(result.emitted, &[3, 4]);
        assert_eq!(result.stop, Some(StopReason::ContextFull));
    }

    #[test]
    fn invalid_transaction_is_rejected_without_mutating_history() {
        let mut arena = DecodeArena::new(&[1, 2], 100, None, 999, 8).unwrap();
        assert!(arena.commit_verified(&[3, 4], &[3, 4]).is_err());
        assert_eq!(arena.history(), &[1, 2]);
    }

    #[test]
    fn ordinary_decode_is_the_zero_draft_case() {
        let mut arena = DecodeArena::new(&[1], 100, None, 999, 0).unwrap();
        assert_eq!(arena.commit_target(7).unwrap().emitted, &[7]);
        assert_eq!(arena.history(), &[1, 7]);
    }

    #[test]
    fn immutable_plan_covers_zero_mid_and_full_acceptance_before_apply() {
        let cases: &[(&[u32], &[u32], &[u32], usize)] = &[
            (&[], &[9], &[9], 0),
            (&[2, 3, 4], &[8, 9, 10, 11], &[8], 0),
            (&[2, 3, 4], &[2, 3, 9, 10], &[2, 3, 9], 2),
            (&[2, 3, 4], &[2, 3, 4, 5], &[2, 3, 4, 5], 3),
        ];
        for &(drafts, targets, expected, accepted) in cases {
            let mut arena = DecodeArena::new(&[1], 100, None, 999, 8).unwrap();
            let before = arena.history().to_vec();
            let plan = arena.plan_verified(drafts, targets).unwrap();
            assert_eq!(arena.history(), before, "planning must not publish history");
            assert_eq!(arena.stop_reason(), None);
            assert_eq!(plan.emitted(), expected);
            assert_eq!(plan.accepted_drafts(), accepted);
            assert_eq!(plan.stop(), None);

            let applied = arena.apply_verified(plan).unwrap();
            assert_eq!(applied.emitted, expected);
            assert_eq!(applied.accepted_drafts, accepted);
            assert_eq!(&arena.history()[1..], expected);
        }
    }

    #[test]
    fn plan_stops_on_eos_from_an_accepted_draft_or_target_correction() {
        let mut draft_eos = DecodeArena::new(&[1], 100, None, 3, 8).unwrap();
        let plan = draft_eos.plan_verified(&[2, 3, 4], &[2, 3, 4, 5]).unwrap();
        assert_eq!(plan.emitted(), &[2, 3]);
        assert_eq!(plan.accepted_drafts(), 2);
        assert_eq!(plan.stop(), Some(StopReason::Eos));
        assert_eq!(draft_eos.history(), &[1]);
        assert_eq!(
            draft_eos.apply_verified(plan).unwrap().stop,
            Some(StopReason::Eos)
        );

        let mut correction_eos = DecodeArena::new(&[1], 100, None, 7, 8).unwrap();
        let plan = correction_eos
            .plan_verified(&[2, 3, 4], &[2, 7, 8, 9])
            .unwrap();
        assert_eq!(plan.emitted(), &[2, 7]);
        assert_eq!(plan.accepted_drafts(), 1);
        assert_eq!(plan.stop(), Some(StopReason::Eos));
        assert_eq!(correction_eos.history(), &[1]);
        correction_eos.apply_verified(plan).unwrap();
        assert_eq!(correction_eos.history(), &[1, 2, 7]);
    }

    #[test]
    fn planned_limits_match_committed_max_new_and_context_boundaries() {
        let mut maximum = DecodeArena::new(&[1], 100, Some(2), 999, 8).unwrap();
        let max_plan = maximum.plan_verified(&[2, 3, 4], &[2, 3, 4, 5]).unwrap();
        assert_eq!(max_plan.emitted(), &[2, 3]);
        assert_eq!(max_plan.stop(), Some(StopReason::MaxNew));
        assert_eq!(maximum.history(), &[1]);
        let max_result = maximum.apply_verified(max_plan).unwrap();
        assert_eq!(max_result.emitted, &[2, 3]);
        assert_eq!(max_result.stop, Some(StopReason::MaxNew));

        let mut context = DecodeArena::new(&[1, 2], 4, None, 999, 8).unwrap();
        let context_plan = context.plan_verified(&[3, 4, 5], &[3, 4, 5, 6]).unwrap();
        assert_eq!(context_plan.emitted(), &[3, 4]);
        assert_eq!(context_plan.stop(), Some(StopReason::ContextFull));
        assert_eq!(context.history(), &[1, 2]);
        let context_result = context.apply_verified(context_plan).unwrap();
        assert_eq!(context_result.emitted, &[3, 4]);
        assert_eq!(context_result.stop, Some(StopReason::ContextFull));
    }

    #[test]
    fn stale_or_foreign_plan_fails_before_any_mutation() {
        let mut arena = DecodeArena::new(&[1], 100, None, 999, 8).unwrap();
        let stale = arena.plan_verified(&[2], &[2, 3]).unwrap();
        arena.commit_target(9).unwrap();
        let after_commit = arena.history().to_vec();
        assert!(arena.apply_verified(stale).is_err());
        assert_eq!(arena.history(), after_commit);

        let foreign = DecodeArena::new(&[1, 9], 100, None, 999, 8)
            .unwrap()
            .plan_verified(&[4], &[4, 5])
            .unwrap();
        let before_foreign = arena.history().to_vec();
        assert!(arena.apply_verified(foreign).is_err());
        assert_eq!(arena.history(), before_foreign);
    }
}
