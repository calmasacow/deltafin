//! Host-slab lifetime state for borrowed spine providers.
//!
//! Production CPU T=1 can borrow exact-BF16 matrices today; Metal and CUDA use
//! the same device-neutral protocol as their direct paths are qualified. A
//! Metal command-buffer event, CUDA event set, or CPU-immediate completion
//! implements one consume-once fence before Rust may recycle a reader-arena
//! allocation. Detached fallbacks retain the established V1 ownership rule.

use crate::error::{DeltafinError, Result};
use crate::provider::{
    NativeProviderSession, SpineLayerRetention, SpineSourceUse, SpineSourceUseToken,
};
use std::mem;

pub(crate) trait SpineSourceFence {
    fn source_use_session_identity(&self) -> u64;
    fn seal_source_use(&self, token: SpineSourceUseToken) -> Result<()>;
    fn try_reclaim_source_use(&self, token: SpineSourceUseToken) -> Result<bool>;
    fn abort_source_use(&self, token: SpineSourceUseToken) -> Result<()>;
}

impl SpineSourceFence for NativeProviderSession {
    fn source_use_session_identity(&self) -> u64 {
        self.identity()
    }

    fn seal_source_use(&self, token: SpineSourceUseToken) -> Result<()> {
        self.seal_spine_source_use(token)
    }

    fn try_reclaim_source_use(&self, token: SpineSourceUseToken) -> Result<bool> {
        self.try_reclaim_spine_source_use(token)
    }

    fn abort_source_use(&self, token: SpineSourceUseToken) -> Result<()> {
        self.abort_spine_source_use(token)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum BorrowState {
    Open,
    Sealed,
}

struct ActiveSourceUse<Lease> {
    token: SpineSourceUseToken,
    state: BorrowState,
    lease: Option<Lease>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ReclaimAdmission<Admission> {
    Pending,
    Reclaimed(Option<Admission>),
}

/// Owns at most one borrowed arena slot while a second slot may hold the next
/// layer read. Reclamation drops the old lease *before* invoking the supplied
/// non-blocking admission closure, so a two-slot arena cannot deadlock by
/// waiting for the very slot whose fence just completed.
pub(crate) struct SpineSourceUseController<Lease> {
    session_identity: Option<u64>,
    active: Option<ActiveSourceUse<Lease>>,
    highest_consumed_generation: u64,
    /// Sticky evidence that a provider may still own a source lease whose
    /// token was never safely admitted to `active`. The lease is already
    /// leaked; engine teardown must additionally retain the provider session.
    unproven_untracked_source: bool,
}

impl<Lease> Default for SpineSourceUseController<Lease> {
    fn default() -> Self {
        Self {
            session_identity: None,
            active: None,
            highest_consumed_generation: 0,
            unproven_untracked_source: false,
        }
    }
}

impl<Lease> SpineSourceUseController<Lease> {
    pub(crate) fn admit<Fence: SpineSourceFence>(
        &mut self,
        fence: &Fence,
        expected_generation: u64,
        retention: SpineLayerRetention,
        source_use: SpineSourceUse,
        lease: Lease,
    ) -> Result<()> {
        let session_identity = fence.source_use_session_identity();
        if session_identity == 0 || expected_generation == 0 {
            drop(lease);
            return Err(DeltafinError::new(
                "spine source-use admission needs a live session and generation",
            ));
        }
        if let Some(bound_identity) = self.session_identity
            && bound_identity != session_identity
        {
            match source_use {
                SpineSourceUse::Detached => drop(lease),
                SpineSourceUse::Borrowed(_) => {
                    self.unproven_untracked_source = true;
                    mem::forget(lease);
                }
            }
            return Err(DeltafinError::new(
                "spine source-use cannot cross provider sessions",
            ));
        }
        self.session_identity = Some(session_identity);

        match source_use {
            SpineSourceUse::Detached => {
                if self.active.is_some() {
                    drop(lease);
                    return Err(DeltafinError::new(
                        "spine source-use admission found an unreclaimed borrowed layer",
                    ));
                }
                // V1-compatible production path: release the arena slot at
                // precisely the same point as the original synchronous bind.
                drop(lease);
                Ok(())
            }
            SpineSourceUse::Borrowed(token) => {
                if token.session_identity != session_identity
                    || token.generation != expected_generation
                    || token.handle == 0
                    || token.generation <= self.highest_consumed_generation
                {
                    // The provider may still own this pointer, but the token
                    // cannot be addressed safely through this session. Leak
                    // the bounded lease rather than permit use-after-free.
                    self.unproven_untracked_source = true;
                    mem::forget(lease);
                    return Err(DeltafinError::new(
                        "spine provider returned a stale or cross-session borrowed source use",
                    ));
                }
                if self.active.is_some() {
                    // A second borrowed use cannot fit the two-slot schedule.
                    // Try to cancel the unpublished newcomer; if cancellation
                    // itself fails, retain its source forever (fail closed).
                    if fence.abort_source_use(token).is_ok() {
                        drop(lease);
                    } else {
                        self.unproven_untracked_source = true;
                        mem::forget(lease);
                    }
                    return Err(DeltafinError::new(
                        "spine source-use already owns the current arena slot",
                    ));
                }

                self.active = Some(ActiveSourceUse {
                    token,
                    state: BorrowState::Open,
                    lease: Some(lease),
                });
                if retention == SpineLayerRetention::Retained {
                    let abort = fence.abort_source_use(token);
                    if abort.is_ok() {
                        self.consume_active();
                    }
                    return Err(match abort {
                        Ok(()) => DeltafinError::new(
                            "a retained spine prefix may never borrow reader-arena storage",
                        ),
                        Err(error) => DeltafinError::new(format!(
                            "a retained spine prefix reported borrowed storage and abort failed: {error}"
                        )),
                    });
                }
                Ok(())
            }
        }
    }

    pub(crate) fn has_active_borrow(&self) -> bool {
        self.active.is_some()
    }

    pub(crate) fn active_generation(&self) -> Option<u64> {
        self.active.as_ref().map(|active| active.token.generation)
    }

    pub(crate) fn has_unproven_untracked_source(&self) -> bool {
        self.unproven_untracked_source
    }

    pub(crate) fn seal<Fence: SpineSourceFence>(
        &mut self,
        fence: &Fence,
        generation: u64,
    ) -> Result<()> {
        let active = self.checked_active(fence, generation)?;
        if active.state != BorrowState::Open {
            return Err(DeltafinError::new(
                "spine source-use seal is consume-once and was already sealed",
            ));
        }
        fence.seal_source_use(active.token)?;
        self.active
            .as_mut()
            .expect("checked source use must remain active")
            .state = BorrowState::Sealed;
        Ok(())
    }

    pub(crate) fn try_reclaim_then<Fence, Admission, Admit>(
        &mut self,
        fence: &Fence,
        generation: u64,
        admit_nonblocking: Admit,
    ) -> Result<ReclaimAdmission<Admission>>
    where
        Fence: SpineSourceFence,
        Admit: FnOnce() -> Result<Option<Admission>>,
    {
        let active = self.checked_active(fence, generation)?;
        if active.state != BorrowState::Sealed {
            return Err(DeltafinError::new(
                "spine source-use must be sealed before reclaim",
            ));
        }
        if !fence.try_reclaim_source_use(active.token)? {
            return Ok(ReclaimAdmission::Pending);
        }
        self.consume_active();
        // `consume_active` releases the old arena lease first. Admission must
        // remain non-blocking so a provider bug cannot strand the decode loop.
        Ok(ReclaimAdmission::Reclaimed(admit_nonblocking()?))
    }

    pub(crate) fn abort<Fence: SpineSourceFence>(
        &mut self,
        fence: &Fence,
        generation: u64,
    ) -> Result<()> {
        let token = self.checked_active(fence, generation)?.token;
        // Keep the lease live when abort fails. The caller may retry, and Drop
        // will deliberately leak it rather than race an unknown device use.
        fence.abort_source_use(token)?;
        self.consume_active();
        Ok(())
    }

    fn checked_active<Fence: SpineSourceFence>(
        &self,
        fence: &Fence,
        generation: u64,
    ) -> Result<&ActiveSourceUse<Lease>> {
        let active = self.active.as_ref().ok_or_else(|| {
            DeltafinError::new("spine source-use token is stale, consumed, or unknown")
        })?;
        if self.session_identity != Some(fence.source_use_session_identity())
            || active.token.session_identity != fence.source_use_session_identity()
        {
            return Err(DeltafinError::new(
                "spine source-use operation crossed provider sessions",
            ));
        }
        if generation == 0 || active.token.generation != generation {
            return Err(DeltafinError::new(
                "spine source-use operation named a stale generation",
            ));
        }
        Ok(active)
    }

    fn consume_active(&mut self) {
        let mut active = self
            .active
            .take()
            .expect("source-use consumption needs one active lease");
        self.highest_consumed_generation = self
            .highest_consumed_generation
            .max(active.token.generation);
        drop(active.lease.take());
    }
}

impl<Lease> Drop for SpineSourceUseController<Lease> {
    fn drop(&mut self) {
        if let Some(mut active) = self.active.take()
            && let Some(lease) = active.lease.take()
        {
            // Teardown without a confirmed abort/reclaim is an ownership
            // failure. Leaking one bounded slot is safer than recycling pages
            // that a device may still read. Session teardown owns the native
            // fence; process exit reclaims the leaked host allocation.
            mem::forget(lease);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct FakeFenceState {
        sealed: bool,
        parts: Vec<bool>,
        abort_fails: bool,
    }

    struct FakeFence {
        identity: u64,
        uses: RefCell<HashMap<u64, FakeFenceState>>,
    }

    impl FakeFence {
        fn new(identity: u64) -> Self {
            Self {
                identity,
                uses: RefCell::new(HashMap::new()),
            }
        }

        fn insert(&self, token: SpineSourceUseToken, parts: usize) {
            assert_eq!(token.session_identity, self.identity);
            assert!(
                self.uses
                    .borrow_mut()
                    .insert(
                        token.handle,
                        FakeFenceState {
                            sealed: false,
                            parts: vec![false; parts],
                            abort_fails: false,
                        },
                    )
                    .is_none()
            );
        }

        fn complete(&self, handle: u64, part: usize) {
            self.uses.borrow_mut().get_mut(&handle).unwrap().parts[part] = true;
        }

        fn fail_abort(&self, handle: u64) {
            self.uses.borrow_mut().get_mut(&handle).unwrap().abort_fails = true;
        }
    }

    impl SpineSourceFence for FakeFence {
        fn source_use_session_identity(&self) -> u64 {
            self.identity
        }

        fn seal_source_use(&self, token: SpineSourceUseToken) -> Result<()> {
            if token.session_identity != self.identity {
                return Err(DeltafinError::new("fake fence crossed sessions"));
            }
            let mut uses = self.uses.borrow_mut();
            let state = uses
                .get_mut(&token.handle)
                .ok_or_else(|| DeltafinError::new("fake fence is stale"))?;
            if state.sealed {
                return Err(DeltafinError::new("fake fence was sealed twice"));
            }
            state.sealed = true;
            Ok(())
        }

        fn try_reclaim_source_use(&self, token: SpineSourceUseToken) -> Result<bool> {
            let mut uses = self.uses.borrow_mut();
            let state = uses
                .get(&token.handle)
                .ok_or_else(|| DeltafinError::new("fake fence is stale"))?;
            if !state.sealed {
                return Err(DeltafinError::new("fake fence is not sealed"));
            }
            if state.parts.iter().all(|part| *part) {
                uses.remove(&token.handle);
                Ok(true)
            } else {
                Ok(false)
            }
        }

        fn abort_source_use(&self, token: SpineSourceUseToken) -> Result<()> {
            let mut uses = self.uses.borrow_mut();
            let state = uses
                .get(&token.handle)
                .ok_or_else(|| DeltafinError::new("fake fence is stale"))?;
            if state.abort_fails {
                return Err(DeltafinError::new("fake abort failed"));
            }
            uses.remove(&token.handle);
            Ok(())
        }
    }

    struct FakeLease {
        occupied: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }

    impl FakeLease {
        fn acquire(occupied: &Arc<AtomicUsize>, drops: &Arc<AtomicUsize>) -> Self {
            let previous = occupied.fetch_add(1, Ordering::SeqCst);
            assert!(previous < 2, "fake two-slot arena over-admitted");
            Self {
                occupied: Arc::clone(occupied),
                drops: Arc::clone(drops),
            }
        }
    }

    impl Drop for FakeLease {
        fn drop(&mut self) {
            self.occupied.fetch_sub(1, Ordering::SeqCst);
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn token(session_identity: u64, generation: u64, handle: u64) -> SpineSourceUseToken {
        SpineSourceUseToken {
            session_identity,
            generation,
            handle,
        }
    }

    #[test]
    fn detached_cpu_use_releases_its_slot_immediately() {
        let occupied = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let lease = FakeLease::acquire(&occupied, &drops);
        let fence = FakeFence::new(7);
        let mut controller = SpineSourceUseController::default();
        controller
            .admit(
                &fence,
                1,
                SpineLayerRetention::Transient,
                SpineSourceUse::Detached,
                lease,
            )
            .unwrap();
        assert_eq!(occupied.load(Ordering::SeqCst), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(!controller.has_active_borrow());
    }

    #[test]
    fn immediate_fence_reclaims_before_nonblocking_two_slot_admission() {
        let occupied = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let current = FakeLease::acquire(&occupied, &drops);
        let next = FakeLease::acquire(&occupied, &drops);
        let fence = FakeFence::new(11);
        let token = token(11, 4, 91);
        fence.insert(token, 0);
        let mut controller = SpineSourceUseController::default();
        controller
            .admit(
                &fence,
                4,
                SpineLayerRetention::Transient,
                SpineSourceUse::Borrowed(token),
                current,
            )
            .unwrap();
        controller.seal(&fence, 4).unwrap();
        let outcome = controller
            .try_reclaim_then(&fence, 4, || {
                assert_eq!(occupied.load(Ordering::SeqCst), 1);
                Ok(Some(FakeLease::acquire(&occupied, &drops)))
            })
            .unwrap();
        let ReclaimAdmission::Reclaimed(Some(admitted)) = outcome else {
            panic!("immediate fence did not reclaim and admit");
        };
        assert_eq!(occupied.load(Ordering::SeqCst), 2);
        drop(admitted);
        drop(next);
        assert_eq!(occupied.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn delayed_fence_keeps_pages_until_completion() {
        let occupied = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let lease = FakeLease::acquire(&occupied, &drops);
        let fence = FakeFence::new(13);
        let token = token(13, 2, 5);
        fence.insert(token, 1);
        let mut controller = SpineSourceUseController::default();
        controller
            .admit(
                &fence,
                2,
                SpineLayerRetention::Transient,
                SpineSourceUse::Borrowed(token),
                lease,
            )
            .unwrap();
        assert_eq!(controller.active_generation(), Some(2));
        controller.seal(&fence, 2).unwrap();
        assert_eq!(
            controller
                .try_reclaim_then(&fence, 2, || Ok(Some(())))
                .unwrap(),
            ReclaimAdmission::Pending
        );
        assert_eq!(occupied.load(Ordering::SeqCst), 1);
        fence.complete(token.handle, 0);
        assert_eq!(
            controller
                .try_reclaim_then(&fence, 2, || Ok(None::<()>))
                .unwrap(),
            ReclaimAdmission::Reclaimed(None)
        );
        assert_eq!(occupied.load(Ordering::SeqCst), 0);
        assert_eq!(controller.active_generation(), None);
    }

    #[test]
    fn failed_newcomer_abort_is_sticky_and_requires_provider_retention() {
        let occupied = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let current = FakeLease::acquire(&occupied, &drops);
        let newcomer = FakeLease::acquire(&occupied, &drops);
        let fence = FakeFence::new(15);
        let current_token = token(15, 3, 20);
        let newcomer_token = token(15, 4, 21);
        fence.insert(current_token, 1);
        fence.insert(newcomer_token, 1);
        fence.fail_abort(newcomer_token.handle);

        let mut controller = SpineSourceUseController::default();
        controller
            .admit(
                &fence,
                current_token.generation,
                SpineLayerRetention::Transient,
                SpineSourceUse::Borrowed(current_token),
                current,
            )
            .unwrap();
        let error = controller
            .admit(
                &fence,
                newcomer_token.generation,
                SpineLayerRetention::Transient,
                SpineSourceUse::Borrowed(newcomer_token),
                newcomer,
            )
            .unwrap_err();
        assert!(error.to_string().contains("already owns"));
        assert!(controller.has_unproven_untracked_source());
        assert_eq!(controller.active_generation(), Some(3));
        assert_eq!(occupied.load(Ordering::SeqCst), 2);

        // The tracked source can still be cancelled, but the failed newcomer
        // remains deliberately leaked and therefore keeps teardown unproven.
        controller.abort(&fence, current_token.generation).unwrap();
        assert_eq!(occupied.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(controller.has_unproven_untracked_source());
    }

    #[test]
    fn composite_fence_waits_for_every_device_use() {
        let occupied = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let lease = FakeLease::acquire(&occupied, &drops);
        let fence = FakeFence::new(17);
        let token = token(17, 8, 44);
        fence.insert(token, 3);
        let mut controller = SpineSourceUseController::default();
        controller
            .admit(
                &fence,
                8,
                SpineLayerRetention::Transient,
                SpineSourceUse::Borrowed(token),
                lease,
            )
            .unwrap();
        controller.seal(&fence, 8).unwrap();
        for part in 0..2 {
            fence.complete(token.handle, part);
            assert_eq!(
                controller
                    .try_reclaim_then(&fence, 8, || Ok(None::<()>))
                    .unwrap(),
                ReclaimAdmission::Pending
            );
        }
        fence.complete(token.handle, 2);
        assert_eq!(
            controller
                .try_reclaim_then(&fence, 8, || Ok(None::<()>))
                .unwrap(),
            ReclaimAdmission::Reclaimed(None)
        );
        assert_eq!(occupied.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn stale_cross_session_and_consume_once_operations_are_rejected() {
        let occupied = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let lease = FakeLease::acquire(&occupied, &drops);
        let first = FakeFence::new(19);
        let other = FakeFence::new(20);
        let token = token(19, 3, 9);
        first.insert(token, 0);
        let mut controller = SpineSourceUseController::default();
        controller
            .admit(
                &first,
                3,
                SpineLayerRetention::Transient,
                SpineSourceUse::Borrowed(token),
                lease,
            )
            .unwrap();
        assert!(controller.seal(&other, 3).is_err());
        assert!(controller.seal(&first, 2).is_err());
        controller.seal(&first, 3).unwrap();
        assert!(controller.seal(&first, 3).is_err());
        assert_eq!(
            controller
                .try_reclaim_then(&first, 3, || Ok(None::<()>))
                .unwrap(),
            ReclaimAdmission::Reclaimed(None)
        );
        assert!(
            controller
                .try_reclaim_then(&first, 3, || Ok(None::<()>))
                .is_err()
        );
    }

    #[test]
    fn retained_prefix_aborts_borrow_before_releasing_pages() {
        let occupied = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let lease = FakeLease::acquire(&occupied, &drops);
        let fence = FakeFence::new(23);
        let token = token(23, 1, 77);
        fence.insert(token, 1);
        let mut controller = SpineSourceUseController::default();
        let error = controller
            .admit(
                &fence,
                1,
                SpineLayerRetention::Retained,
                SpineSourceUse::Borrowed(token),
                lease,
            )
            .unwrap_err();
        assert!(error.to_string().contains("retained spine prefix"));
        assert_eq!(occupied.load(Ordering::SeqCst), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(!controller.has_active_borrow());
        assert!(!fence.uses.borrow().contains_key(&token.handle));
    }

    #[test]
    fn abort_failure_and_teardown_fail_closed() {
        let occupied = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let lease = FakeLease::acquire(&occupied, &drops);
        let fence = FakeFence::new(29);
        let token = token(29, 6, 12);
        fence.insert(token, 1);
        fence.fail_abort(token.handle);
        let mut controller = SpineSourceUseController::default();
        controller
            .admit(
                &fence,
                6,
                SpineLayerRetention::Transient,
                SpineSourceUse::Borrowed(token),
                lease,
            )
            .unwrap();
        assert!(controller.abort(&fence, 6).is_err());
        drop(controller);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(occupied.load(Ordering::SeqCst), 1);
    }
}
