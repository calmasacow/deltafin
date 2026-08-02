//! Exact, allocation-reusing route materialization for K3's top-16 MoE.
//!
//! The device mailbox supplies ordered expert IDs and raw float32 weight bits.
//! This module validates that snapshot transactionally, retains edge order and
//! weight bits verbatim, and builds a deterministic expert-to-storage-slot map
//! without Python lists, sets, dictionaries, or per-edge heap objects.

use crate::error::{DeltafinError, Result};

pub const ROUTED_EXPERTS: usize = 16;
pub const K3_EXPERTS: usize = 896;
const EXPERT_BITMAP_WORDS: usize = K3_EXPERTS.div_ceil(u64::BITS as usize);

#[derive(Debug)]
pub struct RouteArena {
    ordered_experts: Vec<u16>,
    ordered_weight_bits: Vec<u32>,
    unique_experts: Vec<u16>,
    edge_to_slot: Vec<u16>,
    expert_to_slot: Vec<u16>,
    max_positions: usize,
    positions: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct RouteView<'a> {
    pub positions: usize,
    pub ordered_experts: &'a [u16],
    pub ordered_weight_bits: &'a [u32],
    pub unique_experts: &'a [u16],
    pub edge_to_slot: &'a [u16],
}

impl RouteArena {
    pub fn new(max_positions: usize) -> Result<Self> {
        let edge_capacity = max_positions
            .checked_mul(ROUTED_EXPERTS)
            .ok_or_else(|| DeltafinError::new("route capacity overflows usize"))?;
        Ok(Self {
            ordered_experts: Vec::with_capacity(edge_capacity),
            ordered_weight_bits: Vec::with_capacity(edge_capacity),
            unique_experts: Vec::with_capacity(K3_EXPERTS.min(edge_capacity)),
            edge_to_slot: Vec::with_capacity(edge_capacity),
            expert_to_slot: vec![u16::MAX; K3_EXPERTS],
            max_positions,
            positions: 0,
        })
    }

    pub fn materialize(
        &mut self,
        positions: usize,
        ordered_experts: &[u16],
        ordered_weight_bits: &[u32],
    ) -> Result<RouteView<'_>> {
        if positions == 0 || positions > self.max_positions {
            return Err(DeltafinError::new(format!(
                "route snapshot has {positions} positions; valid capacity is 1..={}",
                self.max_positions
            )));
        }
        let edges = positions
            .checked_mul(ROUTED_EXPERTS)
            .ok_or_else(|| DeltafinError::new("route edge count overflows usize"))?;
        if ordered_experts.len() != edges || ordered_weight_bits.len() != edges {
            return Err(DeltafinError::new(format!(
                "route snapshot needs {edges} IDs and weights for {positions} positions; got {} IDs and {} weights",
                ordered_experts.len(),
                ordered_weight_bits.len()
            )));
        }

        // Validate the complete candidate before touching the published arena.
        // A corrupt mailbox therefore cannot partially replace the last valid
        // route while its expert buffers may still be leased by a backend.
        let mut selected = [0_u64; EXPERT_BITMAP_WORDS];
        for position in 0..positions {
            let row = &ordered_experts[position * ROUTED_EXPERTS..(position + 1) * ROUTED_EXPERTS];
            let mut row_selected = [0_u64; EXPERT_BITMAP_WORDS];
            for (column, &expert) in row.iter().enumerate() {
                let expert = usize::from(expert);
                if expert >= K3_EXPERTS {
                    return Err(DeltafinError::new(format!(
                        "route position {position} column {column} has out-of-range expert {expert}"
                    )));
                }
                let word = expert / u64::BITS as usize;
                let bit = 1_u64 << (expert % u64::BITS as usize);
                if row_selected[word] & bit != 0 {
                    return Err(DeltafinError::new(format!(
                        "route position {position} repeats expert {expert}"
                    )));
                }
                row_selected[word] |= bit;
                selected[word] |= bit;
            }
        }
        for (edge, &bits) in ordered_weight_bits.iter().enumerate() {
            let weight = f32::from_bits(bits);
            if !weight.is_finite() || weight < 0.0 {
                return Err(DeltafinError::new(format!(
                    "route edge {edge} has invalid weight bits 0x{bits:08x}"
                )));
            }
        }

        self.ordered_experts.clear();
        self.ordered_experts.extend_from_slice(ordered_experts);
        self.ordered_weight_bits.clear();
        self.ordered_weight_bits
            .extend_from_slice(ordered_weight_bits);
        self.unique_experts.clear();
        self.edge_to_slot.clear();
        self.edge_to_slot.resize(edges, u16::MAX);
        self.positions = positions;

        // Fourteen bitmap words replace a 896-entry generation scan while
        // preserving the established ascending fetch order exactly. This runs
        // for every routed layer of every target position, so even this small
        // host-side reduction belongs off the hot path.
        for (word_index, mut word) in selected.into_iter().enumerate() {
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let expert = word_index * u64::BITS as usize + bit;
                let slot = u16::try_from(self.unique_experts.len())
                    .map_err(|_| DeltafinError::new("route slot index exceeds u16"))?;
                self.unique_experts.push(expert as u16);
                self.expert_to_slot[expert] = slot;
                word &= word - 1;
            }
        }
        for (edge, &expert) in ordered_experts.iter().enumerate() {
            let slot = self.expert_to_slot[usize::from(expert)];
            if slot == u16::MAX {
                return Err(DeltafinError::new(
                    "internal route slot publication was incomplete",
                ));
            }
            self.edge_to_slot[edge] = slot;
        }
        Ok(self.view())
    }

    pub fn view(&self) -> RouteView<'_> {
        RouteView {
            positions: self.positions,
            ordered_experts: &self.ordered_experts,
            ordered_weight_bits: &self.ordered_weight_bits,
            unique_experts: &self.unique_experts,
            edge_to_slot: &self.edge_to_slot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(first: u16) -> Vec<u16> {
        (first..first + ROUTED_EXPERTS as u16).collect()
    }

    #[test]
    fn preserves_edges_and_weight_bits_while_deduplicating_storage() {
        let first = row(32);
        let mut second = row(40);
        second.reverse();
        let ids: Vec<u16> = first.into_iter().chain(second).collect();
        let weights: Vec<u32> = (0..ids.len())
            .map(|index| (0.001_f32 * (index + 1) as f32).to_bits())
            .collect();
        let mut arena = RouteArena::new(2).unwrap();
        let route = arena.materialize(2, &ids, &weights).unwrap();

        assert_eq!(route.positions, 2);
        assert_eq!(route.ordered_experts, ids);
        assert_eq!(route.ordered_weight_bits, weights);
        assert_eq!(route.unique_experts, &(32_u16..56).collect::<Vec<_>>());
        for (edge, expert) in ids.iter().enumerate() {
            assert_eq!(
                route.unique_experts[route.edge_to_slot[edge] as usize],
                *expert
            );
        }
    }

    #[test]
    fn validation_failure_leaves_the_previous_route_published() {
        let ids = row(0);
        let weights = vec![1.0_f32.to_bits(); ROUTED_EXPERTS];
        let mut arena = RouteArena::new(1).unwrap();
        arena.materialize(1, &ids, &weights).unwrap();

        let mut invalid = ids.clone();
        invalid[7] = invalid[0];
        assert!(arena.materialize(1, &invalid, &weights).is_err());
        assert_eq!(arena.view().ordered_experts, ids);

        let mut invalid_weights = weights.clone();
        invalid_weights[3] = f32::NAN.to_bits();
        assert!(arena.materialize(1, &ids, &invalid_weights).is_err());
        assert_eq!(arena.view().ordered_weight_bits, weights);
    }

    #[test]
    fn validates_shape_and_expert_universe() {
        let weights = vec![0.5_f32.to_bits(); ROUTED_EXPERTS];
        let mut arena = RouteArena::new(1).unwrap();
        assert!(arena.materialize(0, &[], &[]).is_err());
        assert!(arena.materialize(1, &row(0)[..15], &weights).is_err());
        let mut invalid = row(0);
        invalid[15] = K3_EXPERTS as u16;
        assert!(arena.materialize(1, &invalid, &weights).is_err());
        let two_rows: Vec<u16> = row(0).into_iter().chain(row(32)).collect();
        let two_weights = vec![0.5_f32.to_bits(); ROUTED_EXPERTS * 2];
        assert!(arena.materialize(2, &two_rows, &two_weights).is_err());
    }

    #[test]
    fn stale_slot_entries_cannot_republish_unselected_experts() {
        let mut arena = RouteArena::new(1).unwrap();
        arena.expert_to_slot[700] = 0;
        let ids = row(100);
        let weights = vec![0.25_f32.to_bits(); ROUTED_EXPERTS];
        let route = arena.materialize(1, &ids, &weights).unwrap();
        assert_eq!(route.unique_experts, ids);
        assert!(!route.unique_experts.contains(&700));
    }
}
