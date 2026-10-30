//! HNSW graph over L2-normalized vectors stored in one contiguous slab.
//!
//! Nodes are addressed by `u32` slots; freed slots are reused. Mutating
//! methods take `&mut self`; searches take `&self` and may run concurrently.

use crate::distance::distance;
use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq)]
struct Ord32(f32);
impl Eq for Ord32 {}
impl PartialOrd for Ord32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ord32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

type Item = (Ord32, u32);

/// Hard cap on node level. With the default multiplier the chance of a level
/// above 64 is below 2^-64, so this only ever rejects corrupt input.
pub const MAX_LEVEL: usize = 64;

thread_local! {
    /// Epoch-stamped visited set: clearing is O(1) instead of O(n) per query.
    static VISITED: RefCell<(Vec<u32>, u32)> = const { RefCell::new((Vec::new(), 0)) };
}

fn with_visited<R>(n: usize, f: impl FnOnce(&mut dyn FnMut(u32) -> bool) -> R) -> R {
    VISITED.with(|cell| {
        let mut state = cell.borrow_mut();
        let (marks, epoch) = &mut *state;
        if marks.len() < n {
            marks.resize(n, 0);
        }
        *epoch = epoch.wrapping_add(1);
        if *epoch == 0 {
            marks.iter_mut().for_each(|m| *m = 0);
            *epoch = 1;
        }
        let current = *epoch;
        // Returns true if `slot` was already visited.
        let mut visit = |slot: u32| {
            let seen = marks[slot as usize] == current;
            marks[slot as usize] = current;
            seen
        };
        f(&mut visit)
    })
}

pub struct Params {
    pub dim: usize,
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ml: Option<usize>,
    level_mult: f64,
}

pub struct Index {
    pub p: Params,
    vecs: Vec<f32>,
    /// Level of each slot, or -1 when the slot is free.
    levels: Vec<i32>,
    links: Vec<Vec<Vec<u32>>>,
    free: Vec<u32>,
    ep: Option<u32>,
    count: usize,
    rng: u64,
}

/// Search-time constraints applied while traversing layer 0.
pub struct Filter<'a> {
    pub predicate: Option<&'a dyn Fn(u32) -> bool>,
    pub max_distance: f32,
    pub max_visits: usize,
}

pub struct Stats {
    pub nodes: usize,
    pub levels: usize,
    pub avg_degree0: f64,
    pub isolated0: usize,
    pub free_slots: usize,
}

impl Index {
    pub fn new(dim: usize, m: usize, ef_construction: usize, ml: Option<usize>, seed: u64) -> Self {
        Index {
            p: Params {
                dim,
                m,
                m_max0: 2 * m,
                ef_construction,
                ml,
                level_mult: 1.0 / (m as f64).ln(),
            },
            vecs: Vec::new(),
            levels: Vec::new(),
            links: Vec::new(),
            free: Vec::new(),
            ep: None,
            count: 0,
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn slots(&self) -> usize {
        self.levels.len()
    }

    pub fn level_of(&self, slot: u32) -> Option<usize> {
        self.levels.get(slot as usize).and_then(|l| usize::try_from(*l).ok())
    }

    pub fn entry_point(&self) -> Option<u32> {
        self.ep
    }

    fn vec(&self, slot: u32) -> &[f32] {
        let start = slot as usize * self.p.dim;
        &self.vecs[start..start + self.p.dim]
    }

    fn dist(&self, a: u32, b: u32) -> f32 {
        distance(self.vec(a), self.vec(b))
    }

    fn limit(&self, level: usize) -> usize {
        if level == 0 {
            self.p.m_max0
        } else {
            self.p.m
        }
    }

    fn next_f64(&mut self) -> f64 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (r >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn random_level(&mut self) -> usize {
        let u = self.next_f64().max(f64::MIN_POSITIVE);
        let level = (-u.ln() * self.p.level_mult) as usize;
        let level = level.min(MAX_LEVEL);
        match self.p.ml {
            Some(cap) => level.min(cap),
            None => level,
        }
    }

    // ---- search -----------------------------------------------------------------------

    fn greedy(&self, q: &[f32], mut entry: u32, level: usize) -> u32 {
        let mut best = distance(q, self.vec(entry));
        loop {
            let mut moved = false;
            for &nb in &self.links[entry as usize][level] {
                let d = distance(q, self.vec(nb));
                if d < best {
                    best = d;
                    entry = nb;
                    moved = true;
                }
            }
            if !moved {
                return entry;
            }
        }
    }

    fn search_layer(&self, q: &[f32], ep: u32, ef: usize, level: usize) -> Vec<(f32, u32)> {
        with_visited(self.levels.len(), |visit| {
            let d0 = distance(q, self.vec(ep));
            let mut candidates: BinaryHeap<Reverse<Item>> = BinaryHeap::new();
            let mut results: BinaryHeap<Item> = BinaryHeap::new();
            candidates.push(Reverse((Ord32(d0), ep)));
            results.push((Ord32(d0), ep));
            visit(ep);
            while let Some(Reverse((cd, c))) = candidates.pop() {
                if results.len() >= ef && cd.0 > results.peek().unwrap().0 .0 {
                    break;
                }
                for &nb in &self.links[c as usize][level] {
                    if visit(nb) {
                        continue;
                    }
                    let d = distance(q, self.vec(nb));
                    if results.len() < ef || d < results.peek().unwrap().0 .0 {
                        candidates.push(Reverse((Ord32(d), nb)));
                        results.push((Ord32(d), nb));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
            let mut out: Vec<(f32, u32)> = results.into_iter().map(|(d, s)| (d.0, s)).collect();
            out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            out
        })
    }

    /// Layer-0 search where only qualifying nodes count toward the result set.
    /// Returns `(results, exact_fallback_used)`.
    fn search_filtered(
        &self,
        q: &[f32],
        entry: u32,
        k: usize,
        ef: usize,
        filter: &Filter,
    ) -> (Vec<(f32, u32)>, bool) {
        let qualifies = |slot: u32, d: f32| {
            d <= filter.max_distance && filter.predicate.is_none_or(|p| p(slot))
        };
        let (mut results_out, exhausted) = with_visited(self.levels.len(), |visit| {
            let d0 = distance(q, self.vec(entry));
            let mut candidates: BinaryHeap<Reverse<Item>> = BinaryHeap::new();
            let mut results: BinaryHeap<Item> = BinaryHeap::new();
            candidates.push(Reverse((Ord32(d0), entry)));
            if qualifies(entry, d0) {
                results.push((Ord32(d0), entry));
            }
            visit(entry);
            let mut visited = 1usize;
            let mut exhausted = false;
            while let Some(Reverse((cd, c))) = candidates.pop() {
                if results.len() >= ef && cd.0 > results.peek().unwrap().0 .0 {
                    break;
                }
                for &nb in &self.links[c as usize][0] {
                    if visit(nb) {
                        continue;
                    }
                    visited += 1;
                    let d = distance(q, self.vec(nb));
                    if results.len() < ef || d < results.peek().unwrap().0 .0 {
                        candidates.push(Reverse((Ord32(d), nb)));
                        if qualifies(nb, d) {
                            results.push((Ord32(d), nb));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
                if visited >= filter.max_visits && results.len() < k {
                    exhausted = true;
                    break;
                }
            }
            let out: Vec<(f32, u32)> = results.into_iter().map(|(d, s)| (d.0, s)).collect();
            (out, exhausted)
        });
        if exhausted {
            let mut all: Vec<(f32, u32)> = (0..self.levels.len() as u32)
                .filter(|&s| self.levels[s as usize] >= 0)
                .map(|s| (distance(q, self.vec(s)), s))
                .filter(|&(d, s)| qualifies(s, d))
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            all.truncate(k);
            return (all, true);
        }
        results_out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        results_out.truncate(k);
        (results_out, false)
    }

    /// Approximate top-`k` as `(slot, cosine similarity)`, best first.
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&Filter>,
    ) -> (Vec<(u32, f32)>, bool) {
        let Some(mut entry) = self.ep else {
            return (Vec::new(), false);
        };
        let top = self.levels[entry as usize] as usize;
        for level in (1..=top).rev() {
            entry = self.greedy(q, entry, level);
        }
        let ef = ef.max(k);
        let (hits, fallback) = match filter {
            None => {
                let mut r = self.search_layer(q, entry, ef, 0);
                r.truncate(k);
                (r, false)
            }
            Some(f) => self.search_filtered(q, entry, k, ef, f),
        };
        (hits.into_iter().map(|(d, s)| (s, 1.0 - d)).collect(), fallback)
    }

    // ---- construction -----------------------------------------------------------------

    /// Keep close but directionally diverse candidates (Algorithm 4). `candidates`
    /// must be sorted ascending by distance to the base point.
    fn select(&self, candidates: &[(f32, u32)], m: usize) -> Vec<u32> {
        let mut selected: Vec<u32> = Vec::with_capacity(m);
        let mut rejected: Vec<u32> = Vec::new();
        for &(d, c) in candidates {
            if selected.iter().all(|&s| self.dist(c, s) > d) {
                selected.push(c);
                if selected.len() == m {
                    return selected;
                }
            } else {
                rejected.push(c);
            }
        }
        let room = m - selected.len();
        selected.extend(rejected.into_iter().take(room));
        selected
    }

    fn prune(&mut self, node: u32, level: usize) {
        let limit = self.limit(level);
        if self.links[node as usize][level].len() <= limit {
            return;
        }
        let mut candidates: Vec<(f32, u32)> = self.links[node as usize][level]
            .iter()
            .map(|&n| (self.dist(node, n), n))
            .collect();
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let keep = self.select(&candidates, limit);
        let old = std::mem::replace(&mut self.links[node as usize][level], keep.clone());
        for other in old {
            if !keep.contains(&other) {
                self.links[other as usize][level].retain(|&x| x != node);
            }
        }
    }

    /// Insert a normalized vector; returns its slot.
    pub fn add(&mut self, vector: &[f32]) -> u32 {
        debug_assert_eq!(vector.len(), self.p.dim);
        let level = self.random_level();
        let slot = match self.free.pop() {
            Some(slot) => {
                let start = slot as usize * self.p.dim;
                self.vecs[start..start + self.p.dim].copy_from_slice(vector);
                self.levels[slot as usize] = level as i32;
                self.links[slot as usize] = vec![Vec::new(); level + 1];
                slot
            }
            None => {
                self.vecs.extend_from_slice(vector);
                self.levels.push(level as i32);
                self.links.push(vec![Vec::new(); level + 1]);
                (self.levels.len() - 1) as u32
            }
        };
        self.count += 1;
        let Some(ep) = self.ep else {
            self.ep = Some(slot);
            return slot;
        };
        let entry_level = self.levels[ep as usize] as usize;
        let mut entry = ep;
        for l in ((level + 1)..=entry_level).rev() {
            entry = self.greedy(vector, entry, l);
        }
        for l in (0..=level.min(entry_level)).rev() {
            let candidates = self.search_layer(vector, entry, self.p.ef_construction, l);
            let neighbors = self.select(&candidates, self.limit(l));
            self.links[slot as usize][l] = neighbors.clone();
            for &n in &neighbors {
                self.links[n as usize][l].push(slot);
            }
            for &n in &neighbors {
                self.prune(n, l);
            }
            if let Some(&(_, best)) = candidates.first() {
                entry = best;
            }
        }
        if level > entry_level {
            self.ep = Some(slot);
        }
        slot
    }

    /// Remove a node and repair the neighborhoods it leaves behind.
    pub fn remove(&mut self, slot: u32) -> bool {
        if self.level_of(slot).is_none() {
            return false;
        }
        let top = self.levels[slot as usize] as usize;
        let mut orphans: Vec<Vec<u32>> = Vec::with_capacity(top + 1);
        for level in 0..=top {
            let neighbors = std::mem::take(&mut self.links[slot as usize][level]);
            for &n in &neighbors {
                self.links[n as usize][level].retain(|&x| x != slot);
            }
            orphans.push(neighbors);
        }
        self.links[slot as usize] = Vec::new();
        self.levels[slot as usize] = -1;
        self.free.push(slot);
        self.count -= 1;
        if self.ep == Some(slot) {
            self.ep = if self.count == 0 {
                None
            } else if let Some(&n) = orphans[top].iter().min() {
                Some(n)
            } else {
                // Alone on its layer: pick the lowest slot at the new maximum level.
                let best = self.levels.iter().copied().max().unwrap_or(0);
                self.levels.iter().position(|&l| l == best).map(|p| p as u32)
            };
        }
        for (level, group) in orphans.into_iter().enumerate() {
            self.repair(level, group);
        }
        true
    }

    fn repair(&mut self, level: usize, mut orphans: Vec<u32>) {
        orphans.sort_unstable();
        let limit = self.limit(level);
        for &node in &orphans {
            let existing = self.links[node as usize][level].clone();
            let room = limit.saturating_sub(existing.len());
            if room == 0 {
                continue;
            }
            let mut pool: Vec<(f32, u32)> = orphans
                .iter()
                .copied()
                .filter(|&o| o != node && !existing.contains(&o))
                .map(|o| (self.dist(node, o), o))
                .collect();
            if pool.is_empty() {
                continue;
            }
            pool.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut nearest: Vec<f32> = pool
                .iter()
                .map(|&(_, c)| {
                    existing.iter().map(|&e| self.dist(c, e)).fold(f32::INFINITY, f32::min)
                })
                .collect();
            let mut chosen: Vec<usize> = Vec::new();
            let mut skipped: Vec<usize> = Vec::new();
            for i in 0..pool.len() {
                if nearest[i] > pool[i].0 {
                    chosen.push(i);
                    for j in 0..pool.len() {
                        let between = self.dist(pool[i].1, pool[j].1);
                        if between < nearest[j] {
                            nearest[j] = between;
                        }
                    }
                } else {
                    skipped.push(i);
                }
                if chosen.len() == room {
                    break;
                }
            }
            let fill = room - chosen.len();
            chosen.extend(skipped.into_iter().take(fill));
            let added: Vec<u32> = chosen.iter().map(|&i| pool[i].1).collect();
            for &other in &added {
                self.links[node as usize][level].push(other);
                self.links[other as usize][level].push(node);
            }
            for &other in &added {
                self.prune(other, level);
            }
        }
    }

    // ---- introspection ----------------------------------------------------------------

    pub fn stats(&self) -> Stats {
        let top = self.ep.map(|e| self.levels[e as usize] as usize + 1).unwrap_or(0);
        let mut degree_sum = 0usize;
        let mut isolated = 0usize;
        for (slot, level) in self.levels.iter().enumerate() {
            if *level >= 0 {
                let d = self.links[slot][0].len();
                degree_sum += d;
                if d == 0 && self.count > 1 {
                    isolated += 1;
                }
            }
        }
        Stats {
            nodes: self.count,
            levels: top,
            avg_degree0: if self.count == 0 { 0.0 } else { degree_sum as f64 / self.count as f64 },
            isolated0: isolated,
            free_slots: self.free.len(),
        }
    }

    /// Check every structural invariant; `Err` names the first violation.
    pub fn validate(&self) -> Result<(), String> {
        let live: Vec<u32> = (0..self.levels.len() as u32)
            .filter(|&s| self.levels[s as usize] >= 0)
            .collect();
        if live.len() != self.count {
            return Err(format!("count {} != live slots {}", self.count, live.len()));
        }
        match self.ep {
            None if self.count != 0 => return Err("missing entry point".into()),
            Some(_) if self.count == 0 => return Err("entry point on empty index".into()),
            Some(ep) => {
                let max = live.iter().map(|&s| self.levels[s as usize]).max().unwrap();
                if self.levels.get(ep as usize).copied() != Some(max) {
                    return Err(format!("entry point {ep} is not at the maximum level {max}"));
                }
            }
            None => {}
        }
        for &s in &live {
            let level = self.levels[s as usize] as usize;
            if self.links[s as usize].len() != level + 1 {
                return Err(format!("slot {s} has {} link layers for level {level}", self.links[s as usize].len()));
            }
            let norm: f32 = self.vec(s).iter().map(|x| x * x).sum::<f32>().sqrt();
            if !norm.is_finite() || !(norm < 1e-6 || (norm - 1.0).abs() < 1e-4) {
                return Err(format!("slot {s} has norm {norm}"));
            }
            for l in 0..=level {
                let list = &self.links[s as usize][l];
                if list.len() > self.limit(l) {
                    return Err(format!("slot {s} degree {} exceeds {} at level {l}", list.len(), self.limit(l)));
                }
                for (i, &n) in list.iter().enumerate() {
                    if n == s {
                        return Err(format!("self edge at slot {s}"));
                    }
                    if list[..i].contains(&n) {
                        return Err(format!("duplicate edge {s}->{n} at level {l}"));
                    }
                    match self.level_of(n) {
                        None => return Err(format!("edge {s}->{n} references a free slot")),
                        Some(nl) if nl < l => return Err(format!("edge {s}->{n} above level of target")),
                        _ => {}
                    }
                    if !self.links[n as usize][l].contains(&s) {
                        return Err(format!("edge {s}->{n} at level {l} is not bidirectional"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Per-slot levels (-1 = free) and directed edges `(level, src, dst)`.
    pub fn export(&self) -> (Vec<i32>, Vec<[i32; 3]>) {
        let mut edges = Vec::new();
        for (slot, level) in self.levels.iter().enumerate() {
            if *level < 0 {
                continue;
            }
            for (l, list) in self.links[slot].iter().enumerate() {
                for &n in list {
                    edges.push([l as i32, slot as i32, n as i32]);
                }
            }
        }
        (self.levels.clone(), edges)
    }

    #[allow(clippy::too_many_arguments)]
    /// Rebuild from exported topology; slot `i` holds `vectors[i]` (already normalized).
    pub fn import(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ml: Option<usize>,
        seed: u64,
        vectors: &[f32],
        levels: &[i32],
        edges: &[[i32; 3]],
        ep: Option<u32>,
    ) -> Result<Index, String> {
        let n = levels.len();
        if levels.iter().any(|&l| l > MAX_LEVEL as i32) {
            return Err(format!("node level exceeds the maximum of {MAX_LEVEL}"));
        }
        if n.checked_mul(dim) != Some(vectors.len()) {
            return Err("vector slab length does not match node count".into());
        }
        let mut index = Index::new(dim, m, ef_construction, ml, seed);
        index.vecs = vectors.to_vec();
        index.levels = levels.to_vec();
        index.links = levels
            .iter()
            .map(|&l| if l >= 0 { vec![Vec::new(); l as usize + 1] } else { Vec::new() })
            .collect();
        index.free = (0..n as u32).filter(|&s| levels[s as usize] < 0).collect();
        index.count = n - index.free.len();
        for &[l, s, d] in edges {
            let (l, s, d) = (l as usize, s as usize, d as u32);
            if s >= n || d as usize >= n || levels[s] < 0 || l > levels[s] as usize {
                return Err(format!("edge ({l},{s},{d}) is out of range"));
            }
            index.links[s][l].push(d);
        }
        index.ep = ep;
        if let Some(e) = ep {
            if index.level_of(e).is_none() {
                return Err("entry point references a free slot".into());
            }
        }
        index.validate()?;
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(seed: u64, dim: usize) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    fn build(n: usize, dim: usize) -> Index {
        let mut index = Index::new(dim, 8, 48, None, 7);
        for i in 0..n {
            index.add(&unit(i as u64 + 1, dim));
        }
        index
    }

    fn brute(index: &Index, q: &[f32], k: usize, keep: impl Fn(u32) -> bool) -> Vec<u32> {
        let mut all: Vec<(f32, u32)> = (0..index.slots() as u32)
            .filter(|&s| index.level_of(s).is_some() && keep(s))
            .map(|s| (distance(q, index.vec(s)), s))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        all.into_iter().take(k).map(|(_, s)| s).collect()
    }

    fn recall(index: &Index, queries: usize, ef: usize) -> f64 {
        let mut hits = 0;
        for q in 0..queries {
            let query = unit(10_000 + q as u64, index.p.dim);
            let truth = brute(index, &query, 10, |_| true);
            let (found, _) = index.search(&query, 10, ef, None);
            hits += found.iter().filter(|(s, _)| truth.contains(s)).count();
        }
        hits as f64 / (10 * queries) as f64
    }

    #[test]
    fn built_graph_is_valid_and_accurate() {
        let index = build(600, 16);
        index.validate().unwrap();
        assert!(recall(&index, 30, 64) > 0.95);
    }

    #[test]
    fn empty_and_single_node_searches() {
        let mut index = Index::new(4, 4, 16, None, 1);
        assert!(index.search(&unit(1, 4), 3, 8, None).0.is_empty());
        let slot = index.add(&unit(1, 4));
        let (hits, _) = index.search(&unit(1, 4), 3, 8, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, slot);
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
        assert!(index.remove(slot));
        assert!(index.is_empty() && index.entry_point().is_none());
        index.validate().unwrap();
    }

    #[test]
    fn removal_repairs_and_reuses_slots() {
        let mut index = build(300, 16);
        for s in (0..300).step_by(3) {
            assert!(index.remove(s));
            index.validate().unwrap();
        }
        assert!(!index.remove(0), "double remove must report failure");
        assert!(!index.remove(9999));
        let freed = index.stats().free_slots;
        let slot = index.add(&unit(5000, 16));
        assert_eq!(index.stats().free_slots, freed - 1);
        assert!(index.level_of(slot).is_some());
        index.validate().unwrap();
        assert!(recall(&index, 30, 64) > 0.9);
    }

    #[test]
    fn removing_everything_then_reusing_works() {
        let mut index = build(40, 8);
        for s in 0..40 {
            assert!(index.remove(s));
            index.validate().unwrap();
        }
        assert!(index.is_empty());
        index.add(&unit(1, 8));
        index.validate().unwrap();
    }

    #[test]
    fn filtered_search_respects_predicate_and_falls_back_when_sparse() {
        let index = build(500, 16);
        let query = unit(77, 16);
        let even = |s: u32| s % 2 == 0;
        let filter = Filter { predicate: Some(&even), max_distance: f32::INFINITY, max_visits: 2000 };
        let (hits, fallback) = index.search(&query, 10, 64, Some(&filter));
        assert!(!fallback && hits.len() == 10 && hits.iter().all(|(s, _)| s % 2 == 0));
        let rare = |s: u32| s == 499 || s == 3;
        let filter = Filter { predicate: Some(&rare), max_distance: f32::INFINITY, max_visits: 20 };
        let (hits, fallback) = index.search(&query, 2, 8, Some(&filter));
        assert!(fallback);
        let got: Vec<u32> = hits.iter().map(|(s, _)| *s).collect();
        assert_eq!(got, brute(&index, &query, 2, |s| s == 499 || s == 3));
    }

    #[test]
    fn min_score_bounds_results() {
        let index = build(300, 16);
        let query = unit(3, 16);
        let filter = Filter { predicate: None, max_distance: 0.9, max_visits: 5000 };
        let (hits, _) = index.search(&query, 50, 100, Some(&filter));
        assert!(hits.iter().all(|(_, score)| *score >= 0.1 - 1e-5));
    }

    #[test]
    fn export_import_round_trip_preserves_results() {
        let mut index = build(200, 16);
        for s in [5, 17, 90] {
            index.remove(s);
        }
        let (levels, edges) = index.export();
        let slab: Vec<f32> = (0..levels.len() as u32).flat_map(|s| index.vec(s).to_vec()).collect();
        let copy = Index::import(16, 8, 48, None, 1, &slab, &levels, &edges, index.entry_point()).unwrap();
        for q in 0..10 {
            let query = unit(900 + q, 16);
            assert_eq!(index.search(&query, 10, 50, None).0, copy.search(&query, 10, 50, None).0);
        }
    }

    #[test]
    fn import_rejects_corrupt_topology() {
        let index = build(20, 8);
        let (levels, mut edges) = index.export();
        let slab: Vec<f32> = (0..levels.len() as u32).flat_map(|s| index.vec(s).to_vec()).collect();
        edges.push([0, 0, 9999]);
        assert!(Index::import(8, 8, 48, None, 1, &slab, &levels, &edges, index.entry_point()).is_err());
        let (levels, mut edges) = index.export();
        edges.pop(); // breaks bidirectionality
        assert!(Index::import(8, 8, 48, None, 1, &slab, &levels, &edges, index.entry_point()).is_err());
        let (levels, edges) = index.export();
        assert!(Index::import(8, 8, 48, None, 1, &slab[1..], &levels, &edges, index.entry_point()).is_err());
    }

    #[test]
    fn seeded_builds_are_deterministic() {
        let a = build(150, 8);
        let b = build(150, 8);
        assert_eq!(a.export().1, b.export().1);
    }

    #[test]
    fn level_cap_is_honored() {
        let mut index = Index::new(4, 2, 8, Some(1), 3);
        for i in 0..200 {
            index.add(&unit(i + 1, 4));
        }
        assert!(index.stats().levels <= 2);
        index.validate().unwrap();
    }
}
