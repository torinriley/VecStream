//! Native HNSW engine for VecStream, exposed through a small C ABI.
//!
//! Safety contract (enforced by the Python wrapper and re-checked here):
//! every pointer is valid for the stated length; an `Index` handle is mutated
//! by at most one thread at a time and never concurrently with a search.
//! Concurrent searches on one handle are supported. Panics never cross the
//! boundary: they are caught and reported as error codes.

mod distance;
mod index;

use index::{Filter, Index};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

pub const ABI_VERSION: u32 = 1;
pub const ERR_ARGUMENT: i32 = -1;
pub const ERR_PANIC: i32 = -2;
pub const ERR_STATE: i32 = -3;

type Predicate = Option<extern "C" fn(u32) -> i32>;

fn guarded<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

#[no_mangle]
pub extern "C" fn vs_abi_version() -> u32 {
    ABI_VERSION
}

/// 1 when the AVX2+FMA kernel is active, else 0.
#[no_mangle]
pub extern "C" fn vs_simd_kind() -> i32 {
    distance::simd_kind()
}

/// Testing hook: force the portable kernel (`on != 0`) or restore detection.
#[no_mangle]
pub extern "C" fn vs_force_scalar(on: i32) {
    distance::force_scalar(on != 0);
}

#[no_mangle]
pub extern "C" fn vs_new(dim: u32, m: u32, ef_construction: u32, ml: i64, seed: u64) -> *mut c_void {
    guarded(std::ptr::null_mut(), || {
        if dim == 0 || m < 2 || ef_construction < m {
            return std::ptr::null_mut();
        }
        let ml = if ml < 0 { None } else { Some(ml as usize) };
        let index = Index::new(dim as usize, m as usize, ef_construction as usize, ml, seed);
        Box::into_raw(Box::new(index)) as *mut c_void
    })
}

/// # Safety
/// `handle` must come from `vs_new`/`vs_import` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn vs_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut Index));
    }
}

unsafe fn index_ref<'a>(handle: *const c_void) -> Option<&'a Index> {
    (handle as *const Index).as_ref()
}

unsafe fn index_mut<'a>(handle: *mut c_void) -> Option<&'a mut Index> {
    (handle as *mut Index).as_mut()
}

/// Insert one normalized vector of the index dimension; returns its slot or a negative error.
///
/// # Safety
/// `vector` must point to `dim` readable floats.
#[no_mangle]
pub unsafe extern "C" fn vs_add(handle: *mut c_void, vector: *const f32) -> i64 {
    guarded(ERR_PANIC as i64, || {
        let (Some(index), false) = (index_mut(handle), vector.is_null()) else {
            return ERR_ARGUMENT as i64;
        };
        let v = slice::from_raw_parts(vector, index.p.dim);
        if v.iter().any(|x| !x.is_finite()) {
            return ERR_ARGUMENT as i64;
        }
        index.add(v) as i64
    })
}

/// # Safety
/// `handle` must be a live index handle.
#[no_mangle]
pub unsafe extern "C" fn vs_remove(handle: *mut c_void, slot: u32) -> i32 {
    guarded(ERR_PANIC, || match index_mut(handle) {
        Some(index) => i32::from(index.remove(slot)),
        None => ERR_ARGUMENT,
    })
}

/// # Safety
/// `handle` must be null or a live index handle.
#[no_mangle]
pub unsafe extern "C" fn vs_len(handle: *const c_void) -> i64 {
    guarded(ERR_PANIC as i64, || {
        index_ref(handle).map_or(ERR_ARGUMENT as i64, |i| i.len() as i64)
    })
}

/// # Safety
/// `handle` must be null or a live index handle.
#[no_mangle]
pub unsafe extern "C" fn vs_slots(handle: *const c_void) -> i64 {
    guarded(ERR_PANIC as i64, || {
        index_ref(handle).map_or(ERR_ARGUMENT as i64, |i| i.slots() as i64)
    })
}

fn build_filter<'a>(
    predicate: Predicate,
    adapter: &'a dyn Fn(u32) -> bool,
    min_score: f32,
    max_visits: u64,
    ef: usize,
) -> Option<Filter<'a>> {
    let constrained = predicate.is_some() || !min_score.is_nan();
    constrained.then(|| Filter {
        predicate: predicate.map(|_| adapter),
        max_distance: if min_score.is_nan() { f32::INFINITY } else { 1.0 - min_score },
        max_visits: if max_visits == 0 { (50 * ef).max(2000) } else { max_visits as usize },
    })
}

/// Search one query. Writes up to `k` hits to `out_slots`/`out_scores`; returns the hit count,
/// or a negative error. `min_score` NaN disables the score bound; `predicate` may be null.
/// `fallback_used` (nullable) is set to 1 when the exact-scan fallback ran.
///
/// # Safety
/// Pointers must be valid for `dim`, `k`, `k` elements respectively.
#[no_mangle]
pub unsafe extern "C" fn vs_search(
    handle: *const c_void,
    query: *const f32,
    k: u32,
    ef: u32,
    min_score: f32,
    predicate: Predicate,
    max_visits: u64,
    out_slots: *mut u32,
    out_scores: *mut f32,
    fallback_used: *mut i32,
) -> i64 {
    guarded(ERR_PANIC as i64, || {
        let Some(index) = index_ref(handle) else {
            return ERR_ARGUMENT as i64;
        };
        if query.is_null() || out_slots.is_null() || out_scores.is_null() || k == 0 || ef == 0 {
            return ERR_ARGUMENT as i64;
        }
        let q = slice::from_raw_parts(query, index.p.dim);
        let adapter = |slot: u32| predicate.is_some_and(|p| p(slot) != 0);
        let filter = build_filter(predicate, &adapter, min_score, max_visits, ef as usize);
        let (hits, fallback) = index.search(q, k as usize, ef as usize, filter.as_ref());
        for (i, (slot, score)) in hits.iter().enumerate() {
            *out_slots.add(i) = *slot;
            *out_scores.add(i) = *score;
        }
        if !fallback_used.is_null() {
            *fallback_used = i32::from(fallback);
        }
        hits.len() as i64
    })
}

/// Search `n` queries on up to `threads` OS threads without the Python GIL.
/// Output row `i` occupies `[i*k, (i+1)*k)`; `out_counts[i]` is its hit count.
///
/// # Safety
/// `queries` holds `n*dim` floats; `out_slots`/`out_scores` hold `n*k`; `out_counts` holds `n`.
#[no_mangle]
pub unsafe extern "C" fn vs_search_batch(
    handle: *const c_void,
    queries: *const f32,
    n: u32,
    k: u32,
    ef: u32,
    min_score: f32,
    threads: u32,
    out_slots: *mut u32,
    out_scores: *mut f32,
    out_counts: *mut u32,
) -> i32 {
    guarded(ERR_PANIC, || {
        let Some(index) = index_ref(handle) else {
            return ERR_ARGUMENT;
        };
        if queries.is_null() || out_slots.is_null() || out_scores.is_null() || out_counts.is_null() {
            return ERR_ARGUMENT;
        }
        if n == 0 || k == 0 || ef == 0 {
            return ERR_ARGUMENT;
        }
        let (n, k, dim) = (n as usize, k as usize, index.p.dim);
        let qs = slice::from_raw_parts(queries, n * dim);
        let slots = slice::from_raw_parts_mut(out_slots, n * k);
        let scores = slice::from_raw_parts_mut(out_scores, n * k);
        let counts = slice::from_raw_parts_mut(out_counts, n);
        let workers = (threads.max(1) as usize).min(n);
        let per = n.div_ceil(workers);
        let failed = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let rows = slots.chunks_mut(per * k).zip(scores.chunks_mut(per * k)).zip(counts.chunks_mut(per));
            for (chunk, ((slot_rows, score_rows), count_rows)) in rows.enumerate() {
                let failed = &failed;
                scope.spawn(move || {
                    let outcome = catch_unwind(AssertUnwindSafe(|| {
                        let filter = (!min_score.is_nan()).then(|| Filter {
                            predicate: None,
                            max_distance: 1.0 - min_score,
                            max_visits: (50 * ef as usize).max(2000),
                        });
                        for (r, count) in count_rows.iter_mut().enumerate() {
                            let qi = chunk * per + r;
                            let q = &qs[qi * dim..(qi + 1) * dim];
                            let (hits, _) = index.search(q, k, ef as usize, filter.as_ref());
                            for (j, (slot, score)) in hits.iter().enumerate() {
                                slot_rows[r * k + j] = *slot;
                                score_rows[r * k + j] = *score;
                            }
                            *count = hits.len() as u32;
                        }
                    }));
                    if outcome.is_err() {
                        failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });
            }
        });
        if failed.load(std::sync::atomic::Ordering::Relaxed) {
            ERR_PANIC
        } else {
            0
        }
    })
}

/// out[0..5] = nodes, levels, avg level-0 degree, isolated nodes, free slots.
///
/// # Safety
/// `out` must point to 5 writable doubles.
#[no_mangle]
pub unsafe extern "C" fn vs_stats(handle: *const c_void, out: *mut f64) -> i32 {
    guarded(ERR_PANIC, || {
        let (Some(index), false) = (index_ref(handle), out.is_null()) else {
            return ERR_ARGUMENT;
        };
        let s = index.stats();
        let values = [s.nodes as f64, s.levels as f64, s.avg_degree0, s.isolated0 as f64, s.free_slots as f64];
        for (i, v) in values.iter().enumerate() {
            *out.add(i) = *v;
        }
        0
    })
}

/// Returns 0 when valid, 1 when a violation was written (NUL-terminated) to `message`.
///
/// # Safety
/// `message` must hold `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn vs_validate(handle: *const c_void, message: *mut u8, capacity: u32) -> i32 {
    guarded(ERR_PANIC, || {
        let Some(index) = index_ref(handle) else {
            return ERR_ARGUMENT;
        };
        match index.validate() {
            Ok(()) => 0,
            Err(text) => {
                if !message.is_null() && capacity > 0 {
                    let bytes = text.as_bytes();
                    let n = bytes.len().min(capacity as usize - 1);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), message, n);
                    *message.add(n) = 0;
                }
                1
            }
        }
    })
}

/// out[0] = slot count, out[1] = edge count, out[2] = entry point (-1 if none).
///
/// # Safety
/// `out` must point to 3 writable i64.
#[no_mangle]
pub unsafe extern "C" fn vs_export_sizes(handle: *const c_void, out: *mut i64) -> i32 {
    guarded(ERR_PANIC, || {
        let (Some(index), false) = (index_ref(handle), out.is_null()) else {
            return ERR_ARGUMENT;
        };
        let (levels, edges) = index.export();
        *out = levels.len() as i64;
        *out.add(1) = edges.len() as i64;
        *out.add(2) = index.entry_point().map_or(-1, i64::from);
        0
    })
}

/// Fill caller buffers sized by `vs_export_sizes`: `levels[slots]` and `edges[edge_count*3]`
/// as `(level, src, dst)` triples. Vectors are not exported; the caller owns the originals.
///
/// # Safety
/// Buffers must have the sizes reported by `vs_export_sizes`.
#[no_mangle]
pub unsafe extern "C" fn vs_export(
    handle: *const c_void,
    levels: *mut i32,
    edges: *mut i32,
) -> i32 {
    guarded(ERR_PANIC, || {
        let Some(index) = index_ref(handle) else {
            return ERR_ARGUMENT;
        };
        if levels.is_null() || edges.is_null() {
            return ERR_ARGUMENT;
        }
        let (lv, ed) = index.export();
        std::ptr::copy_nonoverlapping(lv.as_ptr(), levels, lv.len());
        for (i, e) in ed.iter().enumerate() {
            std::ptr::copy_nonoverlapping(e.as_ptr(), edges.add(i * 3), 3);
        }
        0
    })
}

/// Build an index from persisted topology; slot `i` holds `vectors[i*dim..]`. Null on any
/// inconsistency (the Python layer then rebuilds from vectors).
///
/// # Safety
/// Arrays must have the lengths given by `slots` and `edge_count`.
#[no_mangle]
pub unsafe extern "C" fn vs_import(
    dim: u32,
    m: u32,
    ef_construction: u32,
    ml: i64,
    seed: u64,
    slots: u64,
    vectors: *const f32,
    levels: *const i32,
    edge_count: u64,
    edges: *const i32,
    entry: i64,
) -> *mut c_void {
    guarded(std::ptr::null_mut(), || {
        if dim == 0 || m < 2 || ef_construction < m || (slots > 0 && (vectors.is_null() || levels.is_null())) {
            return std::ptr::null_mut();
        }
        if edge_count > 0 && edges.is_null() {
            return std::ptr::null_mut();
        }
        let (slots, edge_count) = (slots as usize, edge_count as usize);
        let vec_slice = if slots == 0 { &[][..] } else { slice::from_raw_parts(vectors, slots * dim as usize) };
        let level_slice = if slots == 0 { &[][..] } else { slice::from_raw_parts(levels, slots) };
        let flat = if edge_count == 0 { &[][..] } else { slice::from_raw_parts(edges, edge_count * 3) };
        let triples: Vec<[i32; 3]> = flat.as_chunks::<3>().0.to_vec();
        if vec_slice.iter().any(|x| !x.is_finite()) || triples.iter().any(|t| t.iter().any(|&x| x < 0)) {
            return std::ptr::null_mut();
        }
        let ml = if ml < 0 { None } else { Some(ml as usize) };
        let entry = if entry < 0 { None } else { Some(entry as u32) };
        match Index::import(
            dim as usize,
            m as usize,
            ef_construction as usize,
            ml,
            seed,
            vec_slice,
            level_slice,
            &triples,
            entry,
        ) {
            Ok(index) => Box::into_raw(Box::new(index)) as *mut c_void,
            Err(_) => std::ptr::null_mut(),
        }
    })
}
