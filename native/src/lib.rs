//! Native HNSW engine for VecStream, exposed through a small C ABI.
//!
//! Safety contract:
//! * Every pointer is valid for the length implied by its arguments.
//! * A handle is internally synchronized with a readers-writer lock: searches
//!   run concurrently, mutations are exclusive. Misuse from several threads
//!   therefore blocks instead of racing. A predicate callback must not call
//!   back into a mutating function on the same handle (that would deadlock).
//! * A handle must not be used after `vs_free`.
//! * Panics never cross the boundary: they are caught and reported as error
//!   codes, and a panic inside a mutation poisons the handle (all later calls
//!   fail closed with `ERR_STATE`).
//! * Parameters that would force absurd allocations (which abort the process,
//!   something `catch_unwind` cannot intercept) are rejected up front.

mod distance;
mod index;

use index::{Filter, Index};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub const ABI_VERSION: u32 = 2;
pub const ERR_ARGUMENT: i32 = -1;
pub const ERR_PANIC: i32 = -2;
pub const ERR_STATE: i32 = -3;
pub const ERR_CAPACITY: i32 = -4;

pub const MAX_DIM: u32 = 65_536;
pub const MAX_M: u32 = 1_024;
pub const MAX_EF_CONSTRUCTION: u32 = 1_000_000;
const MAX_SLOTS: usize = (u32::MAX - 2) as usize;

type Predicate = Option<extern "C" fn(u32) -> i32>;

struct Handle {
    index: RwLock<Index>,
}

fn guarded<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

unsafe fn handle<'a>(ptr: *const c_void) -> Option<&'a Handle> {
    (ptr as *const Handle).as_ref()
}

fn read(h: &Handle) -> Option<RwLockReadGuard<'_, Index>> {
    h.index.read().ok()
}

fn write(h: &Handle) -> Option<RwLockWriteGuard<'_, Index>> {
    h.index.write().ok()
}

fn params_ok(dim: u32, m: u32, ef_construction: u32) -> bool {
    (1..=MAX_DIM).contains(&dim)
        && (2..=MAX_M).contains(&m)
        && ef_construction >= m
        && ef_construction <= MAX_EF_CONSTRUCTION
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

/// Returns null for invalid parameters (`dim` 1..=65536, `m` 2..=1024,
/// `ef_construction` in `m..=1_000_000`).
#[no_mangle]
pub extern "C" fn vs_new(dim: u32, m: u32, ef_construction: u32, ml: i64, seed: u64) -> *mut c_void {
    guarded(std::ptr::null_mut(), || {
        if !params_ok(dim, m, ef_construction) {
            return std::ptr::null_mut();
        }
        let ml = if ml < 0 { None } else { Some(ml as usize) };
        let index = Index::new(dim as usize, m as usize, ef_construction as usize, ml, seed);
        Box::into_raw(Box::new(Handle { index: RwLock::new(index) })) as *mut c_void
    })
}

/// # Safety
/// `ptr` must be null or come from `vs_new`/`vs_import`, and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn vs_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr as *mut Handle));
    }
}

/// Insert one normalized vector of the index dimension; returns its slot or a negative error.
///
/// # Safety
/// `vector` must point to `dim` readable floats.
#[no_mangle]
pub unsafe extern "C" fn vs_add(ptr: *mut c_void, vector: *const f32) -> i64 {
    guarded(ERR_PANIC as i64, || {
        let (Some(h), false) = (handle(ptr), vector.is_null()) else {
            return ERR_ARGUMENT as i64;
        };
        let Some(mut index) = write(h) else {
            return ERR_STATE as i64;
        };
        if index.slots() >= MAX_SLOTS {
            return ERR_CAPACITY as i64;
        }
        let v = slice::from_raw_parts(vector, index.p.dim);
        if v.iter().any(|x| !x.is_finite()) {
            return ERR_ARGUMENT as i64;
        }
        index.add(v) as i64
    })
}

/// Returns 1 if removed, 0 if the slot was not live.
///
/// # Safety
/// `ptr` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn vs_remove(ptr: *mut c_void, slot: u32) -> i32 {
    guarded(ERR_PANIC, || match handle(ptr).map(write) {
        Some(Some(mut index)) => i32::from(index.remove(slot)),
        Some(None) => ERR_STATE,
        None => ERR_ARGUMENT,
    })
}

/// # Safety
/// `ptr` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn vs_len(ptr: *const c_void) -> i64 {
    guarded(ERR_PANIC as i64, || match handle(ptr).map(read) {
        Some(Some(index)) => index.len() as i64,
        Some(None) => ERR_STATE as i64,
        None => ERR_ARGUMENT as i64,
    })
}

/// # Safety
/// `ptr` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn vs_slots(ptr: *const c_void) -> i64 {
    guarded(ERR_PANIC as i64, || match handle(ptr).map(read) {
        Some(Some(index)) => index.slots() as i64,
        Some(None) => ERR_STATE as i64,
        None => ERR_ARGUMENT as i64,
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
/// `query` holds `dim` floats; `out_slots` and `out_scores` hold `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vs_search(
    ptr: *const c_void,
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
        let Some(h) = handle(ptr) else {
            return ERR_ARGUMENT as i64;
        };
        if query.is_null() || out_slots.is_null() || out_scores.is_null() || k == 0 || ef == 0 {
            return ERR_ARGUMENT as i64;
        }
        let Some(index) = read(h) else {
            return ERR_STATE as i64;
        };
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
    ptr: *const c_void,
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
        let Some(h) = handle(ptr) else {
            return ERR_ARGUMENT;
        };
        if queries.is_null() || out_slots.is_null() || out_scores.is_null() || out_counts.is_null() {
            return ERR_ARGUMENT;
        }
        if n == 0 || k == 0 || ef == 0 {
            return ERR_ARGUMENT;
        }
        let Some(guard) = read(h) else {
            return ERR_STATE;
        };
        let index: &Index = &guard;
        let (n, k, dim) = (n as usize, k as usize, index.p.dim);
        let (Some(total_q), Some(total_out)) = (n.checked_mul(dim), n.checked_mul(k)) else {
            return ERR_ARGUMENT;
        };
        let qs = slice::from_raw_parts(queries, total_q);
        let slots = slice::from_raw_parts_mut(out_slots, total_out);
        let scores = slice::from_raw_parts_mut(out_scores, total_out);
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
pub unsafe extern "C" fn vs_stats(ptr: *const c_void, out: *mut f64) -> i32 {
    guarded(ERR_PANIC, || {
        let (Some(h), false) = (handle(ptr), out.is_null()) else {
            return ERR_ARGUMENT;
        };
        let Some(index) = read(h) else {
            return ERR_STATE;
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
/// `message` must hold `capacity` bytes (or be null).
#[no_mangle]
pub unsafe extern "C" fn vs_validate(ptr: *const c_void, message: *mut u8, capacity: u32) -> i32 {
    guarded(ERR_PANIC, || {
        let Some(h) = handle(ptr) else {
            return ERR_ARGUMENT;
        };
        let Some(index) = read(h) else {
            return ERR_STATE;
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
pub unsafe extern "C" fn vs_export_sizes(ptr: *const c_void, out: *mut i64) -> i32 {
    guarded(ERR_PANIC, || {
        let (Some(h), false) = (handle(ptr), out.is_null()) else {
            return ERR_ARGUMENT;
        };
        let Some(index) = read(h) else {
            return ERR_STATE;
        };
        let (levels, edges) = index.export();
        *out = levels.len() as i64;
        *out.add(1) = edges.len() as i64;
        *out.add(2) = index.entry_point().map_or(-1, i64::from);
        0
    })
}

/// Fill `levels` (capacity `levels_cap` ints) and `edges` (capacity `edge_cap` triples of
/// `(level, src, dst)`). Returns `ERR_CAPACITY` without writing if either is too small, e.g.
/// because the index changed since `vs_export_sizes`. Vectors are not exported.
///
/// # Safety
/// `levels` holds `levels_cap` ints and `edges` holds `edge_cap * 3` ints.
#[no_mangle]
pub unsafe extern "C" fn vs_export(
    ptr: *const c_void,
    levels: *mut i32,
    levels_cap: u64,
    edges: *mut i32,
    edge_cap: u64,
) -> i32 {
    guarded(ERR_PANIC, || {
        let Some(h) = handle(ptr) else {
            return ERR_ARGUMENT;
        };
        if levels.is_null() || edges.is_null() {
            return ERR_ARGUMENT;
        }
        let Some(index) = read(h) else {
            return ERR_STATE;
        };
        let (lv, ed) = index.export();
        if lv.len() as u64 > levels_cap || ed.len() as u64 > edge_cap {
            return ERR_CAPACITY;
        }
        std::ptr::copy_nonoverlapping(lv.as_ptr(), levels, lv.len());
        for (i, e) in ed.iter().enumerate() {
            std::ptr::copy_nonoverlapping(e.as_ptr(), edges.add(i * 3), 3);
        }
        0
    })
}

/// Build an index from persisted topology; slot `i` holds `vectors[i*dim..]`. Null on any
/// inconsistency or implausible size (the Python layer then rebuilds from vectors).
///
/// # Safety
/// `vectors` holds `slots*dim` floats, `levels` holds `slots` ints, `edges` holds
/// `edge_count*3` ints.
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
        if !params_ok(dim, m, ef_construction) || slots as usize > MAX_SLOTS {
            return std::ptr::null_mut();
        }
        if (slots > 0 && (vectors.is_null() || levels.is_null())) || (edge_count > 0 && edges.is_null()) {
            return std::ptr::null_mut();
        }
        let (slots, edge_count) = (slots as usize, edge_count as usize);
        let (Some(vec_len), Some(edge_len)) = ((slots).checked_mul(dim as usize), edge_count.checked_mul(3)) else {
            return std::ptr::null_mut();
        };
        let vec_slice = if slots == 0 { &[][..] } else { slice::from_raw_parts(vectors, vec_len) };
        let level_slice = if slots == 0 { &[][..] } else { slice::from_raw_parts(levels, slots) };
        let flat = if edge_count == 0 { &[][..] } else { slice::from_raw_parts(edges, edge_len) };
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
            Ok(index) => Box::into_raw(Box::new(Handle { index: RwLock::new(index) })) as *mut c_void,
            Err(_) => std::ptr::null_mut(),
        }
    })
}
