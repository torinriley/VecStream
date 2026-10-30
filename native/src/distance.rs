//! Dot product kernels. Vectors are stored L2-normalized, so cosine
//! similarity is a dot product and distance is `1 - dot`.

use std::sync::atomic::{AtomicU8, Ordering};

const UNKNOWN: u8 = 0;
const SCALAR: u8 = 1;
const AVX2: u8 = 2;
static KIND: AtomicU8 = AtomicU8::new(UNKNOWN);

fn detect() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return AVX2;
        }
    }
    SCALAR
}

fn kind() -> u8 {
    let k = KIND.load(Ordering::Relaxed);
    if k != UNKNOWN {
        return k;
    }
    let detected = detect();
    KIND.store(detected, Ordering::Relaxed);
    detected
}

/// 0 = portable (auto-vectorized), 1 = AVX2+FMA.
pub fn simd_kind() -> i32 {
    if kind() == AVX2 {
        1
    } else {
        0
    }
}

/// Force the portable kernel (tests and benchmarks compare the two paths).
pub fn force_scalar(on: bool) {
    KIND.store(if on { SCALAR } else { detect() }, Ordering::Relaxed);
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if kind() == AVX2 {
            // SAFETY: AVX2 and FMA support was verified at runtime by `detect`.
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let x = &a[i * 8..i * 8 + 8];
        let y = &b[i * 8..i * 8 + 8];
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut total: f32 = acc.iter().sum();
    for i in chunks * 8..a.len() {
        total += a[i] * b[i];
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
            acc1,
        );
        i += 16;
    }
    if i + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        i += 8;
    }
    let sum = _mm256_add_ps(acc0, acc1);
    let high = _mm256_extractf128_ps(sum, 1);
    let low = _mm256_castps256_ps128(sum);
    let quad = _mm_add_ps(high, low);
    let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    let single = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0x55));
    let mut total = _mm_cvtss_f32(single);
    while i < n {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

// ---- 8-bit scalar quantization kernels ----------------------------------------------------

/// Sum of `a[i] * b[i]` over unsigned bytes. Exact: the total is at most 255*255*65536 < 2^32.
#[inline]
pub fn dot_u8(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if kind() == AVX2 {
            // SAFETY: AVX2 support was verified at runtime by `detect`.
            return unsafe { dot_u8_avx2(a, b) };
        }
    }
    dot_u8_scalar(a, b)
}

pub fn dot_u8_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).fold(0u32, |acc, (&x, &y)| acc.wrapping_add(x as u32 * y as u32))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 16 <= n {
        let x = _mm256_cvtepu8_epi16(_mm_loadu_si128(pa.add(i) as *const __m128i));
        let y = _mm256_cvtepu8_epi16(_mm_loadu_si128(pb.add(i) as *const __m128i));
        // Pairs of 16-bit products summed into 32-bit lanes: values stay below 2^31.
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, y));
        i += 16;
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut total = lanes.iter().fold(0u32, |t, &l| t.wrapping_add(l as u32));
    while i < n {
        total = total.wrapping_add(*pa.add(i) as u32 * *pb.add(i) as u32);
        i += 1;
    }
    total
}

/// Sum of the bytes of `a` (exact in u32 for any realistic dimension).
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
pub fn sum_u8(a: &[u8]) -> u32 {
    a.iter().map(|&x| x as u32).sum()
}

/// `sum(q[i] * c[i])` for a float query against a byte code.
#[inline]
pub fn dot_f32_u8(q: &[f32], c: &[u8]) -> f32 {
    debug_assert_eq!(q.len(), c.len());
    #[cfg(target_arch = "x86_64")]
    {
        if kind() == AVX2 {
            // SAFETY: AVX2 and FMA support was verified at runtime by `detect`.
            return unsafe { dot_f32_u8_avx2(q, c) };
        }
    }
    dot_f32_u8_scalar(q, c)
}

pub fn dot_f32_u8_scalar(q: &[f32], c: &[u8]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = q.len() / 8;
    for i in 0..chunks {
        for j in 0..8 {
            acc[j] += q[i * 8 + j] * c[i * 8 + j] as f32;
        }
    }
    let mut total: f32 = acc.iter().sum();
    for i in chunks * 8..q.len() {
        total += q[i] * c[i] as f32;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_u8_avx2(q: &[f32], c: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let n = q.len();
    let (pq, pc) = (q.as_ptr(), c.as_ptr());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let bytes = _mm_loadl_epi64(pc.add(i) as *const __m128i);
        let floats = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(i)), floats, acc);
        i += 8;
    }
    let high = _mm256_extractf128_ps(acc, 1);
    let low = _mm256_castps256_ps128(acc);
    let quad = _mm_add_ps(high, low);
    let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    let single = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0x55));
    let mut total = _mm_cvtss_f32(single);
    while i < n {
        total += *pq.add(i) * *pc.add(i) as f32;
        i += 1;
    }
    total
}

#[inline]
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    (1.0 - dot(a, b)).clamp(0.0, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn kernels_agree_for_every_length_up_to_70() {
        for n in 0..70 {
            let (a, b) = (pseudo(n, 1), pseudo(n, 2));
            let exact: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            assert!((dot(&a, &b) as f64 - exact).abs() < 1e-4, "dispatch n={n}");
            assert!((dot_scalar(&a, &b) as f64 - exact).abs() < 1e-4, "scalar n={n}");
        }
    }

    fn pseudo_bytes(n: usize, seed: u32) -> Vec<u8> {
        pseudo(n, seed).iter().map(|x| ((x + 1.0) * 127.5) as u8).collect()
    }

    #[test]
    fn integer_dot_is_exact_for_every_length_and_extreme_values() {
        for n in 0..140 {
            let (a, b) = (pseudo_bytes(n, 3), pseudo_bytes(n, 4));
            let exact: u64 = a.iter().zip(&b).map(|(&x, &y)| x as u64 * y as u64).sum();
            assert_eq!(dot_u8(&a, &b) as u64, exact, "dispatch n={n}");
            assert_eq!(dot_u8_scalar(&a, &b) as u64, exact, "scalar n={n}");
        }
        let max = vec![255u8; 65_536];
        assert_eq!(dot_u8(&max, &max) as u64, 255u64 * 255 * 65_536);
        assert_eq!(sum_u8(&max), 255 * 65_536);
    }

    #[test]
    fn mixed_float_byte_dot_matches_a_float64_reference() {
        for n in 0..90 {
            let (q, c) = (pseudo(n, 5), pseudo_bytes(n, 6));
            let exact: f64 = q.iter().zip(&c).map(|(&x, &y)| x as f64 * y as f64).sum();
            assert!((dot_f32_u8(&q, &c) as f64 - exact).abs() < 1e-2, "dispatch n={n}");
            assert!((dot_f32_u8_scalar(&q, &c) as f64 - exact).abs() < 1e-2, "scalar n={n}");
        }
    }

    #[test]
    fn distance_is_clamped() {
        assert_eq!(distance(&[1.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(distance(&[1.0, 0.0], &[-1.0, 0.0]), 2.0);
        assert_eq!(distance(&[1.0000001], &[1.0000001]), 0.0);
    }
}
