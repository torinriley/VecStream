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

    #[test]
    fn distance_is_clamped() {
        assert_eq!(distance(&[1.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(distance(&[1.0, 0.0], &[-1.0, 0.0]), 2.0);
        assert_eq!(distance(&[1.0000001], &[1.0000001]), 0.0);
    }
}
