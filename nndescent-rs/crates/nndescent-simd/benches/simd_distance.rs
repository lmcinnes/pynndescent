//! Benchmarks for SIMD distance functions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

#[cfg(target_arch = "x86_64")]
use nndescent_simd::avx2;

use nndescent_simd::avx512;

fn generate_vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos()).collect();
    (a, b)
}

fn scalar_l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

fn scalar_inner_product(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
fn bench_l2_sqr_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_sqr");
    
    for dim in [64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(*dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| black_box(scalar_l2_sqr(&a, &b)))
        });
        
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            group.bench_with_input(BenchmarkId::new("avx2", dim), dim, |bench, _| {
                bench.iter(|| unsafe { black_box(avx2::l2_sqr_avx2(&a, &b)) })
            });
        }
        
        // AVX-512 fallback (uses scalar on stable Rust)
        group.bench_with_input(BenchmarkId::new("avx512_fallback", dim), dim, |bench, _| {
            bench.iter(|| black_box(avx512::l2_sqr_avx512(&a, &b)))
        });
    }
    
    group.finish();
}

#[cfg(target_arch = "x86_64")]
fn bench_inner_product_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");
    
    for dim in [64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(*dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| black_box(scalar_inner_product(&a, &b)))
        });
        
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            group.bench_with_input(BenchmarkId::new("avx2", dim), dim, |bench, _| {
                bench.iter(|| unsafe { black_box(avx2::inner_product_avx2(&a, &b)) })
            });
        }
        
        // AVX-512 fallback
        group.bench_with_input(BenchmarkId::new("avx512_fallback", dim), dim, |bench, _| {
            bench.iter(|| black_box(avx512::inner_product_avx512(&a, &b)))
        });
    }
    
    group.finish();
}

#[cfg(target_arch = "x86_64")]
fn bench_cosine_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine");
    
    for dim in [64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(*dim);
        
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            group.bench_with_input(BenchmarkId::new("avx2", dim), dim, |bench, _| {
                bench.iter(|| unsafe { black_box(avx2::cosine_avx2(&a, &b)) })
            });
        }
        
        // AVX-512 fallback
        group.bench_with_input(BenchmarkId::new("avx512_fallback", dim), dim, |bench, _| {
            bench.iter(|| black_box(avx512::cosine_avx512(&a, &b)))
        });
    }
    
    group.finish();
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    bench_l2_sqr_comparison,
    bench_inner_product_comparison,
    bench_cosine_comparison,
);

#[cfg(not(target_arch = "x86_64"))]
fn bench_placeholder(_c: &mut Criterion) {
    // No SIMD benchmarks on non-x86_64 platforms
}

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(benches, bench_placeholder);

criterion_main!(benches);
