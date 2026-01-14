//! Benchmarks for distance functions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nndescent_core::distance::{Distance, Euclidean, SquaredEuclidean, Cosine, InnerProduct};

fn generate_vectors(n: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.1).cos()).collect();
    (a, b)
}

fn bench_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean");
    
    for dim in [32, 64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(1, *dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(Euclidean.distance(&a, &b))
            })
        });
    }
    
    group.finish();
}

fn bench_squared_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("squared_euclidean");
    
    for dim in [32, 64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(1, *dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(SquaredEuclidean.distance(&a, &b))
            })
        });
    }
    
    group.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine");
    
    for dim in [32, 64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(1, *dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(Cosine.distance(&a, &b))
            })
        });
    }
    
    group.finish();
}

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");
    
    for dim in [32, 64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(1, *dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(InnerProduct.distance(&a, &b))
            })
        });
    }
    
    group.finish();
}

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");
    
    let n_points = 1000;
    let dim = 128;
    let (data, _) = generate_vectors(n_points, dim);
    let query = &data[0..dim];
    
    group.bench_function("euclidean_1000x128", |bench| {
        bench.iter(|| {
            let mut total = 0.0f32;
            for i in 0..n_points {
                let point = &data[i * dim..(i + 1) * dim];
                total += SquaredEuclidean.distance(query, point);
            }
            black_box(total)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_euclidean,
    bench_squared_euclidean,
    bench_cosine,
    bench_inner_product,
    bench_batch_distances,
);
criterion_main!(benches);
