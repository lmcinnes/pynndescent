//! Benchmarks for heap data structures.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nndescent_core::heap::NeighborHeap;
use nndescent_core::rng::TauRand;

fn bench_neighbor_heap_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_heap_push");
    
    for n_neighbors in [10, 30, 50, 100].iter() {
        let n_points = 1000;
        
        group.bench_with_input(
            BenchmarkId::new("random_push", n_neighbors),
            n_neighbors,
            |bench, &k| {
                let mut rng = TauRand::new(42);
                bench.iter(|| {
                    let mut heap = NeighborHeap::new(n_points, k);
                    // Push random neighbors
                    for i in 0..n_points {
                        for _ in 0..k * 2 {
                            let neighbor = (rng.next_int() as usize) % n_points;
                            let dist = (rng.next_int() as f32).abs() / (i32::MAX as f32);
                            heap.checked_flagged_push(i, dist, neighbor as i32, true);
                        }
                    }
                    black_box(heap)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_neighbor_heap_deheap(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_heap_deheap");
    
    for n_neighbors in [10, 30, 50].iter() {
        let n_points = 1000;
        
        group.bench_with_input(
            BenchmarkId::new("deheap_sort", n_neighbors),
            n_neighbors,
            |bench, &k| {
                let mut rng = TauRand::new(42);
                
                // Pre-build heap
                let mut template_heap = NeighborHeap::new(n_points, k);
                for i in 0..n_points {
                    for _ in 0..k * 2 {
                        let neighbor = (rng.next_int() as usize) % n_points;
                        let dist = (rng.next_int() as f32).abs() / (i32::MAX as f32);
                        template_heap.checked_flagged_push(i, dist, neighbor as i32, true);
                    }
                }
                
                bench.iter(|| {
                    let mut heap = template_heap.clone();
                    heap.deheap_sort();
                    black_box(heap)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_neighbor_heap_push,
    bench_neighbor_heap_deheap,
);
criterion_main!(benches);
