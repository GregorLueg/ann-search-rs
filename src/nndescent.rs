use faer::MatRef;
use num_traits::{Float, FromPrimitive};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::iter::Sum;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::*;

use crate::annoy::*;
use crate::dist::*;
use crate::utils::*;

/// Neighbour entry in k-NN graph
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Neighbour<T> {
    pid: u32,
    dist: T,
    is_new: u32,
}

impl<T: Copy> Neighbour<T> {
    #[inline(always)]
    fn new(pid: usize, dist: T, is_new: bool) -> Self {
        Self {
            pid: pid as u32,
            dist,
            is_new: is_new as u32,
        }
    }

    #[inline(always)]
    fn is_new(&self) -> bool {
        self.is_new != 0
    }

    #[inline(always)]
    fn pid(&self) -> usize {
        self.pid as usize
    }
}

/// Sensible parameter defaults based on data size
#[derive(Clone, Copy, Debug)]
pub enum GraphSize {
    Small,  // < 100k
    Medium, // 100k-1M
    Large,  // > 1M
}

impl GraphSize {
    fn from_n(n: usize) -> Self {
        match n {
            0..50_000 => Self::Small,
            50_000..500_000 => Self::Medium,
            _ => Self::Large,
        }
    }

    fn max_candidates(&self) -> usize {
        match self {
            Self::Small => 60,
            Self::Medium => 100,
            Self::Large => 120,
        }
    }

    fn annoy_trees(&self) -> usize {
        match self {
            Self::Small => 24,
            Self::Medium => 32,
            Self::Large => 40,
        }
    }
}

/// Trait for applying neighbour updates (different implementations for f32/f64)
pub trait UpdateNeighbours<T> {
    fn update_neighbours(
        &self,
        updates: &[(usize, usize, T)],
        graph: &mut [Vec<Neighbour<T>>],
        updates_count: &AtomicUsize,
    );
}

thread_local! {
    static HEAP_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static HEAP_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static PID_SET: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
}

/// NN-Descent index for approximate nearest neighbour search
pub struct NNDescent<T> {
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    metric: Dist,
    norms: Vec<T>,
    graph: Vec<Vec<(usize, T)>>,
    converged: bool,
}

impl<T> VectorDistance<T> for NNDescent<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        &self.norms
    }
}

impl<T> NNDescent<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
    Self: UpdateNeighbours<T>,
{
    /// Build a new NN-Descent index
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mat: MatRef<T>,
        k: usize,
        metric: Dist,
        max_iter: usize,
        delta: T,
        max_candidates: Option<usize>, // ← Allow override
        graph_size: Option<GraphSize>,
        diversify_prob: T,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let n = mat.nrows();
        let n_features = mat.ncols();
        let graph_params = graph_size.unwrap_or_else(|| GraphSize::from_n(n));

        // Flatten matrix
        let mut vectors_flat = Vec::with_capacity(n * n_features);
        for i in 0..n {
            vectors_flat.extend(mat.row(i).iter().copied());
        }

        // Precompute norms for cosine
        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * n_features;
                    let end = start + n_features;
                    vectors_flat[start..end]
                        .iter()
                        .map(|x| *x * *x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        let builder = NNDescent {
            vectors_flat,
            dim: n_features,
            n,
            metric,
            norms,
            graph: Vec::new(),
            converged: false,
        };

        // Build initial index
        let start = Instant::now();
        let annoy_index = AnnoyIndex::new(mat, graph_params.annoy_trees(), metric, seed);
        if verbose {
            println!("Built Annoy index: {:.2?}", start.elapsed());
        }

        // Use provided max_candidates or calculate default
        let effective_max_candidates =
            max_candidates.unwrap_or_else(|| graph_params.max_candidates());

        let build_graph = builder.run(
            k,
            max_iter,
            &annoy_index,
            delta,
            effective_max_candidates, // ← Pass explicitly
            seed,
            verbose,
        );

        // Diversify if requested
        let graph = if diversify_prob > T::zero() {
            builder.diversify_graph(&build_graph.0, diversify_prob, seed)
        } else {
            build_graph.0
        };

        NNDescent {
            vectors_flat: builder.vectors_flat,
            dim: builder.dim,
            n: builder.n,
            metric: builder.metric,
            norms: builder.norms,
            graph,
            converged: build_graph.1,
        }
    }

    /// Main NN-Descent algorithm (low memory version)
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        k: usize,
        max_iter: usize,
        annoy_index: &AnnoyIndex<T>,
        delta: T,
        max_candidates: usize, // ← Explicit parameter
        seed: usize,
        verbose: bool,
    ) -> (Vec<Vec<(usize, T)>>, bool) {
        if verbose {
            println!(
                "Running NN-Descent: {} samples, k={}, max_candidates={}",
                self.n.separate_with_underscores(),
                k,
                max_candidates
            );
        }

        let mut converged: bool = false;

        let start = Instant::now();
        let mut graph = self.init_with_annoy(k, annoy_index);

        for iter in 0..max_iter {
            let updates = AtomicUsize::new(0);

            let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_add(iter as u64));

            let reverse_graph = Self::build_reverse_index(&graph);
            let (new_cands, old_cands) =
                self.build_candidates(&graph, &reverse_graph, max_candidates, &mut rng);

            self.mark_as_old(&mut graph, &new_cands);

            let all_updates = self.generate_updates(&new_cands, &old_cands, &graph);

            self.update_neighbours(&all_updates.concat(), &mut graph, &updates);

            let update_count = updates.load(Ordering::Relaxed);

            let update_rate =
                T::from_usize(update_count).unwrap() / T::from_usize(self.n * k).unwrap();

            if verbose {
                println!(
                    "  Iter {}: {} updates (rate={:.4})",
                    iter + 1,
                    update_count.separate_with_underscores(),
                    update_rate.to_f64().unwrap(),
                );
            }

            if update_rate < delta {
                if verbose {
                    println!("  Converged after {} iterations", iter + 1);
                }
                converged = true;
                break;
            }
        }

        if verbose {
            println!("Total time: {:.2?}", start.elapsed());
        }

        let res = graph
            .into_iter()
            .map(|neighbours| neighbours.into_iter().map(|n| (n.pid(), n.dist)).collect())
            .collect();

        (res, converged)
    }

    /// Check if algorithm has
    pub fn index_converged(&self) -> bool {
        self.converged
    }

    fn build_reverse_index(graph: &[Vec<Neighbour<T>>]) -> Vec<Vec<usize>> {
        let n = graph.len();
        let mut reverse: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, neighbours) in graph.iter().enumerate() {
            for n in neighbours {
                reverse[n.pid()].push(i);
            }
        }

        reverse
    }

    /// Build candidate lists (matches Python's new_build_candidates)
    fn build_candidates(
        &self,
        graph: &[Vec<Neighbour<T>>],
        reverse_graph: &[Vec<usize>],
        max_candidates: usize,
        rng: &mut SmallRng,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut new_heaps: Vec<BinaryHeap<(OrderedFloat<f64>, usize)>> =
            vec![BinaryHeap::new(); self.n];
        let mut old_heaps: Vec<BinaryHeap<(OrderedFloat<f64>, usize)>> =
            vec![BinaryHeap::new(); self.n];

        // Forward neighbours (your existing code)
        for i in 0..self.n {
            for neighbour in &graph[i] {
                let j = neighbour.pid();
                let priority = OrderedFloat(-rng.random::<f64>());

                if neighbour.is_new() {
                    Self::heap_push(&mut new_heaps[i], priority, j, max_candidates);
                    Self::heap_push(&mut new_heaps[j], priority, i, max_candidates);
                } else {
                    Self::heap_push(&mut old_heaps[i], priority, j, max_candidates);
                    Self::heap_push(&mut old_heaps[j], priority, i, max_candidates);
                }
            }
        }

        // Reverse neighbours - sample nodes that point TO i
        for i in 0..self.n {
            for &j in &reverse_graph[i] {
                if j == i {
                    continue;
                }

                // Find the neighbour entry in graph[j] that points to i
                let is_new = graph[j]
                    .iter()
                    .find(|n| n.pid() == i)
                    .map(|n| n.is_new())
                    .unwrap_or(false);

                let priority = OrderedFloat(-rng.random::<f64>());

                if is_new {
                    Self::heap_push(&mut new_heaps[i], priority, j, max_candidates);
                } else {
                    Self::heap_push(&mut old_heaps[i], priority, j, max_candidates);
                }
            }
        }

        let new_cands = new_heaps
            .into_iter()
            .map(|heap| heap.into_iter().map(|(_, idx)| idx).collect())
            .collect();
        let old_cands = old_heaps
            .into_iter()
            .map(|heap| heap.into_iter().map(|(_, idx)| idx).collect())
            .collect();

        (new_cands, old_cands)
    }

    #[inline]
    fn heap_push(
        heap: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
        priority: OrderedFloat<f64>,
        idx: usize,
        max_size: usize,
    ) {
        if heap.len() < max_size {
            heap.push((priority, idx));
        } else if let Some(&(worst, _)) = heap.peek() {
            if priority > worst {
                heap.pop();
                heap.push((priority, idx));
            }
        }
    }

    /// Mark neighbours as old (matches Python's flag updating in new_build_candidates)
    fn mark_as_old(&self, graph: &mut [Vec<Neighbour<T>>], new_cands: &[Vec<usize>]) {
        for i in 0..self.n {
            if new_cands[i].is_empty() {
                continue;
            }

            let cand_set: std::collections::HashSet<usize> = new_cands[i].iter().copied().collect();

            for neighbour in &mut graph[i] {
                if cand_set.contains(&neighbour.pid()) && neighbour.is_new() {
                    *neighbour = Neighbour::new(neighbour.pid(), neighbour.dist, false);
                }
            }
        }
    }

    #[inline]
    fn heap_push_dedup(
        &self,
        heap: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
        seen: &mut HashSet<usize>,
        priority: OrderedFloat<f64>,
        idx: usize,
        max_size: usize,
    ) {
        if seen.contains(&idx) {
            return;
        }

        if heap.len() < max_size {
            heap.push((priority, idx));
            seen.insert(idx);
        } else if let Some(&(worst, _)) = heap.peek() {
            if priority > worst {
                if let Some((_, evicted)) = heap.pop() {
                    seen.remove(&evicted);
                }
                heap.push((priority, idx));
                seen.insert(idx);
            }
        }
    }

    /// Generate distance updates (matches Python's generate_graph_updates)
    fn generate_updates(
        &self,
        new_cands: &[Vec<usize>],
        old_cands: &[Vec<usize>],
        graph: &[Vec<Neighbour<T>>],
    ) -> Vec<Vec<(usize, usize, T)>> {
        (0..self.n)
            .into_par_iter()
            .map(|i| {
                let mut updates = Vec::new();

                // Compare new-new pairs
                // CRITICAL FIX: Start at j, not j+1 to match Python
                for j in 0..new_cands[i].len() {
                    let p = new_cands[i][j];
                    if p >= self.n {
                        continue;
                    }

                    for k in j..new_cands[i].len() {
                        // ← Changed from j+1
                        let q = new_cands[i][k];
                        if q >= self.n {
                            continue;
                        }
                        if p == q {
                            continue;
                        }

                        let d = self.distance(p, q);
                        if self.should_add_edge(p, q, d, graph) {
                            updates.push((p, q, d));
                        }
                    }
                }

                // Compare new-old pairs
                for &p in &new_cands[i] {
                    if p >= self.n {
                        continue;
                    }
                    for &q in &old_cands[i] {
                        if q >= self.n {
                            continue;
                        }

                        let d = self.distance(p, q);
                        if self.should_add_edge(p, q, d, graph) {
                            updates.push((p, q, d));
                        }
                    }
                }

                updates
            })
            .collect()
    }

    #[inline]
    fn distance(&self, i: usize, j: usize) -> T {
        match self.metric {
            Dist::Euclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
        }
    }

    #[inline]
    fn should_add_edge(&self, p: usize, q: usize, dist: T, graph: &[Vec<Neighbour<T>>]) -> bool {
        let p_threshold = if graph[p].is_empty() {
            T::infinity()
        } else {
            graph[p].last().unwrap().dist
        }; // No multiplier!

        let q_threshold = if graph[q].is_empty() {
            T::infinity()
        } else {
            graph[q].last().unwrap().dist
        }; // No multiplier!

        dist <= p_threshold || dist <= q_threshold
    }

    /// Diversify graph (matches Python's diversify function)
    fn diversify_graph(
        &self,
        graph: &[Vec<(usize, T)>],
        prune_prob: T,
        seed: usize,
    ) -> Vec<Vec<(usize, T)>> {
        graph
            .par_iter()
            .enumerate()
            .map(|(i, neighbours)| {
                if neighbours.is_empty() {
                    return Vec::new();
                }

                let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_add(i as u64));
                let mut kept = vec![neighbours[0]]; // Always keep closest

                for &(cand_idx, cand_dist) in &neighbours[1..] {
                    let mut should_keep = true;

                    for &(kept_idx, kept_dist) in &kept {
                        let dist_to_kept = self.distance(cand_idx, kept_idx);

                        // Prune if candidate is closer to an already-kept neighbour
                        if kept_dist > T::from_f32(f32::EPSILON).unwrap()
                            && dist_to_kept < cand_dist
                            && rng.random::<f64>() < prune_prob.to_f64().unwrap()
                        {
                            should_keep = false;
                            break;
                        }
                    }

                    if should_keep {
                        kept.push((cand_idx, cand_dist));
                    }
                }

                kept
            })
            .collect()
    }

    /// Query for k nearest neighbours using greedy graph search
    pub fn query(&self, query_vec: &[T], k: usize, ef_search: usize) -> (Vec<usize>, Vec<T>) {
        assert_eq!(query_vec.len(), self.dim);

        let k = k.min(self.n);
        let ef = ef_search.max(k);

        let query_norm = if self.metric == Dist::Cosine {
            query_vec
                .iter()
                .map(|&x| x * x)
                .fold(T::zero(), |a, b| a + b)
                .sqrt()
        } else {
            T::one()
        };

        let compute_dist = |idx: usize| match self.metric {
            Dist::Euclidean => self.euclidean_distance_to_query(idx, query_vec),
            Dist::Cosine => self.cosine_distance_to_query(idx, query_vec, query_norm),
        };

        let mut visited = vec![false; self.n];
        let mut candidates: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::new();

        // Random entry points
        let mut rng =
            SmallRng::seed_from_u64(query_vec.iter().map(|x| x.to_u64().unwrap_or(0)).sum());
        for _ in 0..10.min(self.n) {
            let entry = rng.random_range(0..self.n);
            if !visited[entry] {
                visited[entry] = true;
                let dist = compute_dist(entry);
                candidates.push((OrderedFloat(-dist), entry));
                results.push((OrderedFloat(dist), entry));
            }
        }

        // Greedy beam search
        while let Some((OrderedFloat(neg_dist), current)) = candidates.pop() {
            let current_dist = -neg_dist;

            if results.len() >= ef && current_dist > results.peek().unwrap().0 .0 {
                continue;
            }

            for &(neighbour_idx, _) in &self.graph[current] {
                if visited[neighbour_idx] {
                    continue;
                }
                visited[neighbour_idx] = true;

                let dist = compute_dist(neighbour_idx);

                if results.len() < ef || dist < results.peek().unwrap().0 .0 {
                    if results.len() >= ef {
                        results.pop();
                    }
                    results.push((OrderedFloat(dist), neighbour_idx));
                    candidates.push((OrderedFloat(-dist), neighbour_idx));
                }
            }
        }

        // Extract top k
        let mut final_results: Vec<_> = results.into_iter().collect();
        final_results.sort_unstable_by_key(|&(dist, _)| dist);
        final_results.truncate(k);

        final_results
            .into_iter()
            .map(|(OrderedFloat(dist), idx)| (idx, dist))
            .unzip()
    }

    fn init_with_annoy(&self, k: usize, annoy: &AnnoyIndex<T>) -> Vec<Vec<Neighbour<T>>> {
        (0..self.n)
            .into_par_iter()
            .map(|i| {
                let query = &self.vectors_flat[i * self.dim..(i + 1) * self.dim];
                let search_k = self.n / 20;
                let (indices, distances) = annoy.query(query, k + 1, Some(search_k));

                indices
                    .into_iter()
                    .zip(distances)
                    .skip(1) // Skip self
                    .take(k)
                    .map(|(idx, dist)| Neighbour::new(idx, dist, true)) // All new initially
                    .collect()
            })
            .collect()
    }
}

// Update implementations for f32 and f64 (matches apply_graph_updates_low_memory)
impl UpdateNeighbours<f32> for NNDescent<f32> {
    fn update_neighbours(
        &self,
        updates: &[(usize, usize, f32)],
        graph: &mut [Vec<Neighbour<f32>>],
        updates_count: &AtomicUsize,
    ) {
        let mut per_node: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n];
        for &(p, q, d) in updates {
            if p < self.n && q < self.n {
                per_node[p].push((q, d));
                per_node[q].push((p, d));
            }
        }

        let new_graphs: Vec<Option<(Vec<Neighbour<f32>>, usize)>> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                if per_node[i].is_empty() {
                    return None;
                }

                HEAP_F32.with(|heap_cell| {
                    PID_SET.with(|set_cell| {
                        let mut heap = heap_cell.borrow_mut();
                        let mut pid_set = set_cell.borrow_mut();

                        heap.clear();
                        if pid_set.len() < self.n {
                            pid_set.resize(self.n, false);
                        }

                        let k = graph[i].len();
                        let mut edge_updates = 0usize;

                        for n in &graph[i] {
                            let pid = n.pid();
                            heap.push((OrderedFloat(n.dist), pid, n.is_new()));
                            pid_set[pid] = true;
                        }

                        for &(cand, dist) in &per_node[i] {
                            if pid_set[cand] {
                                continue;
                            }

                            if heap.len() < k {
                                heap.push((OrderedFloat(dist), cand, true));
                                pid_set[cand] = true;
                                edge_updates += 1;
                            } else if let Some(&(OrderedFloat(worst), _, _)) = heap.peek() {
                                if dist < worst {
                                    if let Some((_, old_pid, _)) = heap.pop() {
                                        pid_set[old_pid] = false;
                                    }
                                    heap.push((OrderedFloat(dist), cand, true));
                                    pid_set[cand] = true;
                                    edge_updates += 1;
                                }
                            }
                        }

                        if edge_updates > 0 {
                            let mut result: Vec<_> = heap
                                .drain()
                                .map(|(OrderedFloat(d), p, is_new)| {
                                    pid_set[p] = false;
                                    (d, p, is_new)
                                })
                                .collect();
                            result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            result.truncate(k);

                            Some((
                                result
                                    .into_iter()
                                    .map(|(d, p, is_new)| Neighbour::new(p, d, is_new))
                                    .collect(),
                                edge_updates,
                            ))
                        } else {
                            for (_, pid, _) in heap.iter() {
                                pid_set[*pid] = false;
                            }
                            None
                        }
                    })
                })
            })
            .collect();

        let mut total_edge_updates = 0;
        for (i, new_graph) in new_graphs.into_iter().enumerate() {
            if let Some((new_neighbours, edge_count)) = new_graph {
                graph[i] = new_neighbours;
                total_edge_updates += edge_count;
            }
        }

        updates_count.fetch_add(total_edge_updates, Ordering::Relaxed);
    }
}

impl UpdateNeighbours<f64> for NNDescent<f64> {
    fn update_neighbours(
        &self,
        updates: &[(usize, usize, f64)],
        graph: &mut [Vec<Neighbour<f64>>],
        updates_count: &AtomicUsize,
    ) {
        let mut per_node: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.n];
        for &(p, q, d) in updates {
            if p < self.n && q < self.n {
                per_node[p].push((q, d));
                per_node[q].push((p, d));
            }
        }

        let new_graphs: Vec<Option<(Vec<Neighbour<f64>>, usize)>> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                if per_node[i].is_empty() {
                    return None;
                }

                HEAP_F64.with(|heap_cell| {
                    PID_SET.with(|set_cell| {
                        let mut heap = heap_cell.borrow_mut();
                        let mut pid_set = set_cell.borrow_mut();

                        heap.clear();
                        if pid_set.len() < self.n {
                            pid_set.resize(self.n, false);
                        }

                        let k = graph[i].len();
                        let mut edge_updates = 0usize;

                        for n in &graph[i] {
                            let pid = n.pid();
                            heap.push((OrderedFloat(n.dist), pid, n.is_new()));
                            pid_set[pid] = true;
                        }

                        for &(cand, dist) in &per_node[i] {
                            if pid_set[cand] {
                                continue;
                            }

                            if heap.len() < k {
                                heap.push((OrderedFloat(dist), cand, true));
                                pid_set[cand] = true;
                                edge_updates += 1;
                            } else if let Some(&(OrderedFloat(worst), _, _)) = heap.peek() {
                                if dist < worst {
                                    if let Some((_, old_pid, _)) = heap.pop() {
                                        pid_set[old_pid] = false;
                                    }
                                    heap.push((OrderedFloat(dist), cand, true));
                                    pid_set[cand] = true;
                                    edge_updates += 1;
                                }
                            }
                        }

                        if edge_updates > 0 {
                            let mut result: Vec<_> = heap
                                .drain()
                                .map(|(OrderedFloat(d), p, is_new)| {
                                    pid_set[p] = false;
                                    (d, p, is_new)
                                })
                                .collect();
                            result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            result.truncate(k);

                            Some((
                                result
                                    .into_iter()
                                    .map(|(d, p, is_new)| Neighbour::new(p, d, is_new))
                                    .collect(),
                                edge_updates,
                            ))
                        } else {
                            for (_, pid, _) in heap.iter() {
                                pid_set[*pid] = false;
                            }
                            None
                        }
                    })
                })
            })
            .collect();

        let mut total_edge_updates = 0;
        for (i, new_graph) in new_graphs.into_iter().enumerate() {
            if let Some((new_neighbours, edge_count)) = new_graph {
                graph[i] = new_neighbours;
                total_edge_updates += edge_count;
            }
        }

        updates_count.fetch_add(total_edge_updates, Ordering::Relaxed);
    }
}

///////////
// Tests //
///////////

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use faer::Mat;

//     fn create_simple_matrix() -> Mat<f32> {
//         let data = [
//             1.0, 0.0, 0.0, // Point 0
//             0.0, 1.0, 0.0, // Point 1
//             0.0, 0.0, 1.0, // Point 2
//             1.0, 1.0, 0.0, // Point 3
//             1.0, 0.0, 1.0, // Point 4
//         ];
//         Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
//     }

//     #[test]
//     fn test_nndescent_build_euclidean() {
//         let mat = create_simple_matrix();
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), 5);
//         for neighbours in &index.graph {
//             assert!(neighbours.len() <= 3);
//         }
//     }

//     #[test]
//     fn test_nndescent_build_cosine() {
//         let mat = create_simple_matrix();
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Cosine,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), 5);
//     }

//     #[test]
//     fn test_nndescent_query() {
//         let mat = create_simple_matrix();
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         let query = vec![1.0, 0.0, 0.0];
//         let (indices, distances) = index.query(&query, 3, 50);

//         assert_eq!(indices.len(), 3);
//         assert_eq!(distances.len(), 3);
//         assert!(indices.contains(&0));
//     }

//     #[test]
//     fn test_nndescent_convergence() {
//         let mat = create_simple_matrix();

//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             100,
//             0.5,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), 5);
//     }

//     #[test]
//     fn test_nndescent_reproducibility() {
//         let mat = create_simple_matrix();

//         let graph1 = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         let graph2 = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(graph1.graph.len(), graph2.graph.len());
//     }

//     #[test]
//     fn test_nndescent_k_parameter() {
//         let mat = create_simple_matrix();

//         let graph_k2 = NNDescent::<f32>::new(
//             mat.as_ref(),
//             2,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         let graph_k4 = NNDescent::<f32>::new(
//             mat.as_ref(),
//             4,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         for neighbours in &graph_k2.graph {
//             assert!(neighbours.len() <= 2);
//         }

//         for neighbours in &graph_k4.graph {
//             assert!(neighbours.len() <= 4);
//         }
//     }

//     #[test]
//     fn test_nndescent_larger_dataset() {
//         let n = 50;
//         let dim = 10;
//         let mut data = Vec::with_capacity(n * dim);

//         for i in 0..n {
//             for j in 0..dim {
//                 data.push((i * j) as f32 / 10.0);
//             }
//         }

//         let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             10,
//             Dist::Euclidean,
//             15,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), n);

//         for neighbours in &index.graph {
//             assert!(neighbours.len() <= 10);
//             assert!(!neighbours.is_empty());
//         }
//     }

//     #[test]
//     fn test_nndescent_distance_ordering() {
//         let mat = create_simple_matrix();
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         for neighbours in &index.graph {
//             for i in 1..neighbours.len() {
//                 assert!(neighbours[i].1 >= neighbours[i - 1].1);
//             }
//         }
//     }

//     #[test]
//     fn test_nndescent_with_f64() {
//         let data = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
//         let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);

//         let index = NNDescent::<f64>::new(
//             mat.as_ref(),
//             2,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), 3);
//     }

//     #[test]
//     fn test_nndescent_quality() {
//         let n = 20;
//         let dim = 3;
//         let mut data = Vec::with_capacity(n * dim);

//         for i in 0..10 {
//             let offset = i as f32 * 0.1;
//             data.extend_from_slice(&[offset, 0.0, 0.0]);
//         }
//         for i in 0..10 {
//             let offset = 10.0 + i as f32 * 0.1;
//             data.extend_from_slice(&[offset, 0.0, 0.0]);
//         }

//         let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             5,
//             Dist::Euclidean,
//             20,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         let neighbours_0 = &index.graph[0];
//         let in_cluster = neighbours_0.iter().filter(|(idx, _)| *idx < 10).count();
//         assert!(in_cluster >= 3);

//         let neighbours_10 = &index.graph[10];
//         let in_cluster_2 = neighbours_10.iter().filter(|(idx, _)| *idx >= 10).count();
//         assert!(in_cluster_2 >= 3);
//     }
// }
