use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: usize,
    v: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| self.v.cmp(&other.v))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Edge {
    node: usize,
    cost: usize,
}

// Retruns the shortest path from `src` to `dst` in graph of `ladj`, if any
fn shortest_path(ladj: &Vec<Vec<Edge>>, src: usize, dst: usize) -> Option<usize> {
    let mut dist: Vec<_> = (0..ladj.len()).map(|_| usize::MAX).collect();
    let mut pq = BinaryHeap::new();

    dist[src] = 0;
    pq.push(State { cost: 0, v: src });

    while let Some(State { cost, v }) = pq.pop() {
        if v == dst {
            return Some(cost);
        }

        if cost > dist[v] {
            continue;
        }

        for u in &ladj[v] {
            let next = State {
                cost: cost + u.cost,
                v: u.node,
            };
            if next.cost < dist[next.v] {
                dist[next.v] = next.cost;
                pq.push(next);
            }
        }
    }

    None
}

fn main() {
    let graph = vec![
        vec![Edge { node: 2, cost: 10 }, Edge { node: 1, cost: 1 }],
        vec![Edge { node: 3, cost: 2 }],
        vec![
            Edge { node: 1, cost: 1 },
            Edge { node: 3, cost: 3 },
            Edge { node: 4, cost: 1 },
        ],
        vec![Edge { node: 4, cost: 2 }, Edge { node: 0, cost: 7 }],
        vec![],
    ];

    println!("4 -> 0 :: {:?}", shortest_path(&graph, 4, 0));
    assert_eq!(shortest_path(&graph, 4, 0), None);

    println!("2 -> 0 :: {:?}", shortest_path(&graph, 2, 0));
    assert_eq!(shortest_path(&graph, 2, 0), Some(10));
}
