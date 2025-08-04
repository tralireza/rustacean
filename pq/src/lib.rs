//! # PQ (aka Heap) :: Rusty

/// 407h Trapping Rain Water II
struct Sol407;

impl Sol407 {
    pub fn trap_rain_water(height_map: Vec<Vec<i32>>) -> i32 {
        let (rows, cols) = (height_map.len(), height_map[0].len());

        #[derive(Debug, PartialEq, Eq, Ord, PartialOrd)]
        struct Cell {
            height: i32,
            r: i32,
            c: i32,
        }

        let mut visited = vec![vec![false; cols]; rows];

        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut pq = BinaryHeap::new();
        for r in 0..rows {
            for c in 0..cols {
                if r == 0 || c == 0 || r == rows - 1 || c == cols - 1 {
                    pq.push(Reverse(Cell {
                        height: height_map[r][c],
                        r: r as i32,
                        c: c as i32,
                    }));

                    visited[r][c] = true;
                }
            }
        }

        println!(" -> {pq:?}");
        println!(" -> {visited:?}");

        let mut trap_water = 0;
        let dirs = [-1, 0, 1, 0, -1];

        while let Some(Reverse(cell)) = pq.pop() {
            println!(" -> {cell:?}");

            (0..4).for_each(|i| {
                let (r, c) = (cell.r + dirs[i], cell.c + dirs[i + 1]);
                if 0 <= r
                    && r < rows as i32
                    && 0 <= c
                    && c < cols as i32
                    && !visited[r as usize][c as usize]
                {
                    visited[r as usize][c as usize] = true;

                    let h = height_map[r as usize][c as usize];
                    if h < cell.height {
                        trap_water += cell.height - h;
                    }

                    pq.push(Reverse(Cell {
                        height: h.max(cell.height),
                        r,
                        c,
                    }));
                }
            });
        }

        println!(" -> {visited:?}");

        trap_water
    }
}

/// 1046 Last Stone Weight
struct Sol1046 {}

impl Sol1046 {
    pub fn last_stone_weight(stones: Vec<i32>) -> i32 {
        use std::collections::BinaryHeap;

        let mut hvs = BinaryHeap::new();
        for stone in stones {
            hvs.push(stone);
        }

        while hvs.len() > 1 {
            let w1 = hvs.pop().unwrap();
            let w2 = hvs.pop().unwrap();
            if w1 > w2 {
                hvs.push(w1 - w2);
            }
        }

        hvs.pop().unwrap_or(0)
    }
}

/// 2231 Largest Number After Digit Swaps by Parity
struct Sol2231 {}

impl Sol2231 {
    pub fn largest_integer(mut num: i32) -> i32 {
        use std::collections::BinaryHeap;

        let mut qs = [BinaryHeap::new(), BinaryHeap::new()];
        let mut pars = vec![];

        while num > 0 {
            let parity = (num % 10) as usize & 1;

            pars.push(parity);
            qs[parity].push(num % 10);

            num /= 10;
        }
        println!("-> {qs:?}");

        let mut x = 0;
        for &parity in pars.iter().rev() {
            if let Some(digit) = qs[parity].pop() {
                x = 10 * x + digit;
            }
        }

        x
    }
}

/// 3066m Minimum Operations to Exceed Threshold Value II
struct Sol3066;

impl Sol3066 {
    pub fn min_operations(nums: Vec<i32>, k: i32) -> i32 {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut pq = BinaryHeap::new();
        for n in nums {
            pq.push(Reverse(n as usize));
        }

        let mut ops = 0;
        while let Some(&Reverse(x)) = pq.peek() {
            if x < k as usize {
                pq.pop();
                if let Some(Reverse(y)) = pq.pop() {
                    pq.push(Reverse(x.min(y) * 2 + x.max(y)));
                }
            } else {
                break;
            }
            ops += 1;
        }

        println!("-> {pq:?}");

        ops
    }
}

/// 3362m Zero Array Transformation III
struct Sol3362;

impl Sol3362 {
    /// 1 <= N, Q <= 10^5
    /// 0 <= N_ij <= 10^5
    pub fn max_removal(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> i32 {
        use std::collections::BinaryHeap;

        #[derive(Debug)]
        struct Fenwick {
            tree: Vec<i32>,
        }

        impl Fenwick {
            fn new(size: usize) -> Self {
                Fenwick {
                    tree: vec![0; size + 1],
                }
            }

            fn update(&mut self, mut i: usize, diff: i32) {
                while i < self.tree.len() {
                    self.tree[i] += diff;
                    i += i & (!i + 1);
                }
            }

            fn query(&self, mut i: usize) -> i32 {
                let mut r = 0;
                while i > 0 {
                    r += self.tree[i];
                    i -= i & (!i + 1);
                }
                r
            }
        }

        let mut fwt = Fenwick::new(nums.len() + 2);
        for query in &queries {
            fwt.update(query[0] as usize + 1, 1);
            fwt.update(query[1] as usize + 1 + 1, -1);
        }
        println!(
            "-> {:?} {fwt:?}",
            (1..=nums.len()).map(|i| fwt.query(i)).collect::<Vec<_>>()
        );

        let mut queries = queries;
        queries.sort_by(|q1, q2| {
            if q1[0] == q2[0] {
                q2[1].cmp(&q1[1])
            } else {
                q1[0].cmp(&q2[0])
            }
        });
        println!("-> {queries:?}");

        let mut pq = BinaryHeap::new();

        let (mut diff, mut diffs) = (0, vec![0; nums.len() + 1]);
        let mut q = 0;
        for (i, &n) in nums.iter().enumerate() {
            diff += diffs[i];

            while q < queries.len() && queries[q][0] as usize == i {
                pq.push(queries[q][1]);
                q += 1;
            }

            while diff < n {
                match pq.peek() {
                    Some(&right) if right >= i as i32 => {
                        diff += 1;
                        diffs[right as usize + 1] -= 1;

                        pq.pop();
                    }
                    _ => break,
                }
            }

            if diff < n {
                return -1;
            }
        }

        pq.len() as _
    }
}

#[cfg(test)]
mod tests;
