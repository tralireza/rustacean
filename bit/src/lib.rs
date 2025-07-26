//! # BIT Binary Indexed Tree, Segment Tree, Ordered Set

/// 493h Reverse Pairs
struct Sol493 {}

impl Sol493 {
    /// 1 <= N <= 5*10^4
    /// -2^31 <= N_i <= 2^31-1
    pub fn reverse_pairs(nums: Vec<i32>) -> i32 {
        let brute_force = || {
            nums.iter()
                .map(|&n| 2 * n as i64)
                .enumerate()
                .fold(0, |r, (i, rval)| {
                    r + nums
                        .iter()
                        .take(i)
                        .fold(0, |r, &n| if n as i64 > rval { r + 1 } else { r })
                })
        };

        println!(":: {} (Brute Force)", brute_force());

        let mut bits = vec![0; nums.len() + 1];
        fn bits_update(bits: &mut [i64], mut i: usize, diff: i32) {
            while i > 0 {
                bits[i] += diff as i64;
                i -= i & (!i + 1);
            }
        }
        fn bits_query(bits: &[i64], mut i: usize) -> i64 {
            let mut v = 0;
            while i < bits.len() {
                v += bits[i];
                i += i & (!i + 1);
            }
            v
        }

        let mut sorted: Vec<_> = nums.iter().map(|&n| n as i64).collect();
        sorted.sort_unstable();

        fn bsearch(sorted: &[i64], t: i64) -> usize {
            let (mut l, mut r) = (0, sorted.len());
            while l < r {
                let m = l + ((r - l) >> 1);
                if sorted[m] < t {
                    l = m + 1;
                } else {
                    r = m;
                }
            }

            l
        }

        let mut count = 0;
        for n in nums {
            count += bits_query(&mut bits, bsearch(&sorted, 2 * n as i64 + 1) + 1);
            bits_update(&mut bits, bsearch(&sorted, n as i64) + 1, 1);
        }

        count as _
    }
}

/// 218h The Skyline Problems
struct Sol218;

impl Sol218 {
    pub fn get_skyline(buildings: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        use std::collections::{BTreeSet, BinaryHeap};

        println!("|| {:?}", buildings);

        let mut sweep = BTreeSet::new();
        for building in buildings.iter() {
            sweep.insert(building[0]);
            sweep.insert(building[1]);
        }

        println!("-> {:?}", sweep);

        let mut pq = BinaryHeap::new();

        let (mut i, mut h) = (0, 0);
        let mut skyline = vec![];

        for &x in sweep.iter() {
            while i < buildings.len() && buildings[i][0] == x {
                pq.push((buildings[i][2], buildings[i][0], buildings[i][1]));
                i += 1;
            }

            while let Some(t) = pq.peek() {
                if t.2 <= x {
                    pq.pop();
                } else {
                    break;
                }
            }

            if let Some(t) = pq.peek() {
                if t.0 != h {
                    skyline.push(vec![x, t.0]);
                    h = t.0;
                }
            } else {
                skyline.push(vec![x, 0]);
                h = 0;
            }
        }

        println!(":: {:?}", skyline);

        skyline
    }
}

/// 315h Count of Smaller Numbers After Self
struct Sol315;

impl Sol315 {
    /// 1 <= N <= 10^5
    /// -10^4 <= A_i <= 10^4
    pub fn count_smaller(nums: Vec<i32>) -> Vec<i32> {
        struct Bit {
            fnodes: Vec<i32>,
        }

        impl Bit {
            fn new(size: usize) -> Self {
                Bit {
                    fnodes: vec![0; size],
                }
            }

            fn update(&mut self, mut i: i32) {
                while (i as usize) < self.fnodes.len() {
                    self.fnodes[i as usize] += 1;
                    i += i & -i;
                }
            }

            fn query(&self, mut i: i32) -> i32 {
                let mut v = 0;
                while i > 0 {
                    v += self.fnodes[i as usize];
                    i -= i & -i;
                }

                v
            }
        }

        let mut fenwick = Bit::new(2*10_000 + 1 /* Shift 0 -> 1 */ + 1);

        let mut rst = vec![];
        for n in nums.iter().rev() {
            rst.push(fenwick.query(n - 1 + 10_000 + 1));
            fenwick.update(n + 10_000 + 1);
        }

        rst.reverse();

        rst
    }
}

/// 2179h Count Good Triplets in an Array
struct Sol2179;

impl Sol2179 {
    /// 3 <= n <= 10^5
    /// 0 <= N_i <= n-1
    pub fn good_triplets(nums1: Vec<i32>, nums2: Vec<i32>) -> i64 {
        #[derive(Debug)]
        struct Fenwick {
            nodes: Vec<i32>,
        }

        impl Fenwick {
            fn new(size: usize) -> Self {
                Fenwick {
                    nodes: vec![0; size],
                }
            }

            fn update(&mut self, mut i: usize, diff: i32) {
                while (i as usize) < self.nodes.len() {
                    self.nodes[i as usize] += diff;
                    i += i & (!i + 1); // i & -i: 2's Compliment
                }
            }

            fn query(&self, mut i: usize) -> i32 {
                let mut v = 0;
                while i > 0 {
                    v += self.nodes[i as usize];
                    i -= i & (!i + 1); // i & -i: 2's Compliment
                }
                v
            }
        }

        let n = nums1.len() & nums2.len();

        let mut map = vec![0; n];
        for (i, &n) in nums2.iter().enumerate() {
            map[n as usize] = i;
        }
        println!("-> {:?}", map);

        let mut rmap = vec![0; n];
        for (i, &n) in nums1.iter().enumerate() {
            rmap[map[n as usize]] = i;
        }
        println!("-> {:?}", rmap);

        let mut fenwick = Fenwick::new(n + 1);

        let mut count = 0;
        for v in 0..n {
            let j = rmap[v];

            let left = fenwick.query(j + 1);
            fenwick.update(j + 1, 1);

            let right = (n - 1 - j) - (v - left as usize);
            count += right as i64 * left as i64;
        }

        println!(":: {}", count);

        count
    }
}

/// 3480h Maximize Subarrays After Removing One Conflicting Pair
struct Sol3480 {}

impl Sol3480 {
    pub fn max_subarrays(n: i32, mut conflicting_pairs: Vec<Vec<i32>>) -> i64 {
        use std::cmp::Reverse;

        for pair in conflicting_pairs.iter_mut() {
            if pair[0] > pair[1] {
                pair.swap(0, 1);
            }
        }

        conflicting_pairs.sort_unstable_by_key(|p| Reverse(p[0]));
        println!("-> {conflicting_pairs:?}");

        let mut conflicting_pairs = conflicting_pairs.into_iter().peekable();

        let mut right = n + 1;
        let mut correction = 0;
        let (mut x_penalty, mut penalty) = (0, 0);

        let mut subarray = 0;

        for b in (1..=n).rev() {
            while let Some(p) = conflicting_pairs.next_if(|p| p[0] >= b) {
                if right > p[1] {
                    correction = right - p[1];
                    right = p[1];
                    penalty = 0;
                } else {
                    correction = correction.min(p[1] - right);
                }
            }

            subarray += (right - b) as i64;

            penalty += correction as i64;
            x_penalty = x_penalty.max(penalty);
        }

        subarray + x_penalty
    }
}

#[cfg(test)]
mod tests;
