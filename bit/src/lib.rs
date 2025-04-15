//! # BIT Binary Indexed Tree, Segment Tree, Ordered Set

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

#[cfg(test)]
mod tests;
