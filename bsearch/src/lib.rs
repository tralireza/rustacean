//! # Rust :: Binary Search

/// 153m Find Minimum in Rotated Sorted Array
struct Sol153;

impl Sol153 {
    pub fn find_min(nums: Vec<i32>) -> i32 {
        let (mut l, mut r) = (0, nums.len() - 1);
        while l < r {
            let m = l + ((r - l) >> 1);
            match nums[m].cmp(&nums[r]) {
                std::cmp::Ordering::Greater => l = m + 1,
                _ => r = m,
            }
        }

        nums[l]
    }
}

/// 154h Find Minimum in Rotated Sorted Array II
struct Sol154;

impl Sol154 {
    pub fn find_min(nums: Vec<i32>) -> i32 {
        use std::cmp::Ordering::*;

        let (mut l, mut r) = (0, nums.len() - 1);
        while l < r {
            let m = l + ((r - l) >> 1);
            match nums[m].cmp(&nums[r]) {
                Greater => l = m + 1,
                Less => r = m,
                _ => r -= 1,
            }
        }

        nums[l]
    }
}

/// 315h Count of Smaller Numbers After Self
struct Sol315;

impl Sol315 {
    pub fn count_smaller(nums: Vec<i32>) -> Vec<i32> {
        let mut data = vec![];
        for (i, &n) in nums.iter().enumerate() {
            data.push((n, i, 0usize));
        }

        let mut bfr = data.to_vec();

        fn merge(
            data: &mut Vec<(i32, usize, usize)>,
            bfr: &mut Vec<(i32, usize, usize)>,
            l: usize,
            r: usize,
        ) {
            if l >= r {
                return;
            }

            let m = l + ((r - l) >> 1);
            merge(bfr, data, l, m);
            merge(bfr, data, m + 1, r);

            let mut p = l;
            let mut left = l;
            let mut right = m + 1;

            let mut smaller = 0;
            while left <= m && right <= r {
                if bfr[left].0 <= bfr[right].0 {
                    data[p] = bfr[left];
                    data[p].2 += smaller;

                    left += 1;
                } else {
                    smaller += 1;
                    data[p] = bfr[right];

                    right += 1;
                }
                p += 1;
            }

            while left <= m {
                data[p] = bfr[left];
                data[p].2 += smaller;

                left += 1;
                p += 1;
            }
            while right <= r {
                data[p] = bfr[right];

                right += 1;
                p += 1;
            }
        }

        let n = data.len();
        merge(&mut data, &mut bfr, 0, n - 1);

        println!("-> {:?}", data);

        let mut rst = vec![0; data.len()];
        for (_, i, smaller) in data {
            rst[i] = smaller as i32;
        }

        rst
    }

    /// 1 <= N <= 10^5
    /// -10^4 <= V_i <= 10^4
    fn bit_count_smaller(nums: Vec<i32>) -> Vec<i32> {
        struct BITree {
            tree: Vec<i32>,
        }

        impl BITree {
            fn new(size: usize) -> Self {
                BITree {
                    tree: vec![0; size],
                }
            }

            fn update(&mut self, mut p: i32, value: i32) {
                while p < self.tree.len() as i32 {
                    self.tree[p as usize] += value;
                    p |= p + 1;
                }
            }

            fn query(&self, mut p: i32) -> i32 {
                let mut v = 0;
                while p >= 0 {
                    v += self.tree[p as usize];
                    p &= p + 1;
                    p -= 1;
                }
                v
            }
        }

        const MAX: i32 = 10_000;

        let mut rst = vec![];

        let mut fwt = BITree::new(2 * MAX as usize + 1);
        for p in (0..nums.len()).rev() {
            rst.push(fwt.query(nums[p] + MAX - 1));
            fwt.update(nums[p] + MAX, 1);
        }

        rst.reverse();
        println!(":: {:?}", rst);

        rst
    }
}

/// 704 Binary Search
struct Sol704;

impl Sol704 {
    pub fn search(nums: Vec<i32>, target: i32) -> i32 {
        use std::cmp::Ordering::*;

        let (mut left, mut right) = (0, nums.len() as i32 - 1);
        while left <= right {
            let m = left + ((right - left) >> 1);
            match nums[m as usize].cmp(&target) {
                Equal => return m,
                Greater => right = m - 1,
                Less => left = m + 1,
            }
        }
        -1
    }
}

/// 2226m Maximum Candies Allocated to K Children
struct Sol2226;

impl Sol2226 {
    /// 1 <= C_i <= 10^7
    pub fn maximum_candies(candies: Vec<i32>, k: i64) -> i32 {
        let possible = |m| -> bool {
            if m == 0 {
                return true;
            }

            let mut t = 0;
            for c in &candies {
                t += (c / m) as i64;
            }
            t >= k
        };

        let (mut l, mut r) = (0, 1e7 as i32);
        while l <= r {
            let m = l + ((r - l) >> 1);
            println!("-> {:?}", (l, m, r));
            if possible(m) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }

        println!(":: {}", r);
        r
    }
}

/// 2529 Maximum Count of Positive Integer and Negative Integer
struct Sol2529;

impl Sol2529 {
    pub fn maximum_count(nums: Vec<i32>) -> i32 {
        fn bsleft(nums: &[i32], t: i32) -> i32 {
            let (mut l, mut r) = (0, nums.len() as i32);
            while l < r {
                let m = l + ((r - l) >> 1);
                match nums[m as usize].cmp(&t) {
                    std::cmp::Ordering::Less => l = m + 1,
                    _ => r = m,
                }
            }
            l
        }

        fn bsright(nums: &[i32], t: i32) -> i32 {
            let (mut l, mut r) = (0, nums.len() as i32);
            while l < r {
                let m = l + ((r - l) >> 1);
                match nums[m as usize].cmp(&t) {
                    std::cmp::Ordering::Greater => r = m,
                    _ => l = m + 1,
                }
            }
            r
        }

        (nums.len() as i32 - bsright(&nums, 0)).max(bsleft(&nums, 0))
    }
}

/// 2560m House Robber IV
struct Sol2560;

impl Sol2560 {
    pub fn min_capability(nums: Vec<i32>, k: i32) -> i32 {
        match (nums.iter().min(), nums.iter().max()) {
            (Some(&(mut l)), Some(&(mut r))) => {
                while l < r {
                    let m = l + ((r - l) >> 1);

                    let mut steals = 0;
                    let mut p = 0;
                    while p < nums.len() {
                        if nums[p] <= m {
                            steals += 1;
                            p += 1;
                        }
                        p += 1;
                    }

                    if steals >= k {
                        r = m;
                    } else {
                        l = m + 1;
                    }
                }
                l
            }
            _ => 0,
        }
    }
}

/// 2594m Minimum Time to Repair Cars
struct Sol2594;

impl Sol2594 {
    /// 1 <= Rank_i <= 100
    /// 1 <= N <= 10^5
    pub fn repair_cars(ranks: Vec<i32>, cars: i32) -> i64 {
        let (mut l, mut r) = (
            1 as i64,
            match ranks.iter().min() {
                Some(&r) => r as i64 * cars as i64 * cars as i64,
                _ => i64::MAX,
            },
        );

        while l <= r {
            let m = l + ((r - l) >> 1);
            println!("-> {:?}", (l, m, r));

            let mut repairs = 0;
            for &r in &ranks {
                repairs += (m / r as i64).isqrt();
            }

            if repairs >= cars as i64 {
                r = m - 1;
            } else {
                l = m + 1;
            }
        }

        println!(":: {}", l);

        l
    }
}

/// 3356m Zero Array Transformation II
struct Sol3356;

impl Sol3356 {
    pub fn min_zero_array(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> i32 {
        let possible = |k| -> bool {
            let mut diffs = vec![0; nums.len() + 1];

            for i in 0..k {
                let qry = &queries[i as usize];
                diffs[qry[0] as usize] += qry[2];
                diffs[qry[1] as usize + 1] -= qry[2]
            }

            let mut tsum = 0;
            for (&n, &s) in nums.iter().zip(diffs.iter()) {
                tsum += s;
                if n > tsum {
                    return false;
                }
            }

            true
        };

        if !possible(queries.len() as i32) {
            return -1;
        }

        let (mut l, mut r) = (0, queries.len() as i32);
        while l <= r {
            let m = l + ((r - l) >> 1);
            if possible(m) {
                r = m - 1;
            } else {
                l = m + 1;
            }
        }

        l
    }
}

#[cfg(test)]
mod tests;
