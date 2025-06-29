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

/// 1498m Number of Subsequences That Satisfy the Give Sum Condition
struct Sol1498 {}

impl Sol1498 {
    /// 1 <= N <= 10^5
    /// 1 <= N_i, target <= 10^6
    pub fn num_subseq(mut nums: Vec<i32>, target: i32) -> i32 {
        const M: i64 = 1e9 as i64 + 7;

        let mpower = |mut exponent| {
            let mut mpower = 1;
            let mut base = 2;
            while exponent > 0 {
                if exponent & 1 == 1 {
                    mpower = (base * mpower) % M;
                }
                base = (base * base) % M;
                exponent >>= 1;
            }

            mpower
        };
        let _ = mpower(0);

        let mut powers = vec![1; nums.len()];
        for x in 1..powers.len() {
            powers[x] = (powers[x - 1] * 2) % M;
        }

        nums.sort_unstable();

        let mut count = 0;
        let (mut left, mut right) = (0, nums.len() - 1);
        while left <= right {
            if nums[left] + nums[right] <= target {
                count = (count + powers[right - left]) % M;
                left += 1;
            } else {
                if right == 0 {
                    break;
                }
                right -= 1;
            }
        }

        count as _
    }
}

/// 2040h Kth Smallest Product of Two Sorted Arrays
struct Sol2040 {}

impl Sol2040 {
    /// 1 <= N <= 5*10^4
    /// -10^5 <= N_i <= 10^5
    pub fn kth_smallest_product(nums1: Vec<i32>, nums2: Vec<i32>, k: i64) -> i64 {
        let count_smaller = |v| {
            let mut count = 0;
            for &n in &nums1 {
                let (mut l, mut r) = (0, nums2.len());
                while l < r {
                    let m = l + ((r - l) >> 1);
                    let p = nums2[m] as i64 * n as i64;
                    if n >= 0 && p <= v || n < 0 && p > v {
                        l = m + 1;
                    } else {
                        r = m;
                    }
                }

                count += if n >= 0 { l } else { nums2.len() - l } as i64;
            }

            count
        };

        let (mut l, mut r) = (-1e10 as i64, 1e10 as i64);
        while l < r {
            let m = l + ((r - l) >> 1);
            println!("-> {l} {m} {r}");

            if count_smaller(m) < k {
                l = m + 1;
            } else {
                r = m;
            }
        }

        println!(":: {l}");
        l
    }
}

/// 2071h Maximum Number of Tasks You Can Assign
struct Sol2071;

impl Sol2071 {
    pub fn max_task_assign(tasks: Vec<i32>, workers: Vec<i32>, pills: i32, strength: i32) -> i32 {
        let (mut tasks, mut workers) = (tasks, workers);

        tasks.sort_unstable();
        workers.sort_unstable();

        fn check(tasks: &[i32], workers: &[i32], mut pills: i32, strength: i32) -> bool {
            use std::collections::BTreeMap;

            let mut q = BTreeMap::new();
            for wkr in workers {
                q.entry(wkr).and_modify(|f| *f += 1).or_insert(1);
            }

            for tsk in tasks.iter().rev() {
                if let Some((&wkr, _)) = q.iter().next_back() {
                    if wkr >= tsk {
                        q.entry(wkr).and_modify(|f| *f -= 1);
                        if q[wkr] == 0 {
                            q.remove(wkr);
                        }
                    } else {
                        if pills == 0 {
                            return false;
                        }

                        if let Some((&wkr, _)) = q.range(tsk - strength..).next() {
                            pills -= 1;

                            q.entry(wkr).and_modify(|f| *f -= 1);
                            if q[wkr] == 0 {
                                q.remove(wkr);
                            }
                        } else {
                            return false;
                        }
                    }
                }
            }

            true
        }

        let (mut l, mut r) = (0, tasks.len().min(workers.len()));
        while l < r {
            let m = l + ((r - l + 1) >> 1);

            if check(&tasks[..m], &workers[workers.len() - m..], pills, strength) {
                l = m;
            } else {
                r = m - 1;
            }
        }

        l as _
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

/// 2563m Count the Number of Fair Pairs
struct Sol2563;

impl Sol2563 {
    /// 1 <= N <= 10^5
    /// -10^9 <= N_i, lower, upper <= 10^9
    pub fn count_fair_pairs(nums: Vec<i32>, lower: i32, upper: i32) -> i64 {
        let mut nums = nums;
        nums.sort_unstable();

        println!("** {:?}", nums);

        fn bsearch(nums: &[i32], l: usize, target: i32) -> usize {
            let (mut l, mut r) = (l, nums.len());
            while l < r {
                let m = l + ((r - l) >> 1);
                if nums[m] < target {
                    l = m + 1;
                } else {
                    r = m;
                }
            }

            l
        }

        nums.iter().enumerate().fold(0, |count, (l, &n)| {
            let r_low = bsearch(&nums, l, lower - n);
            let r_up = bsearch(&nums, l, upper - n + 1);

            count + r_up - r_low
        }) as i64
    }
}

/// 2594m Minimum Time to Repair Cars
struct Sol2594;

impl Sol2594 {
    /// 1 <= Rank_i <= 100
    /// 1 <= N <= 10^5
    pub fn repair_cars(ranks: Vec<i32>, cars: i32) -> i64 {
        let (mut l, mut r) = (
            1,
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

/// 2616m Minimize the Maximum Difference of Pairs
struct Sol2616 {}

impl Sol2616 {
    /// 1 <= N <= 10^5, 0 <= N_i <= 10^9
    /// 0 <= p <= N/2
    pub fn minimize_max(nums: Vec<i32>, p: i32) -> i32 {
        let mut nums = nums;
        nums.sort_unstable();

        let count_pairs = |m| {
            let mut count = 0;
            let mut i = 0;
            while i < nums.len() - 1 {
                if nums[i + 1] - nums[i] <= m {
                    count += 1;
                    i += 1;
                }
                i += 1;
            }

            count
        };

        let (mut l, mut r) = (0, nums.last().unwrap() - nums[0]);
        while l < r {
            let m = l + ((r - l) >> 1);

            println!("-> {l} {m} {r}");

            if count_pairs(m) >= p {
                r = m;
            } else {
                l = m + 1;
            }
        }

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
