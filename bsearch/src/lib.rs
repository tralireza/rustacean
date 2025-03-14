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
