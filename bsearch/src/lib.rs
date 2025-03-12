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

#[cfg(test)]
mod tests;
