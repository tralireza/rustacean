//! # Rust :: Binary Search

/// 153m Find Minimum in Rotated Sorted Array
struct Sol153;

impl Sol153 {
    pub fn find_min(nums: Vec<i32>) -> i32 {
        let (mut l, mut r) = (0, nums.len() - 1);

        while l < r {
            let m = l + ((r - l) >> 1);

            if nums[m] > nums[r] {
                l = m + 1;
            } else {
                r = m;
            }
        }

        nums[l]
    }
}

/// 154h Find Minimum in Rotated Sorted Array II
struct Sol154;

impl Sol154 {
    pub fn find_min(nums: Vec<i32>) -> i32 {
        let (mut l, mut r) = (0, nums.len() - 1);

        while l < r {
            let m = l + ((r - l) >> 1);

            if nums[m] > nums[r] {
                l = m + 1;
            } else if nums[m] < nums[r] {
                r = m;
            } else {
                r -= 1;
            }
        }

        nums[l]
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
