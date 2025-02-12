//! # Rust :: Binary Search

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_154() {
        assert_eq!(Sol154::find_min(vec![1, 3, 5]), 1);
        assert_eq!(Sol154::find_min(vec![2, 2, 2, 0, 1]), 0);
    }
}
