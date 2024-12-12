// 1760m Minimum Limit of Balls in a Bag
struct Solution1760 {}

impl Solution1760 {
    pub fn minimum_size(nums: Vec<i32>, max_operations: i32) -> i32 {
        let (mut l, mut r) = (1, *nums.iter().max().unwrap());

        let possible = |m| {
            let mut ops = 0;
            for n in &nums {
                ops += (n - 1) / m;
                if ops > max_operations {
                    return false;
                }
            }

            true
        };

        while l < r {
            let m = l + ((r - l) >> 1);
            if possible(m) {
                r = m;
            } else {
                l = m + 1;
            }
        }

        l
    }
}

// 2779m Maximum Beauty of an Array After Applying Operations
struct Solution2779 {}

impl Solution2779 {
    pub fn maximum_beauty(nums: Vec<i32>, k: i32) -> i32 {
        use std::iter::successors;

        let mut nums = nums;
        nums.sort();

        successors(Some((0usize, 0usize)), |&(l, r)| {
            let r = r + 1;
            if r >= nums.len() {
                None
            } else if nums[r] - nums[l] <= 2*k {
                Some((l, r))
            } else {
                let uval = nums[r];
                let l = (l + 1..=r)
                    .skip_while(|&i| uval - nums[i] > 2*k)
                    .nth(0)
                    .unwrap();
                Some((l, r))
            }
        })
            .map(|(l, r)| (r - l) as i32 + 1)
            .max()
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution1760() {
        assert_eq!(Solution1760::minimum_size(vec![9], 2), 3);
        assert_eq!(Solution1760::minimum_size(vec![2, 4, 8, 2], 4), 2);
    }

    #[test]
    fn test_solution2779() {
        assert_eq!(Solution2779::maximum_beauty(vec![4, 6, 1, 2], 2), 3);
        assert_eq!(Solution2779::maximum_beauty(vec![1, 1, 1, 1], 10), 4);
    }
}
