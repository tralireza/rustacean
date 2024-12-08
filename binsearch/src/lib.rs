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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution1760() {
        assert_eq!(Solution1760::minimum_size(vec![9], 2), 3);
        assert_eq!(Solution1760::minimum_size(vec![2, 4, 8, 2], 4), 2);
    }
}
