//! # Bits (aka Bitwise)

/// 868 Binary Gap
struct Sol868;

impl Sol868 {
    pub fn binary_gap(n: i32) -> i32 {
        let (mut dist, mut cur) = (0, -32);

        let mut n = n;
        while n > 0 {
            cur += 1;
            dist = dist.max(cur);
            if n & 1 == 1 {
                cur = 0;
            }
            n >>= 1;
        }

        dist
    }
}

/// 1863 Sum of All Subset XOR Totals
struct Sol1863;

impl Sol1863 {
    pub fn subset_xor_sum(nums: Vec<i32>) -> i32 {
        fn search(nums: &[i32], start: usize, xor: i32) -> i32 {
            if start == nums.len() {
                return xor;
            }

            search(nums, start + 1, xor) + search(nums, start + 1, nums[start] ^ xor)
        }

        search(&nums, 0, 0)
    }

    fn subset_xor_sum_bitwise(nums: Vec<i32>) -> i32 {
        let mut xsum = 0;
        for n in &nums {
            xsum |= n;
        }

        xsum << (nums.len() - 1)
    }
}

#[cfg(test)]
mod tests;
