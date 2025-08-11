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

/// 2438m Range Product Queries of Powers
struct Sol2438 {}

impl Sol2438 {
    pub fn product_queries(mut n: i32, queries: Vec<Vec<i32>>) -> Vec<i32> {
        let mut powers = vec![];

        let mut power = 1;
        while n > 0 {
            if n & 1 == 1 {
                powers.push(power);
            }
            power <<= 1;
            n >>= 1;
        }

        powers.sort();
        println!("-> {powers:?}");

        const M: i64 = 1e9 as i64 + 7;
        queries
            .iter()
            .map(|query| (query[0] as usize, query[1] as usize))
            .map(|(l, r)| {
                powers
                    .iter()
                    .take(r + 1)
                    .skip(l)
                    .fold(1, |prd, &n| prd * n % M)
            })
            .map(|prd| prd as i32)
            .collect()
    }
}

#[cfg(test)]
mod tests;
