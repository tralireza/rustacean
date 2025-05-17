//! # Sorting

/// 75m Sort Colors
struct Sol75;

impl Sol75 {
    pub fn sort_colors(nums: &mut Vec<i32>) {
        /// Flag Colors Sorting
        fn sort_flag_colors(nums: &mut Vec<i32>) {
            let (mut color0, mut color1, mut color2) = (0, 0, nums.len() - 1);
            while color1 <= color2 && color2 < usize::MAX {
                match nums[color1] {
                    2 => {
                        nums.swap(color1, color2);
                        color2 = color2.wrapping_sub(1);
                    }
                    1 => {
                        color1 += 1;
                    }
                    0 => {
                        nums.swap(color0, color1);
                        color0 += 1;
                        color1 += 1;
                    }
                    _ => {}
                }
            }

            println!(":: {nums:?}");
        }
        sort_flag_colors(&mut nums.to_vec());

        let mut wtr = nums.len() - 1;

        let mut color2 = 0;
        while color2 <= wtr && wtr < usize::MAX {
            match nums[color2] {
                2 => {
                    nums[color2] = nums[wtr];
                    nums[wtr] = 2;
                    wtr = wtr.wrapping_sub(1);
                }
                _ => {
                    color2 += 1;
                }
            }
        }

        let mut color1 = 0;
        while color1 <= wtr && wtr < usize::MAX {
            match nums[color1] {
                1 => {
                    nums[color1] = nums[wtr];
                    nums[wtr] = 1;
                    wtr = wtr.wrapping_sub(1);
                }
                _ => {
                    color1 += 1;
                }
            }
        }

        println!(":: {nums:?}");
    }
}

/// 220h Contains Duplicate III
struct Sol220;

impl Sol220 {
    pub fn contains_nearby_almost_duplicate(
        nums: Vec<i32>,
        index_diff: i32,
        value_diff: i32,
    ) -> bool {
        use std::collections::BTreeSet;

        println!("== {:?}", (&nums, index_diff, value_diff));

        let mut oset = BTreeSet::new();
        for (i, &n) in nums.iter().enumerate() {
            println!("-> {:?}", (n, &oset));

            if i > index_diff as usize {
                let drop = nums[i - index_diff as usize - 1];
                if n == drop {
                    continue;
                }

                oset.remove(&drop);
            }

            if oset.range(n - value_diff..=value_diff + n).count() > 0 {
                return true;
            }

            oset.insert(n);
        }

        false
    }
}

/// 905 Sort Array By Parity
struct Sol905;

impl Sol905 {
    pub fn sort_array_by_parity(nums: Vec<i32>) -> Vec<i32> {
        use std::collections::VecDeque;

        let mut rst = VecDeque::new();
        nums.into_iter().for_each(|n| match n & 1 {
            1 => rst.push_back(n),
            _ => rst.push_front(n),
        });

        rst.into()
    }
}

/// 2551h Put Marbles in Bags
struct Sol2551;

impl Sol2551 {
    pub fn put_marbles(weights: Vec<i32>, k: i32) -> i64 {
        let mut pairs: Vec<i64> = weights.windows(2).map(|w| (w[0] + w[1]) as i64).collect();
        pairs.sort_unstable();

        pairs.iter().skip(weights.len() - k as usize).sum::<i64>()
            - pairs.iter().take(k as usize - 1).sum::<i64>()
    }
}

/// 2948m Make Lexicographically Smallest Array by Swapping Elements
struct Sol2948;

impl Sol2948 {
    pub fn lexicographically_smallest_array(nums: Vec<i32>, limit: i32) -> Vec<i32> {
        let mut nums: Vec<_> = nums.into_iter().enumerate().collect();
        nums.sort_by_key(|t| t.1);
        nums.push((nums.len() + 1, i32::MAX));

        println!(" -> {:?}", nums);

        let mut rst = vec![0; nums.len()];
        let mut groups = vec![nums[0].0];
        let mut p = 0;

        (1..nums.len()).for_each(|i| {
            if nums[i].1 > nums[i - 1].1 + limit {
                groups.sort();
                groups.reverse();

                while let Some(g) = groups.pop() {
                    rst[g] = nums[p].1;
                    p += 1;
                }

                println!(" -> {:?}", rst);
            }

            groups.push(nums[i].0);
        });

        rst.pop();
        rst
    }
}

#[cfg(test)]
mod tests;
