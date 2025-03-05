//! # Sorting

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
mod tests {
    use super::*;

    #[test]
    fn test_220() {
        for (rst, nums, idiff, vdiff) in [
            (true, vec![1, 2, 3, 1], 3, 0),
            (false, vec![1, 5, 9, 1, 5, 9], 2, 3),
            (false, vec![4, 2], 2, 1),
            (true, vec![4, 1, 6, 3], 4, 1),
        ] {
            assert_eq!(
                Sol220::contains_nearby_almost_duplicate(nums, idiff, vdiff),
                rst
            );
        }
    }

    #[test]
    fn test_905() {
        assert_eq!(
            Sol905::sort_array_by_parity(vec![3, 1, 2, 4]),
            vec![4, 2, 3, 1]
        );
        assert_eq!(Sol905::sort_array_by_parity(vec![0]), vec![0]);
    }

    #[test]
    fn test_2948() {
        assert_eq!(
            Sol2948::lexicographically_smallest_array(vec![1, 5, 3, 9, 8], 2),
            vec![1, 3, 5, 8, 9]
        );
        assert_eq!(
            Sol2948::lexicographically_smallest_array(vec![1, 7, 6, 18, 2, 1], 3),
            vec![1, 6, 7, 18, 1, 2]
        );
        assert_eq!(
            Sol2948::lexicographically_smallest_array(vec![1, 7, 28, 19, 10], 2),
            vec![1, 7, 28, 19, 10]
        );
    }
}
