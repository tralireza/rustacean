//! Priority Queue

/// 2182m Construct String With Repeat Limit
struct Solution2182;

impl Solution2182 {
    pub fn repeat_limited_string(s: String, repeat_limit: i32) -> String {
        use std::collections::BinaryHeap;

        let s = s.as_bytes();
        let mut freq = vec![0; 26];
        for l in s {
            freq[(l - b'a') as usize] += 1;
        }

        let mut pq = BinaryHeap::new();
        for (l, &f) in freq.iter().enumerate() {
            if f > 0 {
                pq.push((l as u8 + b'a', f));
            }
        }

        println!(" -> PQ :: {:?}", pq);

        let mut s = String::new();
        while let Some((l, mut f)) = pq.pop() {
            for _ in 0..f.min(repeat_limit) {
                s.push(l as char);
            }
            f -= repeat_limit;

            if f > 0 {
                if let Some((ll, mut lf)) = pq.pop() {
                    s.push(ll as char);
                    lf -= 1;
                    if lf > 0 {
                        pq.push((ll, lf));
                    }

                    pq.push((l, f));
                }
            }
        }

        s
    }
}

/// 2593m Find Score of an Array After Marking All Elements
struct Solution2593;

impl Solution2593 {
    pub fn find_score(nums: Vec<i32>) -> i64 {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Debug, PartialEq, Eq)]
        struct Item {
            n: i32,
            i: usize,
        }

        impl Ord for Item {
            fn cmp(&self, other: &Self) -> Ordering {
                if self.n == other.n {
                    return other.i.cmp(&self.i);
                }
                other.n.cmp(&self.n)
            }
        }

        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut pq = BinaryHeap::with_capacity(nums.len());
        nums.iter()
            .enumerate()
            .for_each(|(i, n)| pq.push(Item { n: *n, i: i }));

        println!(" -> pq :: {:?}", pq);

        let mut marks = vec![false; nums.len()];
        let mut score = 0i64;
        while let Some(Item { n, i }) = pq.pop() {
            if marks[i] {
                continue;
            }

            marks[i] = true;
            if 0 < i {
                marks[i - 1] = true;
            }
            if i < marks.len() - 1 {
                marks[i + 1] = true;
            }

            score += n as i64;
        }

        score
    }
}

/// 2762m Continuous Subarrays
struct Solution2762;

impl Solution2762 {
    pub fn continuous_subarrays(nums: Vec<i32>) -> i64 {
        use std::collections::BTreeMap;

        let mut frq = BTreeMap::new();
        let mut count = 0i64;

        let (mut left, mut right) = (0, 0);
        while right < nums.len() {
            frq.insert(nums[right], frq.get(&nums[right]).unwrap_or(&0) + 1);

            while *frq.last_entry().unwrap().key() - *frq.first_entry().unwrap().key() > 2 {
                frq.insert(nums[left], frq[&nums[left]] - 1);
                if frq[&nums[left]] == 0 {
                    frq.remove(&nums[left]);
                }

                left += 1;
            }

            right += 1;
            count += (right - left) as i64;
        }

        println!(" -> BTreeMap :: {:?}", frq);

        count
    }
}

/// 3264 Find Array State After K Multiplication Operations I
struct Solution3264;

impl Solution3264 {
    pub fn get_final_state(nums: Vec<i32>, k: i32, multiplier: i32) -> Vec<i32> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut pq = BinaryHeap::with_capacity(nums.len());

        for (i, &n) in nums.iter().enumerate() {
            pq.push(Reverse((n, i)));
        }

        for _ in 0..k {
            if let Some(Reverse((n, i))) = pq.pop() {
                pq.push(Reverse((multiplier * n, i)));
            }
        }

        println!(" -> {:?}", pq);

        let mut nums = nums;
        for Reverse((n, i)) in pq {
            nums[i] = n;
        }

        nums
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// 2182m Construct String With Repeat Limit
    fn test_solution2182() {
        assert_eq!(
            Solution2182::repeat_limited_string(String::from("cczazcc"), 3),
            "zzcccac".to_string()
        );
        assert_eq!(
            Solution2182::repeat_limited_string(String::from("aababab"), 2),
            "bbabaa".to_string()
        );
    }

    #[test]
    /// 2593m Find Score of an Array After Marking All Elements
    fn test_solution2593() {
        assert_eq!(Solution2593::find_score(vec![2, 1, 3, 4, 5, 2]), 7);
        assert_eq!(Solution2593::find_score(vec![2, 3, 5, 1, 3, 2]), 5);
        assert_eq!(Solution2593::find_score(vec![8, 6, 1, 9, 2, 2, 8]), 19);
    }

    #[test]
    /// 2762m Continuous Subarrays
    fn test_solution2762() {
        assert_eq!(Solution2762::continuous_subarrays(vec![5, 4, 2, 4]), 8);
        assert_eq!(Solution2762::continuous_subarrays(vec![1, 2, 3]), 6);
    }

    #[test]
    /// 3264 Final Array State After K Multiplication Operations I
    fn test_solution3264() {
        assert_eq!(
            Solution3264::get_final_state(vec![2, 1, 3, 5, 6], 5, 2),
            vec![8, 4, 6, 5, 6]
        );
        assert_eq!(Solution3264::get_final_state(vec![1, 2], 3, 4), vec![16, 8]);
    }
}
