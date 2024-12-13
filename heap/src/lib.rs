//! Priority Queue

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// 2593m Find Score of an Array After Marking All Elements
    fn test_solution2593() {
        assert_eq!(Solution2593::find_score(vec![2, 1, 3, 4, 5, 2]), 7);
        assert_eq!(Solution2593::find_score(vec![2, 3, 5, 1, 3, 2]), 5);
        assert_eq!(Solution2593::find_score(vec![8, 6, 1, 9, 2, 2, 8]), 19);
    }
}
