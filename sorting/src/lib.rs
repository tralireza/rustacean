//! # Sorting

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_905() {
        assert_eq!(
            Sol905::sort_array_by_parity(vec![3, 1, 2, 4]),
            vec![4, 2, 3, 1]
        );
        assert_eq!(Sol905::sort_array_by_parity(vec![0]), vec![0]);
    }
}
