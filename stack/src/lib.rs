//! Stack & Monotonic Stack

/// 1475 Final Prices With a Special Discount in a Shop
struct Solution1475;

impl Solution1475 {
    pub fn final_prices(prices: Vec<i32>) -> Vec<i32> {
        let mut stack = Vec::new();

        let mut rst = prices.clone();
        for i in 0..prices.len() {
            while stack.len() > 0 && prices[stack[stack.len() - 1]] >= prices[i] {
                if let Some(j) = stack.pop() {
                    rst[j] -= prices[i];
                }
            }
            stack.push(i);
        }

        rst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution1475() {
        assert_eq!(
            Solution1475::final_prices(vec![8, 4, 6, 2, 3]),
            vec![4, 2, 4, 2, 3]
        );
        assert_eq!(
            Solution1475::final_prices(vec![1, 2, 3, 4, 5]),
            vec![1, 2, 3, 4, 5]
        );
        assert_eq!(
            Solution1475::final_prices(vec![10, 1, 1, 6]),
            vec![9, 0, 1, 6]
        );
    }
}
