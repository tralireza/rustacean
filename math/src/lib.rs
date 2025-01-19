//! # Math (Core) :: Rusty

/// 989 Add to Array-Form of Integer
struct Sol989;

impl Sol989 {
    pub fn add_to_array_form(num: Vec<i32>, k: i32) -> Vec<i32> {
        let mut rst = vec![];

        let (mut carry, mut p) = (k, num.len());
        while carry > 0 || p > 0 {
            if p > 0 {
                p -= 1;
                carry += num[p]
            }
            rst.push(carry % 10);
            carry /= 10;
        }

        println!(" -> {rst:?}");

        rst.reverse();
        rst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_989() {
        assert_eq!(
            Sol989::add_to_array_form(vec![1, 2, 0, 0], 34),
            vec![1, 2, 3, 4]
        );
        assert_eq!(Sol989::add_to_array_form(vec![2, 7, 4], 181), vec![4, 5, 5]);
        assert_eq!(
            Sol989::add_to_array_form(vec![2, 1, 5], 806),
            vec![1, 0, 2, 1]
        );
    }
}
