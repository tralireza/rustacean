//! # Math (Core) :: Rusty

/// 29m Divide Two Integers
struct Sol29;

impl Sol29 {
    pub fn divide(dividend: i32, divisor: i32) -> i32 {
        if dividend == divisor {
            return 1;
        }

        let psign = (dividend < 0) == (divisor < 0);
        let (mut n, d) = (dividend.unsigned_abs(), divisor.unsigned_abs());

        let mut r: u32 = 0;
        while n >= d {
            let mut p = 0;
            while n > (d << (p + 1)) {
                p += 1;
            }

            r += 1 << p;
            n -= d << p;
        }

        if r == 1 << 31 && psign {
            return i32::MAX; // (1<<31) -1
        }

        match psign {
            true => r as i32,
            _ => -(r as i32),
        }
    }
}

/// 908 Smallest Range I
struct Sol908;

impl Sol908 {
    pub fn smallest_range_i(nums: Vec<i32>, k: i32) -> i32 {
        let mx = nums.iter().max();
        let mn = nums.iter().min();

        println!(" -> {:?} {:?}", mx, mn);

        match (nums.iter().max(), nums.iter().min()) {
            (Some(x), Some(n)) => 0.max(x - n - 2 * k),
            _ => 0,
        }
    }
}

/// 989 Add to Array-Form of Integer
struct Sol989;

impl Sol989 {
    pub fn add_to_array_form(num: Vec<i32>, k: i32) -> Vec<i32> {
        println!(
            " -> {:?}",
            std::iter::successors(Some((k, 0, num.len())), |(mut carry, _, mut p)| {
                match carry > 0 || p > 0 {
                    true => {
                        if p > 0 {
                            p -= 1;
                            carry += num[p];
                        }
                        Some((carry / 10, carry % 10, p))
                    }
                    _ => None,
                }
            })
            .map(|(_, d, _)| d)
            .skip(1)
            .fold(vec![], |mut rst, d| {
                rst.push(d);
                rst
            })
        );

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
    fn test_29() {
        for (n, d, r) in [(10, 3, 3), (7, -3, -2), (-2147483648, -1, 2147483647)] {
            assert_eq!(Sol29::divide(n, d), r);
        }
    }

    #[test]
    fn test_908() {
        assert_eq!(Sol908::smallest_range_i(vec![1], 0), 0);
        assert_eq!(Sol908::smallest_range_i(vec![0, 10], 2), 6);
        assert_eq!(Sol908::smallest_range_i(vec![1, 3, 6], 3), 0);
    }

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
