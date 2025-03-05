//! # Math (Core) :: Rusty

/// 29m Divide Two Integers
struct Sol29;

impl Sol29 {
    /// Divide two integers
    ///
    /// Examples
    ///
    /// ```
    /// use rustacean::math::Sol29;
    ///
    /// assert_eq!(Sol29::divide(10, 3), 3);
    /// ```
    ///
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

/// 1780m Check if Number is a Sum of Powers of Three
struct Sol1780;

impl Sol1780 {
    /// 1 <= N <= 10^7
    pub fn check_powers_of_three(n: i32) -> bool {
        use std::iter::successors;

        let mut powers: Vec<_> = successors(Some(1), |p| {
            if 3 * p <= 1e7 as i32 {
                Some(3 * p)
            } else {
                None
            }
        })
        .collect();
        powers.reverse();

        println!("-> {:?}", powers);

        let mut n = n;
        for p in powers {
            if n >= p {
                n -= p;
                if n == 0 {
                    return true;
                }
            }
        }

        false
    }

    /// O(2^log3(N))
    fn check_powers_of_three_recursive(n: i32) -> bool {
        fn search(n: i32, power: i32) -> bool {
            if n == 0 {
                return true;
            }
            if n < power {
                return false;
            }

            search(n, 3 * power) || search(n - power, 3 * power)
        }

        search(n, 1)
    }
}

/// 2579m Count Total Number of Colored Cells
struct Sol2578;

impl Sol2578 {
    pub fn colored_cells(n: i32) -> i64 {
        let mut rst = 1;
        for n in 2..=n as i64 {
            rst += 4 * (n - 1);
        }

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

    #[test]
    fn test_1780() {
        for f in [
            Sol1780::check_powers_of_three,
            Sol1780::check_powers_of_three_recursive,
        ] {
            assert_eq!(f(12), true);
            assert_eq!(f(91), true);
            assert_eq!(f(21), false);
        }
    }

    #[test]
    fn test_2578() {
        for (n, r) in [(1, 1), (2, 5)] {
            assert_eq!(Sol2578::colored_cells(n), r);
        }
    }
}
