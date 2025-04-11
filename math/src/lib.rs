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

/// 2523m Closest Prime Numbers in Range
struct Sol2523;

impl Sol2523 {
    pub fn closest_primes(left: i32, right: i32) -> Vec<i32> {
        let mut sieve = vec![0; 1 + right as usize];

        for (i, n) in sieve
            .iter_mut()
            .enumerate()
            .take(right as usize + 1)
            .skip(2)
        {
            *n = i;
        }

        for p in 2..=right as usize {
            if sieve[p] == p {
                for m in (p * p..=right as usize).step_by(p) {
                    sieve[m] = p;
                }
            }
        }

        println!("-> {:?}", sieve);

        let primes: Vec<_> = sieve
            .into_iter()
            .enumerate()
            .skip(1)
            .filter(|(i, p)| i == p && *p as i32 >= left)
            .map(|(_, p)| p as i32)
            .collect();

        println!("-> {:?}", primes);

        primes.windows(2).fold(vec![-1, -1], |rst, v| {
            if rst == vec![-1, -1] || v[1] - v[0] < rst[1] - rst[0] {
                vec![v[0], v[1]]
            } else {
                rst
            }
        })
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

/// 2818h Apply Operations to Maximize Score
struct Sol2818;

impl Sol2818 {
    /// 1 <= N, N_i <= 10^5
    pub fn maximum_score(nums: Vec<i32>, k: i32) -> i32 {
        let mut omega = vec![0; 1 + 1e5 as usize];
        for p in 2..omega.len() {
            if omega[p] == 0 {
                for m in (p..omega.len()).step_by(p) {
                    omega[m] += 1;
                }
            }
        }

        const M: i64 = 7 + 1e9 as i64;
        fn mpower(mut b: i64, mut e: i64) -> i64 {
            let mut r = 1;
            while e > 0 {
                if e & 1 == 1 {
                    r = (r * b) % M;
                }
                b = (b * b) % M;
                e >>= 1;
            }

            r
        }

        let n = nums.len();
        let (mut left, mut right) = (vec![-1; n], vec![n as i32; n]);

        let mut stack: Vec<usize> = vec![];

        for (i, &v) in nums.iter().enumerate() {
            while !stack.is_empty()
                && omega[nums[stack[stack.len() - 1]] as usize] < omega[v as usize]
            {
                right[stack.pop().unwrap()] = i as i32;
            }

            if !stack.is_empty() {
                left[i] = *stack.last().unwrap() as i32;
            }

            stack.push(i);
        }

        let mut set = vec![];
        for e in nums
            .iter()
            .map(|&v| v as i64)
            .zip((0..n as i32).zip(left.iter().zip(right.iter())))
        {
            set.push(e);
        }
        set.sort_by_key(|e| -e.0);

        println!("-> {:?}", set);

        let mut score = 1i64;
        let mut k = k;
        for (v, (i, (l, r))) in set {
            let total = (i - l) as i64 * (r - i) as i64;
            if total >= k as i64 {
                score = score * mpower(v, k as i64) % M;
                break;
            }
            score = score * mpower(v, total) % M;
            k -= total as i32;
        }

        score as i32
    }
}

/// 2843 Count Symmetric Integers
struct Sol2843;

impl Sol2843 {
    pub fn count_symmetric_integers(low: i32, high: i32) -> i32 {
        let mut count = 0;

        for n in low..=high {
            let n = n.to_string();
            let n = n.as_bytes();
            let w = n.len();
            if w & 1 == 0 {
                let (mut left, mut right) = (0, 0);
                for i in 0..w / 2 {
                    left += n[i];
                    right += n[w / 2 + i];
                }
                if left == right {
                    count += 1;
                }
            }
        }

        count
    }
}

#[cfg(test)]
mod tests;
