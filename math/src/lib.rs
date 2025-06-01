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

/// 335h Self Crossing
struct Sol335 {}

impl Sol335 {
    pub fn is_self_crossing(distance: Vec<i32>) -> bool {
        if distance.windows(4).any(|w| w[0] >= w[2] && w[3] >= w[1]) {
            return true;
        }

        if distance
            .windows(5)
            .any(|w| w[1] == w[3] && w[0] + w[4] >= w[2])
        {
            return true;
        }

        if distance
            .windows(6)
            .any(|w| w[3] >= w[1] && w[2] > w[4] && w[0] + w[4] >= w[2] && w[1] + w[5] >= w[3])
        {
            return true;
        }

        false
    }
}

/// 838m Push Dominoes
struct Sol838;

impl Sol838 {
    pub fn push_dominoes(dominoes: String) -> String {
        let n = dominoes.len();
        let mut force = vec![0; n];

        let mut r = 0;
        for (i, chr) in dominoes.chars().enumerate() {
            r = match chr {
                'R' => n as i32,
                'L' => 0,
                _ => (r - 1).max(0),
            };

            force[i] += r;
        }

        let mut l = 0;
        for (i, chr) in dominoes.chars().rev().enumerate() {
            l = match chr {
                'R' => 0,
                'L' => n as i32,
                _ => (l - 1).max(0),
            };

            force[n - 1 - i] -= l;
        }

        println!("-> {:?}", force);

        force
            .iter()
            .map(|&f| match f {
                ..=-1 => 'L',
                1.. => 'R',
                _ => '.',
            })
            .collect()
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

/// 970m Powerful Integers
struct Sol970;

impl Sol970 {
    /// 1 <= x,y <= 100
    /// 0 <= bound <= 10^6
    pub fn powerful_integers(x: i32, y: i32, bound: i32) -> Vec<i32> {
        use std::collections::HashSet;

        let mut p_x = vec![1];
        let mut p_y = vec![1];

        if x > 1 {
            let mut p = x;
            while p < bound {
                p_x.push(p);
                p *= x;
            }
        }
        if y > 1 {
            let mut p = y;
            while p < bound {
                p_y.push(p);
                p *= y;
            }
        }

        let mut set = HashSet::new();
        for x in &p_x {
            for y in &p_y {
                if x + y <= bound {
                    set.insert(x + y);
                }
            }
        }

        set.drain().collect()
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

/// 1295 Find Numbers with Even Number of Digits
struct Sol1295;

impl Sol1295 {
    pub fn find_numbers(nums: Vec<i32>) -> i32 {
        println!(
            ":: {}",
            nums.iter()
                .filter(|&&n| {
                    let mut digits = 0;
                    let mut n = n;
                    while n > 0 {
                        n /= 10;
                        digits += 1;
                    }

                    digits & 1 == 0
                })
                .count()
        );

        nums.iter()
            .map(|&n| {
                let mut digits = 0;
                let mut n = n;
                while n > 0 {
                    n /= 10;
                    digits += 1;
                }

                (digits + 1) & 1
            })
            .sum()
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

/// 2338h Count the Number of Ideal Arrays
struct Sol2338;

impl Sol2338 {
    /// 2 <= N <= 10^4, 1 <= Max <= 10^4
    /// Choose & Multi-Choose :: C((n/k)) = C(n+k-1/k)
    pub fn ideal_arrays(n: i32, max_value: i32) -> i32 {
        const P: usize = 15;

        let mut sieve: Vec<usize> = (0..=max_value as usize).collect();
        for p in 2..sieve.len() {
            if sieve[p] == p {
                for m in (p * p..sieve.len()).step_by(p) {
                    sieve[m] = p;
                }
            }
        }
        println!("-> Sieve :: {:?}", sieve);

        let mut factors = vec![vec![]; max_value as usize + 1];
        for n in 2..=max_value as usize {
            let mut x = n;
            while x > 1 {
                let factor = sieve[x];
                let mut count = 0;
                while x % factor == 0 {
                    x /= factor;
                    count += 1;
                }
                factors[n].push(count);
            }
        }
        println!("-> Factors Counts: {:?}", factors);

        let mut c = vec![vec![0; P + 1]; n as usize + P + 1];
        c[0][0] = 1;

        const M: i64 = 1000_000_007;
        for n in 1..c.len() {
            c[n][0] = 1;
            for k in 1..=n.min(P) {
                c[n][k] = (c[n - 1][k] + c[n - 1][k - 1]) % M
            }
        }

        (1..=max_value as usize).fold(0, |count, v| {
            let mut total = 1;
            for &k in &factors[v] {
                total = total * c[n as usize + k - 1][k] % M;
            }

            (count + total) % M
        }) as i32
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

        // 1 <= N_i <= 10^4
        println!(
            "-> {}",
            (low..=high).fold(0, |r, n| {
                match n {
                    1..100 if n % 11 == 0 => r + 1,
                    1000..10000 => {
                        if n / 1000 + (n % 1000) / 100 == (n % 100) / 10 + (n % 10) {
                            r + 1
                        } else {
                            r
                        }
                    }
                    _ => r,
                }
            })
        );

        count
    }
}

/// 2929m Distribute Candies Among Children II
struct Sol2929 {}

impl Sol2929 {
    /// 1 <= N, L <= 10^6
    pub fn distribute_candies(n: i32, limit: i32) -> i64 {
        let mut ways = 0;
        for candy1 in 0..=limit.min(n) {
            if n - candy1 <= 2 * limit {
                println!(
                    "-> Candy1: {candy1} | Candy2: {} ~ {}",
                    (n - candy1).min(limit),
                    0.max(n - candy1 - limit)
                );

                ways += ((n - candy1).min(limit) - (n - candy1 - limit).max(0) + 1) as i64
            }
        }

        ways
    }
}

/// 3024 Type of Triangle
struct Sol3024;

impl Sol3024 {
    pub fn triangle_type(nums: Vec<i32>) -> String {
        let mut nums = nums;
        nums.sort();

        let (a, b, c) = (nums[0], nums[1], nums[2]);
        if a + b <= c {
            "none"
        } else if a == c {
            "equilateral"
        } else if a == b || b == c {
            "isosceles"
        } else {
            "scalene"
        }
        .to_string()
    }
}

/// 3272h Find the Count of Good Integers
struct Sol3272;

impl Sol3272 {
    /// 1 <= n <= 10, 1 <= k <= 9
    pub fn count_good_integers(n: i32, k: i32) -> i64 {
        use std::collections::HashSet;

        let mut set: HashSet<String> = HashSet::new();

        let start = 10u32.pow((n - 1) as u32 / 2);
        for v in start..10 * start {
            let left = v.to_string();
            let right: String = left.chars().rev().skip((n & 1) as usize).collect();

            let palindrome = format!("{}{}", left, right);
            match palindrome.parse::<u64>() {
                Ok(v) if v % k as u64 == 0 => {
                    let mut chrs: Vec<char> = palindrome.chars().collect();
                    chrs.sort_unstable();
                    set.insert(chrs.iter().collect());
                }
                _ => {}
            }
        }

        println!("-> {:?}", set);

        let mut facts = vec![1; (n + 1) as usize];
        for n in 2..=n as usize {
            facts[n] = facts[n - 1] * n;
        }

        let mut count = 0;
        for chrs in set {
            let mut counter = [0; 10];
            for chr in chrs.as_bytes() {
                counter[(chr - b'0') as usize] += 1;
            }

            let mut perms = (n as usize - counter[0]) * facts[n as usize - 1];
            for &count in counter.iter() {
                perms /= facts[count];
            }

            count += perms;
        }

        count as i64
    }
}

#[cfg(test)]
mod tests;
