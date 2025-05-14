//! # Dynamic Programming

/// 115h Distinct Subsequences
struct Sol115;

impl Sol115 {
    pub fn num_distinct(s: String, t: String) -> i32 {
        let mut rst = 0;
        let (s, t) = (s.as_bytes(), t.as_bytes());
        let mut mem = vec![vec![-1; t.len()]; s.len()];

        fn count(s: &[u8], i: usize, j: usize, t: &[u8], mem: &mut Vec<Vec<i32>>) -> i32 {
            if j + 1 >= t.len() {
                return 1;
            }
            if i >= s.len() {
                return 0;
            }

            if mem[i][j] != -1 {
                return mem[i][j];
            }

            let mut rst = 0;
            for p in i + 1..s.len() {
                if s[p] == t[j + 1] && s.len() + j + 1 >= t.len() + p {
                    rst += count(s, p, j + 1, t, mem);
                }
            }
            mem[i][j] = rst;
            rst
        }

        for p in 0..s.len() {
            if s[p] == t[0] && s.len() >= t.len() + p {
                rst += count(s, p, 0, t, &mut mem);
            }
        }

        println!("-> {:?}", mem);

        rst
    }
}

/// 132h Palindrome Partitioning II
struct Sol132;

impl Sol132 {
    pub fn min_cut(s: String) -> i32 {
        let s = s.as_bytes();

        let mut palindrome = vec![vec![true; s.len()]; s.len()];
        for l in (0..s.len()).rev() {
            for r in l + 1..s.len() {
                palindrome[l][r] = s[l] == s[r] && palindrome[l + 1][r - 1];
            }
        }

        println!("-> {:?}", palindrome);

        let mut dp = vec![i32::MAX; s.len()];
        for r in 0..s.len() {
            if palindrome[0][r] {
                dp[r] = 0;
            } else {
                for l in 0..r {
                    if palindrome[l + 1][r] {
                        dp[r] = dp[r].min(dp[l] + 1);
                    }
                }
            }
        }

        dp[s.len() - 1]
    }
}

/// 174h Dungeon Game
struct Sol174;

impl Sol174 {
    pub fn calculate_minimum_hp(dungeon: Vec<Vec<i32>>) -> i32 {
        let (rows, cols) = (dungeon.len(), dungeon[0].len());
        let mut health = vec![vec![i32::MAX; cols + 1]; rows + 1];

        health[rows][cols - 1] = 1;
        health[rows - 1][cols] = 1;

        for r in (0..rows).rev() {
            for c in (0..cols).rev() {
                let x = health[r + 1][c].min(health[r][c + 1]) - dungeon[r][c];
                health[r][c] = if x <= 0 { 1 } else { x };
            }
        }

        println!("-> {:?}", health);

        health[0][0]
    }
}

/// 312h Burst Balloons
struct Sol312;

impl Sol312 {
    pub fn max_coins(nums: Vec<i32>) -> i32 {
        let mut mem = vec![vec![0; nums.len()]; nums.len()];

        fn search(mem: &mut Vec<Vec<i32>>, nums: &Vec<i32>, i: usize, j: usize) -> i32 {
            if i > j {
                return 0;
            }

            if i == j {
                let mut coins = nums[i];
                if i > 0 {
                    coins *= nums[i - 1];
                }
                if j < nums.len() - 1 {
                    coins *= nums[j + 1];
                }
                return coins;
            }

            if mem[i][j] > 0 {
                return mem[i][j];
            }

            let mut xcoins = 0;
            for k in i..=j {
                let mut coins = nums[k];
                if i > 0 {
                    coins *= nums[i - 1];
                }
                if j < nums.len() - 1 {
                    coins *= nums[j + 1];
                }

                if k > 0 {
                    coins += search(mem, nums, i, k - 1);
                }
                coins += search(mem, nums, k + 1, j);

                xcoins = xcoins.max(coins);
            }
            mem[i][j] = xcoins;

            xcoins
        }

        search(&mut mem, &nums, 0, nums.len() - 1)
    }
}

/// 368m Largest Divisible Subset
struct Sol368;

impl Sol368 {
    pub fn largest_divisible_subset(nums: Vec<i32>) -> Vec<i32> {
        let mut dp = vec![1; nums.len()];

        let mut nums = nums;
        nums.sort_unstable();

        println!("-> {:?}", nums);

        let mut longest = (1, 0);
        for i in 0..nums.len() {
            for j in 0..i {
                if nums[i] % nums[j] == 0 {
                    dp[i] = dp[i].max(dp[j] + 1);

                    if dp[i] > longest.0 {
                        longest = (dp[i], i);
                    }
                }
            }
        }

        println!("-> {:?}", (longest, &dp));

        let mut rst = vec![];

        let mut n = nums[longest.1];
        for i in (0..=longest.1).rev() {
            if dp[i] == longest.0 && n % nums[i] == 0 {
                rst.push(nums[i]);
                n = nums[i];
                longest.0 -= 1;
            }
        }
        rst.reverse();

        rst
    }
}

/// 377m Combination Sum IV
struct Sol377;

impl Sol377 {
    pub fn combination_sum4(nums: Vec<i32>, target: i32) -> i32 {
        let mut sums = vec![];
        sums.push(1);

        for t in 1..=target {
            sums.push(
                nums.iter()
                    .filter(|&n| t - n >= 0)
                    .map(|n| sums[(t - n) as usize])
                    .sum(),
            );
        }

        println!("-> {:?}", sums);
        sums[target as usize]
    }
}

/// 516m Longest Palindromic Subsequence
struct Sol516;

impl Sol516 {
    pub fn longest_palindrome_subseq(s: String) -> i32 {
        let s = s.as_bytes();

        let mut lps = vec![vec![0; s.len()]; s.len()];
        for x in 0..s.len() {
            lps[x][x] = 1;
            for y in (0..x).rev() {
                lps[x][y] = match s[x] == s[y] {
                    true => lps[x - 1][y + 1] + 2,
                    _ => lps[x - 1][y].max(lps[x][y + 1]),
                };
            }
        }

        println!("-> {:?}", lps);

        // Longest Common Subsequence
        let mut lcs = vec![vec![0; s.len() + 1]; s.len() + 1];
        for x in 0..s.len() {
            for y in 0..s.len() {
                lcs[x + 1][y + 1] = match s[x] == s[s.len() - 1 - y] {
                    true => lcs[x][y] + 1,
                    _ => lcs[x][y + 1].max(lcs[x + 1][y]),
                }
            }
        }
        println!("-> {} {:?}", lcs[s.len()][s.len()], lcs);

        lps[s.len() - 1][0]
    }
}

/// 790m Domino and Tromino Tiling
struct Sol790;

impl Sol790 {
    // 1 <= n <= 10000
    pub fn num_tilings(n: i32) -> i32 {
        fn two_states(n: i32) -> i32 {
            if n == 1 || n == 2 {
                return n;
            }

            let mut full = [0; 1000 + 1]; // full cover
            let mut lshape = [0; 1000 + 1]; // L shape cover

            (full[1], full[2], lshape[2]) = (1, 2, 1);

            const M: i64 = 1e9 as i64 + 7;
            for i in 3..=n as usize {
                full[i] = (full[i - 1] + full[i - 2] + 2 * lshape[i - 1]) % M;
                lshape[i] = (lshape[i - 1] + full[i - 2]) % M;
            }

            full[n as usize] as _
        }
        println!(":: {}", two_states(n));

        if n == 1 || n == 2 {
            return n;
        }

        const M: i64 = 1e9 as i64 + 7;
        let mut dp = vec![0; n as usize + 1];

        (dp[1], dp[2], dp[3]) = (1, 2, 5);
        for i in 4..=n as usize {
            dp[i] = (2 * dp[i - 1] + dp[i - 3]) % M;
        }

        println!("-> {:?}", dp);

        dp[n as usize] as _
    }
}

/// 873m Length of Longest Fibonacci Subsequence
struct Sol873;

impl Sol873 {
    pub fn len_longest_fib_subseq(arr: Vec<i32>) -> i32 {
        use std::collections::HashSet;

        let mut hset = HashSet::new();
        for n in &arr {
            hset.insert(n);
        }

        println!("-> {:?}", hset);

        let mut xlen = 0;
        for l in 0..arr.len() {
            for r in l + 1..arr.len() {
                let mut prv = arr[r];
                let mut cur = arr[l] + arr[r];

                let mut clen = 2;
                while hset.contains(&cur) {
                    (prv, cur) = (cur, prv + cur);

                    clen += 1;
                    xlen = xlen.max(clen);
                }
            }
        }

        xlen
    }
}

/// 1092h Shortest Common Supersequence
struct Sol1092;

impl Sol1092 {
    /// 1 <= Len_1, Len_2 <= 1000
    pub fn shortest_common_supersequence(str1: String, str2: String) -> String {
        let (str1, str2) = (str1.as_bytes(), str2.as_bytes());

        // Longest Common Subsequence
        let mut dp = vec![vec![0; str2.len() + 1]; str1.len() + 1];
        for (x, chr1) in str1.iter().enumerate() {
            for (y, chr2) in str2.iter().enumerate() {
                dp[x + 1][y + 1] = match chr1 == chr2 {
                    true => dp[x][y] + 1,
                    false => dp[x][y + 1].max(dp[x + 1][y]),
                };
            }
        }

        println!("-> LCS :: {:?}", dp);

        let mut rst = vec![];
        let (mut x, mut y) = (str1.len(), str2.len());
        while x > 0 || y > 0 {
            if x > 0 && y > 0 && str1[x - 1] == str2[y - 1] {
                rst.push(str1[x - 1]);
                x -= 1;
                y -= 1;
            } else if x > 0 && (y == 0 || dp[x - 1][y] >= dp[x][y - 1]) {
                rst.push(str1[x - 1]);
                x -= 1;
            } else if y > 0 {
                rst.push(str2[y - 1]);
                y -= 1;
            }
        }
        rst.reverse();

        String::from_utf8(rst).expect("")
    }

    /// Recursive Memoization (!MLE)
    pub fn scs_recursive(str1: String, str2: String) -> String {
        use std::collections::HashMap;

        let mut mem = HashMap::new();
        fn search<'l>(
            str1: &'l str,
            str2: &'l str,
            mem: &mut HashMap<(&'l str, &'l str), String>,
        ) -> String {
            match (str1.len(), str2.len()) {
                (0, 0) => "".to_string(),
                (0, _) => str2.to_string(),
                (_, 0) => str1.to_string(),
                _ => {
                    println!("-> {:?}", (&str1, &str2));

                    if let Some(rst) = mem.get(&(str1, str2)) {
                        return rst.to_string();
                    }

                    match str1[0..1].cmp(&str2[0..1]) {
                        std::cmp::Ordering::Equal => {
                            let scs = search(&str1[1..], &str2[1..], mem);
                            mem.insert((str1, str2), str1[0..1].to_string() + &*scs);
                        }
                        _ => {
                            let scs1 = search(&str1[1..], str2, mem);
                            let scs2 = search(str1, &str2[1..], mem);

                            if scs1.len() <= scs2.len() {
                                mem.insert((str1, str2), str1[0..1].to_string() + &*scs1);
                            } else {
                                mem.insert((str1, str2), str2[0..1].to_string() + &*scs2);
                            }
                        }
                    }

                    mem[&(str1, str2)].to_string()
                }
            }
        }

        let rst = search(&str1, &str2, &mut mem);
        println!("-> {:?}", mem);

        rst
    }
}

/// 1524m Number of Sub-arrays With Odd Sum
struct Sol1524;

impl Sol1524 {
    /// 1 <= N <= 10^5, 1 <= N_i <= 100
    pub fn num_of_subarrays(arr: Vec<i32>) -> i32 {
        const M: u32 = 1e9 as u32 + 7;

        let mut edp = vec![0; arr.len()]; // evens
        let mut odp = vec![0; arr.len()]; // odds

        match arr[arr.len() - 1] & 1 {
            1 => odp[arr.len() - 1] = 1,
            _ => edp[arr.len() - 1] = 1,
        }

        for i in (0..arr.len() - 1).rev() {
            match arr[i] & 1 {
                1 => {
                    odp[i] = (1 + edp[i + 1]) % M;
                    edp[i] = odp[i + 1];
                }
                _ => {
                    edp[i] = (1 + edp[i + 1]) % M;
                    odp[i] = odp[i + 1];
                }
            }
        }

        println!("-> {:?}", odp);

        (odp.into_iter().sum::<u32>() % M) as i32
    }

    fn num_of_subarrays_psum(arr: Vec<i32>) -> i32 {
        const M: i32 = 1e9 as i32 + 7;

        let mut count = 0;
        let (mut evens, mut odds) = (1, 0);

        let mut psum = 0;
        for n in arr {
            psum += n;

            count += match psum & 1 {
                1 => {
                    odds += 1;
                    evens
                }
                _ => {
                    evens += 1;
                    odds
                }
            } % M;
        }

        count % M
    }
}

/// 1749m Maximum Absolute Sum of Any Subarray
struct Sol1749;

impl Sol1749 {
    pub fn max_absolute_sum(nums: Vec<i32>) -> i32 {
        let mut rst = 0;
        let (mut xsum, mut nsum) = (i32::MIN, i32::MAX);

        let mut pfx = 0;
        for n in nums {
            pfx += n;

            xsum = xsum.max(pfx);
            nsum = nsum.min(pfx);

            use std::cmp::Ordering::*;
            match pfx.cmp(&0) {
                Greater => rst = rst.max(pfx.max(pfx - nsum)),
                Less => rst = rst.max((-pfx).max(xsum - pfx)),
                _ => (),
            }
        }

        rst
    }
}

/// 2140m Solving Questions With Brainpower
struct Sol2140;

impl Sol2140 {
    pub fn most_points(questions: Vec<Vec<i32>>) -> i64 {
        use std::cmp::Ordering::*;

        let mut dp = vec![[0i64; 2]; questions.len() + 1];

        for (i, q) in questions.iter().enumerate().rev() {
            println!("-> {:?}", (i, &q));

            dp[i][0] = dp[i + 1][0].max(dp[i + 1][1]);

            let [pts, skip] = q[..2] else { unreachable!() };
            let next = i + 1 + skip as usize;

            dp[i][1] = pts as i64
                + match next.cmp(&questions.len()) {
                    Less => dp[next][0].max(dp[next][1]),
                    _ => 0,
                };

            println!("-> {:?}", dp);
        }

        println!(":: {:?}", dp[0]);

        dp[0][0].max(dp[0][1])
    }

    /// Recursion with Memo
    fn recursive(questions: Vec<Vec<i32>>) -> i64 {
        let mut mem = vec![0; questions.len()];

        fn search(i: usize, questions: &[Vec<i32>], mem: &mut [i64]) -> i64 {
            if i >= questions.len() {
                return 0;
            }

            if mem[i] > 0 {
                return mem[i];
            }

            let [pts, skip] = questions[i][..] else {
                unreachable!()
            };
            mem[i] = search(i + 1, questions, mem)
                .max(pts as i64 + search(i + 1 + skip as usize, questions, mem));

            mem[i]
        }

        let pts = search(0, &questions, &mut mem);
        println!("-> Mem: {:?}", mem);

        pts
    }
}

/// 2836h Maximize Value of Function in a Ball Passing Game
struct Sol2836;

impl Sol2836 {
    pub fn get_max_function_value(receiver: Vec<i32>, k: i64) -> i64 {
        // Binary Lifting
        // far(p, i) :: 2^p ancestor of i
        // far(p, i) = far(p-1, far(p-1, i))

        println!(" || {:?}", receiver);

        let (bits, nodes) = (k.ilog2() as usize + 1, receiver.len());

        let mut far = vec![vec![0; bits]; nodes];
        (0..bits).for_each(|p| {
            (0..nodes).for_each(|i| match p {
                0 => far[i][0] = receiver[i] as usize,
                _ => far[i][p] = far[far[i][p - 1]][p - 1],
            })
        });

        println!(" -> {:?}", far);

        let mut score = vec![vec![0i64; bits]; nodes];
        (0..bits).for_each(|p| {
            (0..nodes).for_each(|i| match p {
                0 => score[i][0] = receiver[i] as i64,
                _ => score[i][p] = score[i][p - 1] + score[far[i][p - 1]][p - 1],
            })
        });

        println!(" -> {:?}", score);

        (0..nodes).fold(0, |xscore, istart| {
            xscore.max({
                let (mut iscore, mut i) = (0, istart);
                (0..bits).rev().for_each(|p| {
                    if (1 << p) & k != 0 {
                        iscore += score[i][p];
                        i = far[i][p];
                    }
                });
                iscore + istart as i64
            })
        })
    }
}

/// 2999h Count the Number of Powerful Integers
struct Sol2999;

impl Sol2999 {
    pub fn number_of_powerful_int(start: i64, finish: i64, limit: i32, s: String) -> i64 {
        let finish = finish.to_string();
        let start = format!("{:0>width$}", start.to_string(), width = finish.len());

        println!("-> {:?}", (&start, &finish, limit, &s));

        let mut mem = vec![-1; finish.len()];

        fn search(
            i: usize,
            slimit: bool,
            start: &[u8],
            flimit: bool,
            finish: &[u8],
            limit: u8,
            s: &[u8],
            mem: &mut [i64],
        ) -> i64 {
            println!("-> {:?}", (i, slimit, flimit));

            if i == (start.len() & finish.len()) {
                return 1;
            }
            if !slimit && !flimit && mem[i] != -1 {
                return mem[i];
            }

            let mut count = 0;
            let sdigit = if slimit { start[i] } else { b'0' };
            let fdigit = if flimit { finish[i] } else { b'9' };

            if i < finish.len() - s.len() {
                for digit in sdigit..=fdigit.min(limit) {
                    count += search(
                        i + 1,
                        slimit && digit == start[i],
                        start,
                        flimit && digit == finish[i],
                        finish,
                        limit,
                        s,
                        mem,
                    );
                }
            } else {
                let digit = s[i - (finish.len() - s.len())];
                if sdigit <= digit && digit <= fdigit.min(limit) {
                    count += search(
                        i + 1,
                        slimit && digit == start[i],
                        start,
                        flimit && digit == finish[i],
                        finish,
                        limit,
                        s,
                        mem,
                    );
                }
            }

            if !slimit && !flimit {
                mem[i] = count;
            }
            count
        }

        search(
            0,
            true,
            start.as_bytes(),
            true,
            finish.as_bytes(),
            b'0' + limit as u8,
            s.as_bytes(),
            &mut mem,
        )
    }
}

/// 3335m Total Characters in String After Transformations I
struct Sol3335;

impl Sol3335 {
    /// 1 <= N, t <= 10^5
    pub fn length_after_transformations(s: String, t: i32) -> i32 {
        const M: i32 = 1e9 as i32 + 7;

        let mut freqs = vec![vec![0; 26]; t as usize + 1];
        for chr in s.as_bytes() {
            freqs[0][(chr - b'a') as usize] += 1;
        }

        for i in 1..=t as usize {
            freqs[i][0] = freqs[i - 1][25];
            freqs[i][1] = (freqs[i - 1][0] + freqs[i - 1][25]) % M;
            for chr in 2..26 {
                freqs[i][chr] = freqs[i - 1][chr - 1]
            }
        }

        println!("-> {:?}", freqs);

        freqs[t as usize].iter().fold(0, |t, &l| (t + l) % M)
    }
}

/// 3337m Total Characters in String After Transformations II
struct Sol3337;

impl Sol3337 {
    /// 1 <= N <= 10^5, 1 <= t <= 10^9
    pub fn length_after_transformations(s: String, t: i32, nums: Vec<i32>) -> i32 {
        // M a b c d e f g ... y z
        // a   1 1                 / N_a: 2
        // b     1 1 1             / N_b: 3
        // c       1               / N_c: 1
        // ...
        // y 1                   1 / N_y: 2
        // z 1 1 1 1               / N_z: 4

        type M = [[i64; 26]; 26];

        let mut m: M = [[0; 26]; 26];
        for chr in 0..26 {
            for t in 1..=nums[chr] as usize {
                m[chr][(chr + t) % 26] = 1;
            }
        }

        let mut freq = [0; 26];
        for chr in s.as_bytes() {
            freq[(chr - b'a') as usize] += 1;
        }

        const MOD: i64 = 1000_000_007;

        fn mcopy(t: &mut M, f: &M) {
            for i in 0..26 {
                for j in 0..26 {
                    t[i][j] = f[i][j];
                }
            }
        }

        fn mzero(m: &mut M) {
            for row in m.iter_mut() {
                for v in row.iter_mut() {
                    *v = 0;
                }
            }
        }

        fn matrix_multiply(m: &mut M, a: &M, b: &M) {
            mzero(m);

            for i in 0..26 {
                for (k, b_k) in b.iter().enumerate() {
                    let a_ik = a[i][k];
                    if a_ik != 0 {
                        for (j, b_k_j) in b_k.iter().enumerate() {
                            m[i][j] = (m[i][j] + a_ik * b_k_j) % MOD;
                        }
                    }
                }
            }
        }

        /// <- b^e
        fn square_exponentiation(b: &M, mut e: i32) -> M {
            let mut power: M = [[0; 26]; 26];
            for (d, row) in power.iter_mut().enumerate() {
                row[d] = 1;
            }

            let mut base: M = [[0; 26]; 26];
            mcopy(&mut base, b);

            let mut t: M = [[0; 26]; 26];
            while e > 0 {
                if e & 1 == 1 {
                    matrix_multiply(&mut t, &power, &base);
                    mcopy(&mut power, &t);
                }

                matrix_multiply(&mut t, &base, &base);
                mcopy(&mut base, &t);

                e >>= 1;
            }

            power
        }

        let power: M = square_exponentiation(&m, t);

        println!("-> M^t {power:?}");

        let mut tfreq = [0; 26];
        for i in 0..26 {
            for j in 0..26 {
                tfreq[i] = (tfreq[i] + freq[i] * power[i][j]) % MOD;
            }
        }

        println!("-> {tfreq:?}");

        tfreq.iter().fold(0, |l, &n| (l + n) % MOD) as _
    }
}

#[cfg(test)]
mod tests;
