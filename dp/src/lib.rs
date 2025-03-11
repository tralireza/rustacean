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

#[cfg(test)]
mod tests;
