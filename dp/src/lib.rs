//! # Dynamic Programming

#![feature(test)]

extern crate test;

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

/// 329h Longest Increasing Path in a Matrix
struct Sol329 {}

impl Sol329 {
    // 1 <= M, N <= 2000
    // 0 <= M_ij <= 2^31 - 1
    pub fn longest_increasing_path(matrix: Vec<Vec<i32>>) -> i32 {
        let mut dp = vec![vec![0; matrix[0].len()]; matrix.len()];

        fn dfs((r, c): (usize, usize), matrix: &[Vec<i32>], dp: &mut [Vec<i32>]) {
            println!("-> {:?}", (r, c));

            if dp[r][c] != 0 {
                return;
            }

            let v = matrix[r][c];

            let mut steps = 1;
            dp[r][c] = steps;

            let dirs = [-1, 0, 1, 0, -1];
            for d in 0..4 {
                let (r, c) = (
                    r.wrapping_add_signed(dirs[d]),
                    c.wrapping_add_signed(dirs[d + 1]),
                );

                if r < matrix.len() && c < matrix[r].len() {
                    println!("-> . {:?}", (r, c));

                    if matrix[r][c] > v {
                        if dp[r][c] == 0 {
                            dfs((r, c), matrix, dp);
                        }

                        steps = steps.max(dp[r][c] + 1);
                    }
                }
            }

            dp[r][c] = steps;
        }

        for (r, rows) in matrix.iter().enumerate() {
            for c in 0..rows.len() {
                if dp[r][c] == 0 {
                    dfs((r, c), &matrix, &mut dp);
                }
            }
        }

        println!("-> {dp:?}");

        if let Some(&max) = dp.iter().flatten().max() {
            return max;
        }
        1
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

/// 639h Decode Ways II
struct Sol639 {}

impl Sol639 {
    /// 1 <= N <= 10^5
    pub fn num_decodings(s: String) -> i32 {
        let s: Vec<_> = s.chars().collect();
        const M: i64 = 1e9 as i64 + 7;

        let mut dp = vec![0; s.len() + 1];
        dp[0] = 1;
        dp[1] = match s[0] {
            '*' => 9,
            '0' => 0,
            _ => 1,
        };

        for (i, &chr) in s.iter().enumerate().skip(1) {
            match chr {
                '*' => {
                    dp[i + 1] = 9 * dp[i] % M;
                    dp[i + 1] += match s[i - 1] {
                        '1' => 9 * dp[i - 1] % M,
                        '2' => 6 * dp[i - 1] % M,
                        '*' => 15 * dp[i - 1] % M,
                        _ => 0,
                    };
                }
                _ => {
                    dp[i + 1] = if chr != '0' { dp[i] } else { 0 };
                    dp[i + 1] += match s[i - 1] {
                        '1' => dp[i - 1],
                        '2' => {
                            if s[i] <= '6' {
                                dp[i - 1]
                            } else {
                                0
                            }
                        }
                        '*' => (if s[i] <= '6' { 2 } else { 1 }) * dp[i - 1] % M,
                        _ => 0,
                    };
                }
            }
        }

        println!("-> {dp:?}");
        (dp[s.len()] % M) as _
    }
}

/// 730h Count Different Palindromic Subsequences
struct Sol730;

impl Sol730 {
    /// 1 <= N <= 1000
    /// 'a'|'b'|'c'|'d'
    pub fn count_palindromic_subsequences(s: String) -> i32 {
        use std::collections::HashMap;

        let mut mem = HashMap::new();
        let chrs: Vec<_> = s.chars().collect();

        const M: i64 = 1e9 as i64 + 7;

        fn search(
            start: usize,
            end: usize,
            chrs: &[char],
            mem: &mut HashMap<[usize; 2], i64>,
        ) -> i64 {
            if start >= end {
                return 0;
            }

            if let Some(&count) = mem.get(&[start, end]) {
                return count;
            }

            let mut count = 0;
            for chr in 'a'..='d' {
                match (
                    chrs[start..end].iter().position(|&v| v == chr),
                    chrs[start..end].iter().rev().position(|&v| v == chr),
                ) {
                    (None, None) => continue,
                    (Some(l), Some(r)) => {
                        let r = end - start - r - 1;
                        println!("{start}   {l} {chr} {r}   {end}");
                        if l == r {
                            count += 1;
                        } else {
                            count += 2 + search(start + l + 1, start + r, chrs, mem);
                        }
                    }
                    (_, _) => {}
                }
                count %= M;
            }

            mem.insert([start, end], count);
            println!("-> {mem:?}");

            count
        }

        search(0, chrs.len(), &chrs, &mut mem) as _
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

/// 808m Soup Servings
struct Sol808 {}

impl Sol808 {
    pub fn soup_servings(n: i32) -> f64 {
        use std::collections::HashMap;

        let n = (n + 24) / 25;
        let mut memo = HashMap::new();

        fn search(a: i32, b: i32, memo: &mut HashMap<(i32, i32), f64>) -> f64 {
            if a <= 0 && b <= 0 {
                return 0.5;
            }
            if a <= 0 {
                return 1.0;
            }
            if b <= 0 {
                return 0.0;
            }

            if let Some(&p) = memo.get(&(a, b)) {
                return p;
            }

            let p = (search(a - 4, b, memo)
                + search(a - 3, b - 1, memo)
                + search(a - 2, b - 2, memo)
                + search(a - 1, b - 3, memo))
                / 4.0;
            memo.insert((a, b), p);

            p
        }

        for n in 1..=n {
            if search(n, n, &mut memo) > 1.0 - 0.00001 {
                return 1.0;
            }
        }

        search(n, n, &mut memo)
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

/// 1277m Count Square Submatrices with All Ones
struct Sol1277 {}

impl Sol1277 {
    pub fn count_squares(matrix: Vec<Vec<i32>>) -> i32 {
        use std::cmp::min;

        let (cols, rows) = (matrix[0].len(), matrix.len());

        let mut counts = vec![vec![0; cols + 1]; rows + 1];
        let mut count = 0;

        for r in 0..rows {
            for c in 0..cols {
                if matrix[r][c] == 1 {
                    counts[r + 1][c + 1] =
                        1 + min(counts[r][c], min(counts[r + 1][c], counts[r][c + 1]));
                    count += counts[r + 1][c + 1];
                }
            }
        }

        count
    }
}

/// 1504m Count Submatrices With All Ones
struct Sol1504 {}

impl Sol1504 {
    pub fn num_submat(mat: Vec<Vec<i32>>) -> i32 {
        let cols = mat[0].len();

        let mut count = 0;
        let mut counts = vec![0; cols * cols];

        for row in mat.iter() {
            for c in 0..cols {
                let mut flag = true;
                for k in 0..=c {
                    if flag && row[c - k] == 0 {
                        flag = false;
                    }

                    if flag {
                        counts[c * cols + k] += 1;
                        count += counts[c * cols + k];
                    } else {
                        counts[c * cols + k] = 0;
                    }
                }
            }
        }

        count
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

/// 1751h Maximum Number of Events That Can Be Attended
struct Sol1751 {}

impl Sol1751 {
    pub fn max_values(mut events: Vec<Vec<i32>>, k: i32) -> i32 {
        events.sort_by_key(|v| v[0]);
        println!("-> {events:?}");

        let rsearch = |target| {
            let (mut l, mut r) = (0, events.len());
            while l < r {
                let m = l + ((r - l) >> 1);
                if events[m][0] <= target {
                    l = m + 1;
                } else {
                    r = m;
                }
            }

            l
        };

        let mut dp = vec![vec![0; events.len() + 1]; 1 + k as usize];
        for start in (0..events.len()).rev() {
            let x = rsearch(events[start][1]);
            for k in 1..=k as usize {
                dp[k][start] = (dp[k][start + 1]).max(dp[k - 1][x] + events[start][2]);
            }
        }

        dp[k as usize][0]
    }
}

/// 1857h Largest Color Value in a Directed Graph
struct Sol1857 {}

impl Sol1857 {
    /// 1 <= V <= 10^5
    /// 0 <= E <= 10^5
    pub fn largest_path_value(colors: String, edges: Vec<Vec<i32>>) -> i32 {
        use std::collections::{HashMap, HashSet};

        let mut gadj: HashMap<usize, HashSet<usize>> = HashMap::new();
        for edge in edges.iter() {
            gadj.entry(edge[0] as usize)
                .or_default()
                .insert(edge[1] as usize);
        }
        println!("-> {gadj:?}");

        let colors = colors.as_bytes();
        let mut dp = vec![[0; 26]; colors.len()];

        #[derive(Clone, PartialEq, Debug)]
        enum OpColor {
            NotVisited, // White
            Visiting,   // Gray
            Visited,    // Black
        }

        fn search(
            v: usize,
            dp: &mut [[usize; 26]],
            colors: &[u8],
            gadj: &HashMap<usize, HashSet<usize>>,
            coloring: &mut [OpColor],
        ) -> bool {
            if coloring[v] != OpColor::NotVisited {
                return true;
            }

            coloring[v] = OpColor::Visiting;

            if let Some(uset) = gadj.get(&v) {
                for &u in uset.iter() {
                    match coloring[u] {
                        OpColor::Visited => {}
                        OpColor::Visiting => return true,
                        OpColor::NotVisited => {
                            if search(u, dp, colors, gadj, coloring) {
                                return true;
                            }
                        }
                    }

                    for color in 0..26 {
                        dp[v][color] = dp[v][color].max(dp[u][color]);
                    }
                }
            }

            dp[v][(colors[v] - b'a') as usize] += 1;

            coloring[v] = OpColor::Visited;
            false
        }

        let mut coloring = vec![OpColor::NotVisited; colors.len()];
        for src in 0..colors.len() {
            if coloring[src] == OpColor::NotVisited
                && search(src, &mut dp, colors, &gadj, &mut coloring)
            {
                return -1;
            }
        }

        println!("-> {dp:?}");

        match dp
            .iter()
            .map(|row| match row.iter().max() {
                Some(&xrow) => xrow,
                _ => 0,
            })
            .max()
        {
            Some(xgraph) => xgraph as i32,
            _ => 0,
        }
    }
}

/// 1931h Painting a Grid With Three Different Colors
struct Sol1931;

impl Sol1931 {
    /// 1 <= m <= 5, 1 <= n <= 1000
    pub fn color_the_grid(m: i32, n: i32) -> i32 {
        use std::collections::HashMap;

        let mut masks = HashMap::new();
        (0..3i32.pow(m as u32)).for_each(|mask| {
            let mut colors = vec![];

            let mut v = mask;
            for _ in 0..m {
                colors.push(v % 3);
                v /= 3;
            }

            if !colors.windows(2).any(|w| w[0] == w[1]) {
                masks.insert(mask, colors);
            }
        });

        println!("-> Color Masks: {masks:?}");

        let mut adjacents = HashMap::new();
        for (&mask1, color1) in &masks {
            for (&mask2, color2) in &masks {
                if !color1
                    .iter()
                    .zip(color2.iter())
                    .any(|(color1, color2)| color1 == color2)
                {
                    adjacents.entry(mask1).or_insert(vec![]).push(mask2);
                }
            }
        }

        println!("-> Rows Adjacent: {adjacents:?}");

        const M: i32 = 1_000_000_007;

        {
            let mut dp_cur = vec![0; [1, 3, 9, 27, 81, 243][m as usize]];
            for &mask in masks.keys() {
                dp_cur[mask as usize] = 1;
            }
            for _ in 1..n {
                let mut dp_next = vec![0; dp_cur.len()];
                for mask in 0..[1, 3, 9, 27, 81, 243][m as usize] {
                    if dp_cur[mask as usize] > 0
                        && let Some(adjacent) = adjacents.get(&mask)
                    {
                        dp_next[mask as usize] = adjacent
                            .iter()
                            .fold(0, |count, &mask| (count + dp_cur[mask as usize]) % M);
                    }
                }
                dp_cur = dp_next;
            }
            println!(":: {}", dp_cur.iter().fold(0, |count, &n| (count + n) % M));
        }

        let mut dp_cur = HashMap::new();
        for &mask in masks.keys() {
            dp_cur.insert(mask, 1);
        }
        for _ in 1..n {
            let mut dp_next = HashMap::new();
            for &mask in dp_cur.keys() {
                if let Some(adjacent) = adjacents.get(&mask) {
                    dp_next.insert(
                        mask,
                        adjacent.iter().fold(0, |count, &mask| {
                            (count + dp_cur.get(&mask).unwrap_or(&0)) % M
                        }),
                    );
                }
            }
            dp_cur = dp_next;
        }

        dp_cur.values().fold(0, |count, &n| (count + n) % M)
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

/// 2163h Minimum Difference in Sums After Removal of Elements
struct Sol2163 {}

impl Sol2163 {
    pub fn minimum_difference(nums: Vec<i32>) -> i64 {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let n = nums.len() / 3;
        let mut part1 = vec![0i64; n + 1];
        let mut sum = 0i64;

        let mut ql = BinaryHeap::new();
        for &v in nums.iter().take(n) {
            sum += v as i64;
            ql.push(v);
        }

        part1[0] = sum;
        for i in n..2 * n {
            sum += nums[i] as i64;
            ql.push(nums[i]);
            sum -= ql.pop().unwrap() as i64;
            part1[i - (n - 1)] = sum;
        }

        let mut part2 = 0i64;

        let mut qr = BinaryHeap::new();
        for i in (2 * n..3 * n).rev() {
            part2 += nums[i] as i64;
            qr.push(Reverse(nums[i]));
        }

        let mut m_diff = part1[n] - part2;
        for i in (n..2 * n).rev() {
            part2 += nums[i] as i64;
            qr.push(Reverse(nums[i]));
            if let Some(Reverse(val)) = qr.pop() {
                part2 -= val as i64;
            }
            m_diff = m_diff.min(part1[i - n] - part2);
        }

        m_diff
    }
}

/// 2787m Ways to Express an Integer as Sum of Powers
struct Sol2787 {}

impl Sol2787 {
    pub fn number_of_ways(n: i32, x: i32) -> i32 {
        const M: i64 = 1e9 as i64 + 7;

        let n = n as usize;
        let mut ks = vec![vec![0; n + 1]; n + 1]; // Knapsack

        let items: Vec<_> = (0..=n).map(|p| (p as i64).pow(x as u32)).collect();
        println!("-> {items:?}");

        ks[0][0] = 1;
        for item in 1..=n {
            let power = items[item] as usize;
            for capacity in 0..=n {
                ks[item][capacity] = ks[item - 1][capacity];
                if power <= capacity {
                    ks[item][capacity] = (ks[item][capacity] + ks[item - 1][capacity - power]) % M
                }
            }
        }

        println!("-> {ks:?}");

        ks[n][n] as _
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

/// 2901m Longest Unequal Adjacent Groups Subsequence II
struct Sol2901;

impl Sol2901 {
    pub fn get_words_in_longest_subsequence(words: Vec<String>, groups: Vec<i32>) -> Vec<String> {
        let mut dists = vec![vec![]; groups.len()];
        for (i, x) in words.iter().enumerate() {
            for y in words.iter().take(i) {
                dists[i].push(if x.len() != y.len() {
                    0
                } else {
                    x.chars()
                        .zip(y.chars())
                        .fold(0, |d, (x, y)| if x == y { d } else { d + 1 })
                });
            }
            dists[i].push(0);
        }

        println!("-> {dists:?}");

        fn valid_distance(x: &str, y: &str) -> bool {
            use std::cmp::Ordering::Equal;
            match x.len().cmp(&y.len()) {
                Equal => {
                    x.chars()
                        .zip(y.chars())
                        .filter(|(x, y)| x != y)
                        .collect::<Vec<_>>()
                        .len()
                        == 1
                }
                _ => false,
            }
        }

        let mut picks = vec![usize::MAX; groups.len()];
        let mut longest = vec![1; groups.len()];

        let (mut lmax, mut ilast) = (1, 0);

        for i in 1..groups.len() {
            for j in 0..i {
                if groups[j] != groups[i]
                    && (dists[i][j] == 1 && valid_distance(&words[i], &words[j]))
                    && longest[i] < longest[j] + 1
                {
                    longest[i] = longest[j] + 1;
                    picks[i] = j;
                }
            }

            if longest[i] > lmax {
                (lmax, ilast) = (longest[i], i);
            }
        }

        println!("-> {lmax} | {longest:?}");
        println!("-> {ilast} | {picks:?}");

        let mut lss = vec![];
        while ilast != usize::MAX {
            lss.push(words[ilast].to_owned());
            ilast = picks[ilast];
        }
        lss.reverse();

        lss
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

/// 3068h Find the Maximum Sum of Node Values
struct Sol3068 {}

impl Sol3068 {
    pub fn maximum_value_sum(nums: Vec<i32>, k: i32, edges: Vec<Vec<i32>>) -> i64 {
        println!("-> {edges:?}");

        fn recursive(
            start: usize,
            xors: usize,
            nums: &[i32],
            k: i32,
            memo: &mut [[i64; 2]],
        ) -> i64 {
            if start == nums.len() {
                return match xors & 1 {
                    0 => 0,
                    _ => i64::MIN,
                };
            }

            if memo[start][xors] != -1 {
                return memo[start][xors];
            }

            let xsum = (nums[start] as i64 + recursive(start + 1, xors, nums, k, memo))
                .max((nums[start] ^ k) as i64 + recursive(start + 1, xors ^ 1, nums, k, memo));
            memo[start][xors] = xsum;
            xsum
        }

        let mut memo = vec![[-1, -1]; nums.len()];
        println!(":: {}", recursive(0, 0, &nums, k, &mut memo));

        let tabulation = || {
            let mut dp = vec![[-1, -1]; nums.len() + 1];
            (dp[nums.len()][0], dp[nums.len()][1]) = (0, i64::MIN);

            for i in (0..nums.len()).rev() {
                for xor in [0, 1] {
                    dp[i][xor] = (dp[i + 1][xor] + nums[i] as i64)
                        .max(dp[i + 1][xor ^ 1] + (nums[i] ^ k) as i64);
                }
            }

            dp[0][0]
        };
        println!(":: {}", tabulation());

        let mut changes: Vec<_> = nums.iter().map(|&n| (n ^ k) - n).collect();
        changes.sort_unstable_by_key(|&n| std::cmp::Reverse(n));

        changes
            .chunks(2)
            .take_while(|chunk| chunk.len() == 2)
            .inspect(|chunk| println!("-> {chunk:?}"))
            .map(|v| (v[0] + v[1]) as i64)
            .take_while(|&diff| diff > 0)
            .sum::<i64>()
            + nums.iter().map(|&n| n as i64).sum::<i64>()
    }
}

/// 3333h Find the Original Typed String II
struct Sol3333 {}

impl Sol3333 {
    /// 1 <= N <= 10^5
    /// 1 <= k <= 2000
    pub fn possible_string_count(word: String, k: i32) -> i32 {
        let mut f = 1;
        let mut freqs = word
            .chars()
            .collect::<Vec<_>>()
            .windows(2)
            .fold(vec![], |mut freqs, w| {
                if w[0] == w[1] {
                    f += 1;
                } else {
                    freqs.push(f);
                    f = 1;
                }
                freqs
            });
        freqs.push(f);
        println!("-> {freqs:?}");

        const M: i64 = 1e9 as i64 + 7;

        let k = k as usize;
        let total = freqs.iter().fold(1, |total, &f| f * total % M);
        if freqs.len() >= k {
            return total as _;
        }

        let mut counts = vec![vec![0; k]; freqs.len() + 1];
        counts[0][0] = 1;

        for g in 0..freqs.len() {
            for l in 1..k {
                for x in 1..=freqs[g] as usize {
                    if l >= x {
                        counts[g + 1][l] = (counts[g + 1][l] + counts[g][l - x]) % M;
                    }
                }
            }
        }

        counts
            .last()
            .unwrap()
            .iter()
            .fold(total, |total, &count| (total - count + M) % M) as _
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

        println!("-> {freqs:?}");

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

        const MOD: i64 = 1e9 as i64 + 7;

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

/// 3363h Find the Maximum Number of Fruits Collected
struct Sol3363 {}

impl Sol3363 {
    pub fn max_collected_fruits(mut fruits: Vec<Vec<i32>>) -> i32 {
        fn walk(fruits: &[Vec<i32>]) -> i32 {
            let n = fruits.len();
            let mut prv = vec![i32::MIN; n];
            prv[n - 1] = fruits[0][n - 1];

            let mut cur = vec![i32::MIN; n];
            for (r, fruits) in fruits.iter().enumerate().skip(1).take(fruits.len() - 2) {
                for c in (n - 1 - r).max(r + 1)..n {
                    let mut best = prv[c];
                    if c > 0 {
                        best = best.max(prv[c - 1]);
                    }
                    if c < n - 1 {
                        best = best.max(prv[c + 1]);
                    }
                    cur[c] = best + fruits[c];
                }
                (0..n).for_each(|i| prv[i] = cur[i]);
            }

            prv[n - 1]
        }

        let mut xcolls = (0..fruits.len()).map(|d| fruits[d][d]).sum();
        xcolls += walk(&fruits);
        for r in 0..fruits.len() {
            for c in 0..r {
                (fruits[r][c], fruits[c][r]) = (fruits[c][r], fruits[r][c]);
            }
        }
        xcolls += walk(&fruits);

        xcolls
    }
}

/// 3459h Length of Longest V-Shaped Diagonal Segment
struct Sol3459 {}

impl Sol3459 {
    pub fn len_of_v_diagonal(grid: Vec<Vec<i32>>) -> i32 {
        use std::collections::HashMap;

        const DIRS: [(isize, isize); 4] = [(1, 1), (1, -1), (-1, -1), (-1, 1)];
        let mut cache: HashMap<(usize, usize, usize, bool), i32> = HashMap::new();

        fn search(
            grid: &[Vec<i32>],
            y: usize,
            x: usize,
            dir: usize,
            turned: bool,
            target: i32,
            cache: &mut HashMap<(usize, usize, usize, bool), i32>,
        ) -> i32 {
            let (r, c) = (
                y.wrapping_add_signed(DIRS[dir].0),
                x.wrapping_add_signed(DIRS[dir].1),
            );

            if r >= grid.len() || c >= grid[0].len() || grid[r][c] != target {
                return 0;
            }

            if let Some(&steps) = cache.get(&(r, c, dir, turned)) {
                return steps;
            }

            let mut steps = search(grid, r, c, dir, turned, 2 - grid[r][c], cache);
            if !turned {
                steps = search(grid, r, c, (dir + 1) % 4, true, 2 - grid[r][c], cache).max(steps);
            }
            cache.insert((r, c, dir, turned), steps + 1);

            steps + 1
        }

        grid.iter().enumerate().fold(0, |x_steps, (r, row)| {
            x_steps.max(
                row.iter()
                    .enumerate()
                    .filter_map(|(c, &g)| if g == 1 { Some(c) } else { None })
                    .map(|c| {
                        (0..4)
                            .map(|dir| search(&grid, r, c, dir, false, 2, &mut cache) + 1)
                            .max()
                            .unwrap_or(0)
                    })
                    .max()
                    .unwrap_or(0),
            )
        })
    }
}

#[cfg(test)]
mod tests;
