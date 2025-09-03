//! # Rust :: Recursion, Backtracking

#![feature(test)]
extern crate test;

/// 37h Sudoku Solver
struct Sol37;

impl Sol37 {
    pub fn solve_sudoku(board: &mut Vec<Vec<char>>) {
        fn valid(board: &[Vec<char>], r: usize, c: usize, digit: char) -> bool {
            for i in 0..9 {
                if board[i][c] == digit || board[r][i] == digit {
                    return false;
                }
            }

            for row in board.iter().skip(r / 3 * 3).take(3) {
                for &v in row.iter().skip(c / 3 * 3).take(3) {
                    if v == digit {
                        return false;
                    }
                }
            }

            true
        }

        fn solve(board: &mut Vec<Vec<char>>, r: usize, c: usize) -> bool {
            if r == 9 {
                return true;
            }

            let (mut x, mut y) = (r, c + 1);
            if y == 9 {
                (x, y) = (r + 1, 0);
            }

            if board[r][c] != '.' {
                return solve(board, x, y);
            }

            for digit in ['1', '2', '3', '4', '5', '6', '7', '8', '9'] {
                if valid(board, r, c, digit) {
                    board[r][c] = digit;
                    if solve(board, x, y) {
                        return true;
                    }
                    board[r][c] = '.';
                }
            }
            false
        }

        solve(board, 0, 0);
        for r in board.iter() {
            println!(" -> {:?}", r);
        }
    }
}

/// 50m Power(x, n)
struct Sol50;

impl Sol50 {
    /// -100 <= x <= 100
    /// -2^31 <= n <= 2^31-1
    pub fn my_pow(x: f64, n: i32) -> f64 {
        let mut p = 1.;

        let mut x = x;
        let mut e = if n < 0 { -(n as i64) } else { n as i64 };
        while e > 0 {
            if e & 1 == 1 {
                p *= x;
            }
            x *= x;
            e >>= 1;
        }

        use std::cmp::Ordering::*;
        match n.cmp(&0) {
            Less => 1. / p,
            _ => p,
        }
    }
}

/// 60h Permutation Sequence
struct Sol60;

impl Sol60 {
    pub fn get_permutation(n: i32, k: i32) -> String {
        // 1 <= n <= 9
        let mut seq = vec![];
        for chr in ('1'..='9').take(n as usize) {
            seq.push(chr);
        }

        fn perms(n: i32, p: usize, seq: &mut Vec<char>) {
            if p == n as usize {
                println!("-> {:?}", seq);
                return;
            }

            perms(n, p + 1, seq);
            for i in p + 1..n as usize {
                (seq[i], seq[p]) = (seq[p], seq[i]);
                perms(n, p + 1, seq);
                (seq[i], seq[p]) = (seq[p], seq[i]);
            }
        }

        perms(n, 0, &mut seq);

        let mut rst = vec![];
        let mut vis = vec![false; n as usize];

        fn perms_sql(n: usize, k: i32, vis: &mut Vec<bool>, rst: &mut Vec<char>) -> i32 {
            let mut k = k;
            if rst.len() == n {
                k -= 1;
                if k == 0 {
                    println!(":: {:?}", rst);
                }
                return k;
            }

            for (i, chr) in ('1'..='9').enumerate().take(n) {
                if !vis[i] {
                    vis[i] = true;
                    rst.push(chr);

                    k = perms_sql(n, k, vis, rst);
                    if k == 0 {
                        return 0;
                    }

                    rst.pop();
                    vis[i] = false;
                }
            }

            k
        }

        perms_sql(n as usize, k, &mut vis, &mut rst);
        rst.iter().collect()
    }
}

/// 282h Expression Add Operators
struct Sol282;

impl Sol282 {
    pub fn add_operators(num: String, target: i32) -> Vec<String> {
        println!("** {:?}", num);

        let mut rst = vec![];

        fn search(
            num: &str,
            target: i32,
            rst: &mut Vec<String>,
            start: usize,
            prv_opr: i64,
            cur_val: i64,
            expr: &str,
        ) -> Result<(), std::num::ParseIntError> {
            if start == num.len() {
                if cur_val == target as i64 {
                    rst.push(expr.to_string());
                }
                return Ok(());
            }

            for i in start..num.len() {
                if num.as_bytes()[start] == b'0' && i > start {
                    break;
                }

                let v = num[start..i + 1].parse()?;
                if start == 0 {
                    search(
                        num,
                        target,
                        rst,
                        i + 1,
                        v,
                        cur_val + v,
                        &(expr.to_string() + &num[start..i + 1]),
                    )?;
                } else {
                    search(
                        num,
                        target,
                        rst,
                        i + 1,
                        v,
                        cur_val + v,
                        &(expr.to_string() + "+" + &num[start..i + 1]),
                    )?;
                    search(
                        num,
                        target,
                        rst,
                        i + 1,
                        -v,
                        cur_val - v,
                        &(expr.to_string() + "-" + &num[start..i + 1]),
                    )?;
                    search(
                        num,
                        target,
                        rst,
                        i + 1,
                        prv_opr * v,
                        cur_val - prv_opr + prv_opr * v,
                        &(expr.to_string() + "*" + &num[start..i + 1]),
                    )?;
                }
            }

            Ok(())
        }

        let _ = search(&num, target, &mut rst, 0, 0, 0, "");
        println!(":: {:?}", rst);

        rst
    }
}

/// 301h Remove Invalid Parentheses
struct Sol301 {}

impl Sol301 {
    /// 1 <= N <= 25
    pub fn remove_invalid_parentheses(s: String) -> Vec<String> {
        use std::collections::{HashMap, HashSet};

        let chrs: Vec<char> = s.chars().collect();
        println!("* {chrs:?}");

        fn search(
            start: usize,
            opens: usize,
            closes: usize,
            chrs: &[char],
            picks: &mut [bool],
            lengths: &mut HashMap<usize, HashSet<String>>,
        ) {
            if start == picks.len() {
                let s: String = chrs
                    .iter()
                    .zip(picks.iter())
                    .filter(|(_, pick)| **pick)
                    .map(|(&chr, _)| chr)
                    .collect();

                let valid = || -> bool {
                    let mut stack = 0;
                    for chr in s.chars() {
                        match chr {
                            '(' => stack += 1,
                            ')' => {
                                if stack == 0 {
                                    return false;
                                } else {
                                    stack -= 1;
                                }
                            }
                            _ => (),
                        }
                    }
                    stack == 0
                };

                if opens == closes && valid() {
                    let entry = lengths.entry(s.len()).or_default();
                    entry.insert(s);
                }

                return;
            }

            match chrs[start] {
                '(' => {
                    if opens > 0 {
                        search(start + 1, opens - 1, closes, chrs, picks, lengths);
                    }

                    picks[start] = true;
                    search(start + 1, opens, closes, chrs, picks, lengths);
                    picks[start] = false;
                }
                ')' => {
                    if closes > 0 {
                        search(start + 1, opens, closes - 1, chrs, picks, lengths);
                    }

                    picks[start] = true;
                    search(start + 1, opens, closes, chrs, picks, lengths);
                    picks[start] = false;
                }
                _ => {
                    picks[start] = true;
                    search(start + 1, opens, closes, chrs, picks, lengths);
                }
            }
        }

        let mut picks = vec![false; chrs.len()];
        let mut lengths: HashMap<usize, HashSet<String>> = HashMap::new();

        let (mut opens, mut closes) = (0, 0);
        for chr in chrs.iter() {
            match chr {
                '(' => opens += 1,
                ')' => {
                    if opens > 0 {
                        opens -= 1;
                    } else {
                        closes += 1;
                    }
                }
                _ => (),
            }
        }
        println!("-> Extra# (Must Remove): {opens} (   {closes} )");

        search(0, opens, closes, &chrs, &mut picks, &mut lengths);

        println!("-> {lengths:?}");

        match lengths.keys().max() {
            Some(lmax) => lengths[lmax].clone().into_iter().collect(),
            _ => vec![],
        }
    }
}

/// 386m Lexicographical Numbers
struct Sol386 {}

impl Sol386 {
    pub fn lexical_order(n: i32) -> Vec<i32> {
        let mut lorder = vec![];

        fn dfs(v: i32, n: i32, lorder: &mut Vec<i32>) {
            lorder.push(v);
            (0..=9)
                .take_while(|d| 10 * v + d <= n)
                .for_each(|d| dfs(10 * v + d, n, lorder));
        }

        for v in 1..=9.min(n) {
            dfs(v, n, &mut lorder);
        }

        lorder
    }
}

/// 679h 24 Game
struct Sol679 {}

impl Sol679 {
    /// L_cards = 4, 1 <= Cards_i <= 9
    pub fn judge_point24(cards: Vec<i32>) -> bool {
        fn possible_results(a: f64, b: f64) -> Vec<f64> {
            let mut rs = vec![a + b, a - b, b - a, a * b];
            if b != 0.0 {
                rs.push(a / b);
            }
            if a != 0.0 {
                rs.push(b / a);
            }

            println!("-> ({a}, {b}) :: {rs:?}");
            rs
        }

        fn search(cards: &[f64]) -> bool {
            println!("-> {cards:?}");

            if cards.len() == 1 {
                return (cards[0] - 24.0).abs() <= 1e-5;
            }

            for a in 0..cards.len() {
                for b in a + 1..cards.len() {
                    let mut n_cards: Vec<_> = cards
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != a && i != b)
                        .map(|(_, &v)| v)
                        .collect();

                    for v in possible_results(cards[a], cards[b]) {
                        n_cards.push(v);
                        if search(&n_cards) {
                            return true;
                        }
                        n_cards.pop();
                    }
                }
            }

            false
        }

        search(&cards.iter().map(|&n| n as f64).collect::<Vec<_>>())
    }
}

/// 1079m Letter Tile Possibilities
struct Sol1079;

impl Sol1079 {
    pub fn num_tile_possibilities(tiles: String) -> i32 {
        use std::collections::HashSet;

        println!("|| {:?}", tiles);

        fn gen_combinations(
            hset: &mut HashSet<String>,
            used: &mut Vec<bool>,
            tiles: &mut Vec<char>,
            tile: &mut String,
        ) {
            hset.insert(tile.clone());

            for i in 0..tiles.len() {
                if !used[i] {
                    used[i] = true;
                    tile.push(tiles[i]);

                    gen_combinations(hset, used, tiles, tile);

                    used[i] = false;
                    tile.pop();
                }
            }
        }

        let mut hset = HashSet::new();
        let mut used = vec![false; tiles.len()];

        gen_combinations(
            &mut hset,
            &mut used,
            &mut tiles.chars().collect(),
            &mut "".to_string(),
        );

        println!("-> {:?}", hset);

        hset.len() as i32 - 1
    }
}

/// 1718m Construct the Lexicographically Largest Valid Sequence
struct Sol1718;

impl Sol1718 {
    pub fn construct_distanced_sequence(n: i32) -> Vec<i32> {
        let mut rst = vec![0; 2 * n as usize - 1];
        let mut numbers = vec![false; 1 + n as usize];

        fn search(rst: &mut Vec<i32>, numbers: &mut Vec<bool>, start: usize) -> bool {
            if start == rst.len() {
                return true;
            }

            if rst[start] != 0 {
                return search(rst, numbers, start + 1);
            }

            for n in (1..numbers.len()).rev() {
                if numbers[n] {
                    continue;
                }

                rst[start] = n as i32;
                numbers[n] = true;

                if n == 1 && search(rst, numbers, start + 1) {
                    return true;
                }

                if n > 1 && start + n < rst.len() && rst[start + n] == 0 {
                    rst[start + n] = n as i32;
                    if search(rst, numbers, start + 1) {
                        return true;
                    }

                    rst[start + n] = 0; // backtrack
                }

                rst[start] = 0; // backtrack
                numbers[n] = false; // backtrack
            }

            false
        }

        search(&mut rst, &mut numbers, 0);

        println!(":: {:?}", rst);

        rst
    }
}

/// 1922m Count Good Numbers
struct Sol1922;

impl Sol1922 {
    /// 1 <= N <= 10^15
    pub fn count_good_numbers(n: i64) -> i32 {
        const M: i64 = 1e9 as i64 + 7;

        fn mpower(mut b: i64, mut e: i64) -> i64 {
            let mut mpower = 1;
            while e > 0 {
                if e & 1 == 1 {
                    mpower = (b * mpower) % M;
                }
                b = (b * b) % M;
                e >>= 1;
            }
            mpower
        }

        (mpower(5, (n + 1) / 2) * mpower(4, n / 2) % M) as i32
    }
}

/// 2044m Count Number of Maximum Bitwise-OR Subsets
struct Sol2044 {}

impl Sol2044 {
    pub fn count_max_or_subsets(nums: Vec<i32>) -> i32 {
        let x_or = nums.iter().fold(0, |x_or, &n| x_or | n);

        fn search(start: usize, or: i32, x_or: i32, nums: &[i32]) -> i32 {
            if start == nums.len() {
                return 0;
            }

            (if or | nums[start] == x_or { 1 } else { 0 })
                + search(start + 1, or, x_or, nums)
                + search(start + 1, or | nums[start], x_or, nums)
        }

        search(0, 0, x_or, &nums)
    }
}

/// 2375m Construct Smallest Number From DI String
struct Sol2375;

impl Sol2375 {
    pub fn smallest_number(pattern: String) -> String {
        println!("|| {}", pattern);

        fn search(rst: &mut String, last: usize, used: &mut [bool], pattern: &Vec<char>) -> bool {
            println!("-> {:?}", rst);

            if rst.len() == pattern.len() + 1 {
                return true;
            }

            for d in 1..=9 {
                if !used[d] {
                    match (pattern[rst.len() - 1], d < last) {
                        ('D', true) | ('I', false) => {
                            used[d] = true;
                            rst.push((b'0' + d as u8) as char);

                            if search(rst, d, used, pattern) {
                                return true;
                            }

                            used[d] = false;
                            rst.pop();
                        }
                        _ => (),
                    }
                }
            }

            false
        }

        let mut used = [false; 10];
        let pattern = pattern.chars().collect();

        let mut rst = String::new();
        for d in 1..=9 {
            rst.push((b'0' + d as u8) as char);
            used[d] = true;

            if search(&mut rst, d, &mut used, &pattern) {
                return rst;
            }

            rst.pop();
            used[d] = false;
        }

        "".to_string()
    }
}

/// 2698m Find the Punishment Number of an Integer
struct Sol2698;

impl Sol2698 {
    pub fn punishment_number(n: i32) -> i32 {
        fn partition(digits: &Vec<i32>, n: i32, parts: &mut [bool], start: usize) -> bool {
            println!("-> {:?}", (n, start, &parts));

            for p in start..digits.len() {
                parts[p] = true;

                let (mut dsum, mut csum) = (0, 0);
                for p in 0..digits.len() {
                    match parts[p] {
                        true => {
                            dsum += 10 * csum + digits[p];
                            csum = 0;
                        }
                        _ => csum = 10 * csum + digits[p],
                    }
                }

                dsum += csum;
                if n == dsum {
                    return true;
                }

                if partition(digits, n, parts, p + 1) {
                    return true;
                }

                parts[p] = false;
            }

            false
        }

        (1..=n)
            .filter(|&n| {
                let mut digits = vec![];
                let mut sqr = n * n;
                while sqr > 0 {
                    digits.push(sqr % 10);
                    sqr /= 10;
                }
                digits.reverse();

                partition(&digits, n, &mut vec![false; digits.len()], 0)
            })
            .map(|n| n * n)
            .sum::<i32>()
    }
}

#[cfg(test)]
mod tests;
