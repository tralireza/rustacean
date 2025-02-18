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
mod tests {
    use super::*;

    #[test]
    fn test_37() {
        Sol37::solve_sudoku(&mut vec![
            vec!['5', '3', '.', '.', '7', '.', '.', '.', '.'],
            vec!['6', '.', '.', '1', '9', '5', '.', '.', '.'],
            vec!['.', '9', '8', '.', '.', '.', '.', '6', '.'],
            vec!['8', '.', '.', '.', '6', '.', '.', '.', '3'],
            vec!['4', '.', '.', '8', '.', '3', '.', '.', '1'],
            vec!['7', '.', '.', '.', '2', '.', '.', '.', '6'],
            vec!['.', '6', '.', '.', '.', '.', '2', '8', '.'],
            vec!['.', '.', '.', '4', '1', '9', '.', '.', '5'],
            vec!['.', '.', '.', '.', '8', '.', '.', '7', '9'],
        ]);
    }

    #[bench]
    fn bench_37(b: &mut test::Bencher) {
        b.iter(|| test_37());
    }

    #[test]
    fn test_60() {
        assert_eq!(Sol60::get_permutation(3, 3), "213".to_string());
        assert_eq!(Sol60::get_permutation(4, 9), "2314".to_string());
        assert_eq!(Sol60::get_permutation(3, 1), "123".to_string());
    }

    #[test]
    fn test_1079() {
        assert_eq!(Sol1079::num_tile_possibilities("AAB".to_string()), 8);
        assert_eq!(Sol1079::num_tile_possibilities("AAABBC".to_string()), 188);
        assert_eq!(Sol1079::num_tile_possibilities("V".to_string()), 1);

        assert_eq!(
            Sol1079::num_tile_possibilities("ABCDEFG".to_string()),
            13699
        );
    }

    #[test]
    fn test_1718() {
        // 1 <= n <= 20
        assert_eq!(
            Sol1718::construct_distanced_sequence(3),
            vec![3, 1, 2, 3, 2]
        );
        assert_eq!(
            Sol1718::construct_distanced_sequence(5),
            vec![5, 3, 1, 4, 3, 5, 2, 4, 2]
        );
    }

    #[bench]
    fn bench_1718(b: &mut test::Bencher) {
        b.iter(|| Sol1718::construct_distanced_sequence(20));
    }

    #[test]
    fn test_2375() {
        assert_eq!(
            Sol2375::smallest_number("IIIDIDDD".to_string()),
            "123549876".to_string()
        );
        assert_eq!(
            Sol2375::smallest_number("DDD".to_string()),
            "4321".to_string()
        );
    }

    #[test]
    fn test_2698() {
        // 1 <= n <= 1000
        assert_eq!(Sol2698::punishment_number(10), 182);
        assert_eq!(Sol2698::punishment_number(37), 1478);
    }
}
