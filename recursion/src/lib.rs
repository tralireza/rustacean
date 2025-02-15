//! # Rust :: Recursion, Backtracking

#![feature(test)]
extern crate test;

/// 37h Sudoku Solver
struct Sol37;

impl Sol37 {
    pub fn solve_sudoku(board: &mut Vec<Vec<char>>) {
        fn valid(board: &mut Vec<Vec<char>>, r: usize, c: usize, digit: char) -> bool {
            for i in 0..9 {
                if board[i][c] == digit || board[r][i] == digit {
                    return false;
                }
            }

            let (r, c) = (r / 3 * 3, c / 3 * 3);
            for x in r..=r + 2 {
                for y in c..=c + 2 {
                    if board[x][y] == digit {
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
    fn test_2698() {
        // 1 <= n <= 1000
        assert_eq!(Sol2698::punishment_number(10), 182);
        assert_eq!(Sol2698::punishment_number(37), 1478);
    }
}
