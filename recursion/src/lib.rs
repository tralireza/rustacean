//! # Rust :: Recursion, Backtracking

/// 37 Sudoku Solver
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
        for r in board {
            println!(" -> {:?}", r);
        }
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
}
