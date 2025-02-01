//! # Array

/// 2017m Grid Game
struct Sol2017;

impl Sol2017 {
    pub fn grid_game(grid: Vec<Vec<i32>>) -> i64 {
        let mut top = grid[0].iter().map(|&n| n as i64).sum::<i64>();
        let mut bottom = 0;

        (0..grid[0].len()).fold(i64::MAX, |mx, i| {
            top -= grid[0][i] as i64;
            bottom += grid[1][i] as i64;
            mx.min(top.max(bottom - grid[1][i] as i64))
        })
    }
}

/// 3151 Special Array I
struct Sol3151;

impl Sol3151 {
    pub fn is_array_special(nums: Vec<i32>) -> bool {
        nums.windows(2)
            .fold(true, |r, v| if (v[0] ^ v[1]) & 1 == 0 { false } else { r })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2017() {
        assert_eq!(Sol2017::grid_game(vec![vec![2, 5, 4], vec![1, 5, 1]]), 4);
        assert_eq!(Sol2017::grid_game(vec![vec![3, 3, 1], vec![8, 5, 2]]), 4);
        assert_eq!(
            Sol2017::grid_game(vec![vec![1, 3, 1, 15], vec![1, 3, 3, 1]]),
            7
        );
    }

    #[test]
    fn test_3151() {
        assert_eq!(Sol3151::is_array_special(vec![1]), true);
        assert_eq!(Sol3151::is_array_special(vec![2, 1, 4]), true);
        assert_eq!(Sol3151::is_array_special(vec![4, 3, 1, 6]), false);
    }
}
