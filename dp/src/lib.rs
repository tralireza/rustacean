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

/// 1524m Number of Sub-arrays With Odd Sum
struct Sol1524;

impl Sol1524 {
    /// 1 <= N <= 10^5
    pub fn num_of_subarrays(arr: Vec<i32>) -> i32 {
        const M: usize = 1e9 as usize + 7;

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

        (odp.into_iter().sum::<usize>() % M) as i32
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
mod tests {
    use super::*;

    #[test]
    fn test_115() {
        assert_eq!(
            Sol115::num_distinct("rabbbit".to_string(), "rabbit".to_string()),
            3
        );
        assert_eq!(
            Sol115::num_distinct("babgbag".to_string(), "bag".to_string()),
            5
        );
    }

    #[test]
    fn test_132() {
        assert_eq!(Sol132::min_cut("aab".to_string()), 1);
        assert_eq!(Sol132::min_cut("a".to_string()), 0);
        assert_eq!(Sol132::min_cut("ab".to_string()), 1);
    }

    #[test]
    fn test_174() {
        assert_eq!(
            Sol174::calculate_minimum_hp(vec![vec![-2, -3, 3], vec![-5, -10, 1], vec![10, 30, -5]]),
            7
        );
        assert_eq!(Sol174::calculate_minimum_hp(vec![vec![0]]), 1);
    }

    #[test]
    fn test_1524() {
        assert_eq!(Sol1524::num_of_subarrays(vec![1, 3, 5]), 4);
        assert_eq!(Sol1524::num_of_subarrays(vec![2, 4, 6]), 0);
        assert_eq!(Sol1524::num_of_subarrays(vec![1, 2, 3, 4, 5, 6, 7]), 16);
    }

    #[test]
    fn test_2836() {
        // 1 <= Receiver.Length <= 10^5,  1 <= k <= 10^10
        assert_eq!(Sol2836::get_max_function_value(vec![2, 0, 1], 4), 6);
        assert_eq!(Sol2836::get_max_function_value(vec![1, 1, 1, 2, 3], 3), 10);
    }
}
