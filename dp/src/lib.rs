//! Dynamic Programming

/// 10h Regular Expression Matching
pub struct Solution10;

impl Solution10 {
    pub fn is_match(s: String, p: String) -> bool {
        let mut dp = vec![vec![false; p.len() + 1]; s.len() + 1];

        let s = s.as_bytes();
        let p = p.as_bytes();

        dp[s.len()][p.len()] = true;

        for i in (0..=s.len()).rev() {
            for j in (0..p.len()).rev() {
                let fmatch = i < s.len() && (p[j] == s[i] || p[j] == b'.');
                if j + 1 < p.len() && p[j + 1] == b'*' {
                    dp[i][j] = dp[i][j + 2] || fmatch && dp[i + 1][j];
                } else {
                    dp[i][j] = fmatch && dp[i + 1][j + 1];
                }
            }
        }

        println!(" -> {:?}", dp);

        dp[0][0]
    }
}

/// 44h Wildcard Matching
pub struct Solution44;

impl Solution44 {
    pub fn is_match(s: String, p: String) -> bool {
        let mut dp = vec![vec![false; p.len() + 1]; s.len() + 1];
        dp[s.len()][p.len()] = true;

        let s = s.as_bytes();
        let p = p.as_bytes();

        for i in (0..=s.len()).rev() {
            for j in (0..p.len()).rev() {
                if p[j] == b'*' {
                    dp[i][j] = dp[i][j + 1] || i < s.len() && dp[i + 1][j];
                } else if i < s.len() {
                    dp[i][j] = (s[i] == p[j] || p[j] == b'?') && dp[i + 1][j + 1];
                }
            }
        }

        println!(" -> {:?}", dp);

        dp[0][0]
    }
}

/// 91m Decode Ways
struct Solution91;

impl Solution91 {
    pub fn num_decodings(s: String) -> i32 {
        let mut dp = vec![0; s.len() + 1];

        let s = s.as_bytes();
        println!(" -> :: {:?}", s);

        if s[0] == b'0' {
            return 0;
        }

        dp[0] = 1;
        dp[1] = 1;

        for i in 1..s.len() {
            dp[i + 1] += match s[i] - b'0' {
                1..=9 => dp[i],
                _ => 0,
            };
            dp[i + 1] += match 10 * (s[i - 1] - b'0') + s[i] - b'0' {
                10..=26 => dp[i - 1],
                _ => 0,
            };

            println!("{} -> {:?}", i, dp);
        }

        dp[s.len()]
    }
}

/// 3393m Count Paths With the Given XOR Value
struct Solution3393;

impl Solution3393 {
    pub fn count_paths_with_xor_value(grid: Vec<Vec<i32>>, k: i32) -> i32 {
        let (rows, cols) = (grid.len(), grid[0].len());
        const M: i32 = 1000_000_007;

        let mut dp = vec![vec![vec![0; 16]; cols]; rows];

        dp[0][0][grid[0][0] as usize] = 1;
        for r in 1..rows {
            for x in 0..16 {
                dp[r][0][grid[r][0] as usize ^ x] += dp[r - 1][0][x];
            }
        }
        for c in 1..cols {
            for x in 0..16 {
                dp[0][c][grid[0][c] as usize ^ x] += dp[0][c - 1][x];
            }
        }

        for r in 1..rows {
            for c in 1..cols {
                for x in 0..16 {
                    dp[r][c][grid[r][c] as usize ^ x] += (dp[r - 1][c][x] + dp[r][c - 1][x]) % M;
                }
            }
        }

        println!("{:?}", dp);

        dp[rows - 1][cols - 1][k as usize]
    }
}

/// 494m Target Sum
struct Solution494;

impl Solution494 {
    pub fn find_target_sum_ways(nums: Vec<i32>, target: i32) -> i32 {
        let tsum = nums.iter().fold(0, |acc, v| acc + v);
        if target.abs() > tsum {
            return 0;
        }

        let mut dp = vec![vec![0; (2 * tsum + 1) as usize]; nums.len()];
        dp[0][(tsum + nums[0]) as usize] = 1;
        dp[0][(tsum - nums[0]) as usize] += 1;

        for i in 1..nums.len() {
            for t in -tsum..=tsum {
                if dp[i - 1][(tsum + t) as usize] > 0 {
                    dp[i][(tsum + t + nums[i]) as usize] += dp[i - 1][(tsum + t) as usize];
                    dp[i][(tsum + t - nums[i]) as usize] += dp[i - 1][(tsum + t) as usize];
                }
            }
        }

        println!(" -> {:?}", dp);

        dp[nums.len() - 1][(tsum + target) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution494() {
        assert_eq!(Solution494::find_target_sum_ways(vec![1, 1, 1, 1, 1], 3), 5);
        assert_eq!(Solution494::find_target_sum_ways(vec![1], 1), 1);
        assert_eq!(
            Solution494::find_target_sum_ways(vec![0, 0, 0, 0, 0, 0, 0, 0, 1], 1),
            256
        );
    }

    #[test]
    fn test_solution10() {
        assert!(!Solution10::is_match("aa".to_string(), "a".to_string()));
        assert!(Solution10::is_match("aa".to_string(), "a*".to_string()));
        assert!(Solution10::is_match("ab".to_string(), ".*".to_string()));
        assert!(Solution10::is_match("aab".to_string(), "c*a*b".to_string()));
    }

    #[test]
    fn test_solution44() {
        assert!(!Solution44::is_match("aa".to_string(), "a".to_string()));
        assert!(Solution44::is_match("aa".to_string(), "*".to_string()));
        assert!(!Solution44::is_match("cb".to_string(), "?a".to_string()));
    }

    #[test]
    fn test_solution91() {
        assert_eq!(Solution91::num_decodings("12".to_string()), 2);
        assert_eq!(Solution91::num_decodings("226".to_string()), 3);
        assert_eq!(Solution91::num_decodings("06".to_string()), 0);
        assert_eq!(Solution91::num_decodings("2101".to_string()), 1);
    }

    #[test]
    fn test_solution3393() {
        assert_eq!(
            Solution3393::count_paths_with_xor_value(
                vec![vec![2, 1, 5], vec![7, 10, 0], vec![12, 6, 4]],
                11,
            ),
            3,
        );
        println!();
        assert_eq!(
            Solution3393::count_paths_with_xor_value(
                vec![vec![1, 3, 3, 3], vec![0, 3, 3, 2], vec![3, 0, 1, 1]],
                2,
            ),
            5,
        );
    }
}
