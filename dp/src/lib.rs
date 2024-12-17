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

#[cfg(test)]
mod tests {
    use super::*;

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
}
