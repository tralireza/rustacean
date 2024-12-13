//! Dynamic Programming

/// 10h Regular Expression Matching
struct Solution10;

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
}
