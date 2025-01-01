//! Prefix Sum

/// 1422 Maximum Score After Splitting a String
struct Solution1422;

impl Solution1422 {
    pub fn max_score(s: String) -> i32 {
        let s = s.as_bytes();

        let (mut lzero, mut rone) = (((s[0] - b'0') ^ 1) as i32, 0i32);
        for i in (1..s.len()).rev() {
            match s[i] {
                b'1' => rone += 1,
                _ => (),
            }
        }

        let mut xscore = 0;
        for i in 1..s.len() - 1 {
            xscore = xscore.max(lzero + rone);
            match s[i] {
                b'1' => rone -= 1,
                b'0' => lzero += 1,
                _ => (),
            }
        }

        xscore
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution1422() {
        assert_eq!(Solution1422::max_score(String::from("011101")), 5);
        assert_eq!(Solution1422::max_score(String::from("00111")), 5);
    }
}
