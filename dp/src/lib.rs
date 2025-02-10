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
                    if 1 << p & k != 0 {
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
    fn test_2836() {
        // 1 <= Receiver.Length <= 10^5,  1 <= k <= 10^10
        assert_eq!(Sol2836::get_max_function_value(vec![2, 0, 1], 4), 6);
        assert_eq!(Sol2836::get_max_function_value(vec![1, 1, 1, 2, 3], 3), 10);
    }
}
