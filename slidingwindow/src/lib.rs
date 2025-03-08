//! # Rust :: Sliding Window

/// 2379m Minimum Recolors to Get K Consecutive Black Blocks
struct Sol2379;

impl Sol2379 {
    pub fn minimum_recolors(blocks: String, k: i32) -> i32 {
        println!(
            "** {} -> {}",
            blocks,
            blocks
                .as_bytes()
                .windows(k as usize)
                .fold(usize::MAX, |recolors, w| recolors
                    .min(w.iter().filter(|&b| b == &b'W').count()))
        );

        let mut recolors = i32::MAX;

        let mut cur = 0;
        let mut left = 0;

        let blocks = blocks.as_bytes();
        for right in 0..blocks.len() {
            cur += match blocks[right] {
                b'W' => 1,
                _ => 0,
            };

            if right - left + 1 >= k as usize {
                recolors = recolors.min(cur);

                cur -= match blocks[left] {
                    b'W' => 1,
                    _ => 0,
                };

                left += 1;
            }
        }

        match recolors {
            i32::MAX => 0,
            _ => recolors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2379() {
        for (rst, blocks, k) in [
            (3, "WBBWWBBWBW".to_string(), 7),
            (0, "WBWBBBW".to_string(), 2),
        ] {
            assert_eq!(Sol2379::minimum_recolors(blocks, k), rst);
        }
    }
}
