//! # Trie, Suffix Tree & Suffix Array (Rust)

/// 1163h Last Substring in Lexicographical Order
struct Sol1163 {}

impl Sol1163 {
    pub fn last_substring(s: String) -> String {
        let (mut i, mut j, mut k) = (0, 1, 0);

        let s: Vec<_> = s.chars().collect();
        let n = s.len();

        use std::cmp::Ordering::*;
        while j + k < n {
            match s[i + k].cmp(&s[j + k]) {
                Equal => k += 1,
                Greater => {
                    j += k + 1;
                    k = 0;
                }
                Less => {
                    i = (i + k + 1).max(j);
                    j = i + 1;
                    k = 0;
                }
            }
        }

        s.iter().skip(i).collect()
    }
}

#[cfg(test)]
mod tests;
