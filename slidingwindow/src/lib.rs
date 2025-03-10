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

/// 3208m Alternating Groups II
struct Sol3208;

impl Sol3208 {
    pub fn number_of_alternating_groups(colors: Vec<i32>, k: i32) -> i32 {
        let (mut wsize, mut groups) = (1, 0);

        let mut prv = colors[0];
        for i in 1..colors.len() + k as usize - 1 {
            let cur = colors[i % colors.len()];
            match cur.cmp(&prv) {
                std::cmp::Ordering::Equal => wsize = 1,
                _ => {
                    wsize += 1;
                    if wsize >= k {
                        groups += 1;
                    }
                    prv = cur;
                }
            }
        }

        groups
    }
}

/// 3306m Count of Substrings Containing Every Vowel and K Consonants II
struct Sol3306;

impl Sol3306 {
    pub fn count_of_substrings(word: String, k: i32) -> i64 {
        println!("** {:?}", (&word, k));

        fn n_least(word: &str, k: i32) -> i64 {
            use std::collections::HashMap;

            let mut frq = HashMap::new();
            let mut k = k;

            let mut l = 0;
            let mut count = 0;
            for (r, chr) in word.chars().enumerate() {
                match chr {
                    'a' | 'e' | 'i' | 'o' | 'u' => {
                        frq.entry(chr).and_modify(|f| *f += 1).or_insert(1);
                    }
                    _ => {
                        k -= 1;
                    }
                }

                println!("-> {:?}", frq);

                while k <= 0 && frq.len() == 5 {
                    count += (word.len() - r) as i64;

                    if let Some(lchr) = word[l..].chars().next() {
                        match lchr {
                            'a' | 'e' | 'i' | 'o' | 'u' => {
                                frq.entry(lchr).and_modify(|f| *f -= 1);
                                if frq[&lchr] == 0 {
                                    frq.remove(&lchr);
                                }
                            }
                            _ => k += 1,
                        }
                    }

                    l += 1;
                }
            }

            count
        }

        n_least(&word, k) - n_least(&word, k + 1)
    }
}

#[cfg(test)]
mod test;
