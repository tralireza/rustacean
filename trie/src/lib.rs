//! # Trie, Suffix Tree & Suffix Array (Rust)

/// 1163h Last Substring in Lexicographical Order
struct Sol1163 {}

impl Sol1163 {
    pub fn last_substring(s: String) -> String {
        fn suffix_array(s: String) -> Vec<i32> {
            let n = s.len();

            let mut pfx_arr: Vec<i32> = s
                .as_bytes()
                .iter()
                .map(|&chr| (chr - b'a') as i32)
                .collect();

            let mut k = 1;
            let mut l = vec![(0, 0, 0); n];
            while (k >> 1) < n {
                let mut p = vec![0; n];

                for (i, t) in l.iter_mut().enumerate() {
                    t.0 = pfx_arr[i];
                    t.1 = if i + k < n { pfx_arr[i + k] } else { -1 };
                    t.2 = i;
                }
                println!("-> {k} {l:?}");

                l.sort_by(|a, b| {
                    if a.0 == b.0 {
                        a.1.cmp(&b.1)
                    } else {
                        a.0.cmp(&b.0)
                    }
                });
                println!("-> {k} {l:?}");

                for i in 0..n {
                    p[l[i].2] = if i > 0 && l[i].0 == l[i - 1].0 && l[i].1 == l[i - 1].1 {
                        p[l[i - 1].2]
                    } else {
                        i as i32
                    };
                }

                pfx_arr = p;
                k <<= 1;
            }

            println!(
                "-> {}",
                s.chars()
                    .skip(pfx_arr.iter().position(|&i| i as usize + 1 == n).unwrap())
                    .collect::<String>()
            );

            pfx_arr
        }
        println!("-> Suffix Array: {:?}", suffix_array(s.clone()));

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
