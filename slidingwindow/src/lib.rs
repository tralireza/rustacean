//! # Rust :: Sliding Window

/// 1358m Number of Substrings Containing All Three Characters
struct Sol1358;

impl Sol1358 {
    pub fn number_of_substrings(s: String) -> i32 {
        use std::collections::HashMap;

        let mut count = 0;
        let mut frq = HashMap::new();

        let mut left = 0;
        for (right, chr) in s.chars().enumerate() {
            frq.entry(chr).and_modify(|f| *f += 1).or_insert(1);

            println!("-> {:?}", frq);

            while frq.len() == 3 {
                count += s.len() - right;

                let lchr = s.as_bytes()[left] as char;
                frq.entry(lchr).and_modify(|f| *f -= 1);
                if frq[&lchr] == 0 {
                    frq.remove(&lchr);
                }

                left += 1;
            }
        }

        count as i32
    }
}

/// 2302h Count Subarrays With Score Less Than K
struct Sol2302;

impl Sol2302 {
    pub fn count_subarrays(nums: Vec<i32>, k: i64) -> i64 {
        fn rusty(nums: Vec<i32>, k: i64) -> i64 {
            let (mut psum, mut l) = (0, 0);
            nums.iter()
                .enumerate()
                .map(|(r, &n)| {
                    psum += n as i64;
                    while psum * (r - l + 1) as i64 >= k {
                        psum -= nums[l] as i64;
                        l += 1;
                    }
                    r - l + 1
                })
                .sum::<usize>() as _
        }
        println!(":: {}", rusty(nums.to_vec(), k));

        let mut count = 0;

        let mut l = 0;
        let mut psum = 0;
        for (r, n) in nums.iter().map(|&n| n as i64).enumerate() {
            psum += n;
            while psum * (r - l + 1) as i64 >= k {
                psum -= nums[l] as i64;
                l += 1;
            }
            count += r - l + 1;
        }

        count as i64
    }
}

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

/// 2401m Longest Nice Subarray
struct Sol2401;

impl Sol2401 {
    pub fn longest_nice_subarray(nums: Vec<i32>) -> i32 {
        nums.iter()
            .enumerate()
            .fold((0, 0, 0), |(mut l, xlen, mut bits), (r, &n)| {
                while bits & n != 0 {
                    bits ^= nums[l];
                    l += 1;
                }

                bits |= n;

                (l, xlen.max(r - l + 1), bits)
            })
            .1 as i32
    }
}

/// 2444h Count Subarrays With Fixed Bounds
struct Sol2444;

impl Sol2444 {
    /// 1 <= Min, Max, N_i <= 10^6
    pub fn count_subarrays(nums: Vec<i32>, min_k: i32, max_k: i32) -> i64 {
        let mut count = 0;

        let (mut i_x, mut i_m) = (None, None);
        let mut l = 0;
        for (r, &n) in nums.iter().enumerate() {
            if n < min_k || max_k < n {
                (i_m, i_x) = (None, None);
                l = r + 1;
            } else {
                if n == min_k {
                    i_m = Some(r);
                }
                if n == max_k {
                    i_x = Some(r);
                }

                match (i_m, i_x) {
                    (Some(m), Some(x)) => count += x.min(m) - l + 1,
                    _ => (),
                }
            }
        }

        count as i64
    }
}

/// 2537m Count the Number of Good Subarrays
struct Sol2537;

impl Sol2537 {
    /// 1 <= N_i, k <= 10^9
    pub fn count_good(nums: Vec<i32>, k: i32) -> i64 {
        use std::collections::HashMap;

        let mut freq = HashMap::new();
        let mut count = 0;

        let mut left = 0;
        let mut window = 0;
        for (right, &n) in nums.iter().enumerate() {
            freq.entry(n).and_modify(|f| *f += 1).or_insert(1);

            let f = freq[&n];
            window += f * (f - 1) / 2 - (f - 1) * (f - 2) / 2; // ~ f - 1

            println!("-> +W {:?}", (right, window));

            while window >= k {
                count += nums.len() - right;

                freq.entry(nums[left]).and_modify(|f| *f -= 1);

                let f = freq[&nums[left]];
                window -= (f + 1) * f / 2 - f * (f - 1) / 2; // ~ f

                println!("-> w- {:?}", (right, window));

                left += 1;
            }
        }

        println!("-> {:?}", freq);

        count as i64
    }
}

/// 2799m Count Complete Subarrays in an Array
struct Sol2799;

impl Sol2799 {
    pub fn count_complete_subarrays(nums: Vec<i32>) -> i32 {
        use std::collections::{HashMap, HashSet};

        let uset: HashSet<i32> = HashSet::from_iter(nums.to_vec());
        println!("-> {:?}", uset);

        let mut left = 0;
        let mut w = 0;

        let mut frq = HashMap::new();
        nums.iter().enumerate().fold(0, |mut count, (right, &n)| {
            frq.entry(n).and_modify(|f| *f += 1).or_insert(1);
            if frq[&n] == 1 {
                w += 1;
            }

            while w == uset.len() {
                count += nums.len() - right;

                frq.entry(nums[left]).and_modify(|f| *f -= 1);
                if frq[&nums[left]] == 0 {
                    w -= 1;
                }
                left += 1;
            }

            count
        }) as i32
    }
}

/// 3191m Minimum Operations to Make Binary Array Elements Equal to One I
struct Sol3191;

impl Sol3191 {
    pub fn min_operations(nums: Vec<i32>) -> i32 {
        let mut ops = 0;

        let mut nums = nums;
        for s in 0..nums.len() - 2 {
            if nums[s] == 0 {
                for w in 0..3 {
                    nums[s + w] ^= 1;
                }

                ops += 1;
            }
        }

        if nums.contains(&0) {
            return -1;
        }
        ops
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
mod tests;
