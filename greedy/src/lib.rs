//! # Rust Greedy

#![feature(test)]

extern crate test;

/// 135h Candy
struct Sol135 {}

impl Sol135 {
    pub fn candy(ratings: Vec<i32>) -> i32 {
        let mut candies = vec![1; ratings.len()];

        for i in 0..candies.len() - 1 {
            if ratings[i] < ratings[i + 1] {
                candies[i + 1] = candies[i + 1].max(candies[i] + 1);
            }
        }

        for i in (1..candies.len()).rev() {
            if ratings[i - 1] > ratings[i] {
                candies[i - 1] = candies[i - 1].max(candies[i] + 1);
            }
        }

        println!("-> {candies:?}");

        candies.iter().sum()
    }
}

/// 781m Rabbits in Forest
struct Sol781;

impl Sol781 {
    /// 1 <= N <= 1000, 0 <= N_i < 10000
    pub fn num_rabbits(answers: Vec<i32>) -> i32 {
        let freq = answers.iter().fold([0; 1000], |mut freq, &f| {
            freq[f as usize] += 1;
            freq
        });

        freq.iter()
            .enumerate()
            .map(|(n, f)| (n as i32 + 1, f))
            .fold(0, |count, (n, f)| count + (f + n - 1) / n * n)
    }
}

/// 1007m Minimum Domino Rotations For Equal Row
struct Sol1007;

impl Sol1007 {
    pub fn min_domino_rotations(tops: Vec<i32>, bottoms: Vec<i32>) -> i32 {
        let mut r = i32::MAX;

        'LOOP: for n in 1..=6 {
            let mut top = 0;
            let mut bottom = 0;

            for (&t, &b) in tops.iter().zip(bottoms.iter()) {
                if t != n && b != n {
                    continue 'LOOP;
                }

                if t != n {
                    top += 1;
                }
                if b != n {
                    bottom += 1;
                }
            }

            r = r.min(top.min(bottom));
        }

        if r < i32::MAX {
            return r;
        }
        -1
    }
}

/// 2131m Longest Palindrome by Concatenating Two Letter Words
struct Sol2131 {}

impl Sol2131 {
    pub fn longest_palindrome(words: Vec<String>) -> i32 {
        use std::collections::HashMap;

        let mut fwords = HashMap::new();
        for word in &words {
            fwords.entry(word).and_modify(|f| *f += 1).or_insert(1);
        }

        println!("-> {fwords:?}");

        let mut extra = 0;
        fwords.keys().fold(0, |length, &w| match fwords.get(&w) {
            Some(&f) => {
                let chrs: Vec<_> = w.chars().collect();
                length
                    + match chrs
                        .iter()
                        .zip(chrs.iter().rev())
                        .all(|(chr1, chr2)| chr1 == chr2)
                    {
                        true => match f & 1 {
                            1 => {
                                extra = 2;
                                f - 1
                            }
                            _ => f,
                        },
                        _ => match fwords.get(&String::from_iter(chrs.iter().rev())) {
                            Some(&p) => f.min(p),
                            _ => 0,
                        },
                    }
            }
            _ => length,
        }) * 2
            + extra
    }
}

/// 2434m Using a Robot to Print the Lexicographically Smallest String
struct Sol2434 {}

impl Sol2434 {
    pub fn robot_with_string(s: String) -> String {
        use std::collections::HashMap;

        let mut freqs: HashMap<char, usize> = HashMap::new();
        for chr in s.chars() {
            freqs.entry(chr).and_modify(|f| *f += 1).or_insert(1);
        }

        let mut prints = vec![];

        let mut stack = vec![];
        for chr in s.chars() {
            stack.push(chr);
            freqs.entry(chr).and_modify(|f| *f -= 1);

            if let Some(marker) = ('a'..='z').find(|chr| freqs.contains_key(chr) && freqs[chr] != 0)
            {
                while let Some(&chr) = stack.last()
                    && chr <= marker
                {
                    prints.push(chr);
                    stack.pop();
                }
            }
        }

        while let Some(chr) = stack.pop() {
            prints.push(chr);
        }

        prints.iter().collect()
    }
}

/// 2900 Longest Unequal Adjacent Groups Subsequence I
struct Sol2900;

impl Sol2900 {
    /// 1 <= |words, groups| <= 100
    pub fn get_longest_subsequence(mut words: Vec<String>, groups: Vec<i32>) -> Vec<String> {
        words.reverse();
        groups
            .iter()
            .skip(1)
            .fold(
                (vec![words.pop().unwrap()], groups[0]),
                |(mut ls, cur_group), &g| {
                    if cur_group == g {
                        words.pop();
                        (ls, g)
                    } else {
                        ls.push(words.pop().unwrap());
                        (ls, g)
                    }
                },
            )
            .0
    }
}

/// 2918m Minimum Equal Sum of Two Arrays After Replacing Zeros
struct Sol2918;

impl Sol2918 {
    pub fn min_sum(nums1: Vec<i32>, nums2: Vec<i32>) -> i64 {
        let folder = |(sum, zeros), n| match n == 0 {
            true => (sum + 1, zeros + 1),
            _ => (sum + n as i64, zeros),
        };

        let (sum1, zeros1) = nums1.into_iter().fold((0, 0), folder);
        let (sum2, zeros2) = nums2.into_iter().fold((0, 0), folder);

        if sum1 > sum2 && zeros2 == 0 || sum2 > sum1 && zeros1 == 0 {
            return -1;
        }
        sum1.max(sum2)
    }
}

/// 3170m Lexicographically Minimum String After Removing Stars
struct Sol3170 {}

impl Sol3170 {
    pub fn clear_stars(s: String) -> String {
        let mut data = vec![vec![]; 26];
        let mut buffer: Vec<_> = s.chars().collect();

        for (i, chr) in s.as_bytes().iter().enumerate() {
            match chr {
                b'*' => {
                    for chr_data in data.iter_mut() {
                        if !chr_data.is_empty() {
                            if let Some(last) = chr_data.pop() {
                                buffer[last] = '*';
                            }

                            break;
                        }
                    }
                }
                _ => data[(chr - b'a') as usize].push(i),
            }
        }

        println!("-> {buffer:?}");

        buffer.iter().filter(|&chr| chr != &'*').collect()
    }
}

#[cfg(test)]
mod tests;
