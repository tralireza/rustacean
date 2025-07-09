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

/// 630h Course Schedule III
struct Sol630 {}
impl Sol630 {
    pub fn schedule_course(courses: Vec<Vec<i32>>) -> i32 {
        use std::collections::BinaryHeap;

        let mut courses = courses;
        courses.sort_by_key(|course| course[1]);

        let mut start = 0;
        let mut pq = BinaryHeap::new();
        for course in courses {
            start += course[0];
            pq.push(course[0]);

            if start > course[1]
                && let Some(days) = pq.pop()
            {
                start -= days;
            }
        }

        pq.len() as _
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

/// 1353m Maximum Number of Events That Can Be Attended
struct Sol1353 {}

impl Sol1353 {
    /// 1 <= N, Start_i, End_i <= 10^5
    pub fn max_events(mut events: Vec<Vec<i32>>) -> i32 {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let final_day = events.iter().map(|v| v[1]).max().unwrap_or(0);

        events.sort_by_key(|v| v[0]);
        println!("-> {events:?}");

        let mut pq = BinaryHeap::new();

        let mut count = 0;
        let mut p = 0;
        for day in 1..=final_day {
            while p < events.len() && events[p][0] <= day {
                pq.push(Reverse(events[p][1]));
                p += 1;
            }

            while let Some(&Reverse(end)) = pq.peek() {
                if end < day {
                    pq.pop();
                    continue;
                }
                break;
            }

            if let Some(Reverse(_)) = pq.pop() {
                count += 1;
            }
        }

        count
    }
}

/// 2014h Longest Subsequence Repeated K Times
struct Sol2014 {}

impl Sol2014 {
    pub fn longest_subsequence_repeated_k(s: String, k: i32) -> String {
        use std::collections::{HashMap, VecDeque};

        let mut freqs = HashMap::new();
        for chr in s.chars() {
            *freqs.entry(chr).or_insert(0) += 1;
        }
        println!("-> {freqs:?}");

        let mut chrs = vec![];
        for chr in ('a'..='z').rev() {
            if let Some(&f) = freqs.get(&chr)
                && f >= k
            {
                chrs.push(chr);
            }
        }
        println!("-> {chrs:?}");

        let mut queue = VecDeque::new();
        for chr in chrs.iter() {
            queue.push_back(chr.to_string());
        }

        let check = |p: &str| {
            let p: Vec<_> = p.chars().collect();

            let (mut fcount, mut i) = (0, 0);
            for chr in s.chars() {
                if chr == p[i] {
                    i += 1;
                    if i == p.len() {
                        fcount += 1;
                        if fcount == k {
                            return true;
                        }
                        i = 0;
                    }
                }
            }

            false
        };

        let mut lsr = String::new();
        while let Some(cur) = queue.pop_front() {
            println!("-> {cur:?}   {queue:?}");

            if cur.len() > lsr.len() {
                lsr = cur.clone();
            }

            for &chr in chrs.iter() {
                let cur = cur.clone() + &chr.to_string();
                if check(&cur) {
                    queue.push_back(cur);
                }
            }
        }

        lsr
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

/// 2294m Partition Array Such That Maximum Difference Is K
struct Sol2294 {}

impl Sol2294 {
    /// 1 <= N, k <= 10^5
    pub fn partition_array(mut nums: Vec<i32>, k: i32) -> i32 {
        nums.sort_unstable();

        let mut start = nums[0];
        nums[1..].iter().fold(0, |r, &n| {
            if n - start > k {
                start = n;
                r + 1
            } else {
                r
            }
        }) + 1
    }
}

/// 2311m Longest Binary Subsequence Less Than or Equal to K
struct Sol2311 {}

impl Sol2311 {
    pub fn longest_subsequence(s: String, k: i32) -> i32 {
        let mut sval = 0;
        let bits = k.ilog2() as usize + 1;

        s.chars()
            .rev()
            .enumerate()
            .fold(0, |mut longest, (i, chr)| {
                match chr {
                    '1' => {
                        if i < bits && sval + (1 << i) <= k {
                            sval += 1 << i;
                            longest += 1
                        }
                    }
                    _ => longest += 1,
                }

                longest
            })
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
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut buffer: Vec<_> = s.chars().collect();
        let mut pq = BinaryHeap::new();

        for (i, chr) in s.chars().enumerate() {
            println!("-> {pq:?}");
            match chr {
                '*' => {
                    if let Some((_, i)) = pq.pop() {
                        buffer[i] = '*';
                    }
                }
                _ => pq.push((Reverse(chr), i)),
            }
        }

        println!("-> {buffer:?}");

        buffer.iter().filter(|&chr| chr != &'*').collect()
    }
}

/// 3439m Reschedule Meetings for Maximum Free Time I
struct Sol3439 {}

impl Sol3439 {
    pub fn max_free_time(event_time: i32, k: i32, start_time: Vec<i32>, end_time: Vec<i32>) -> i32 {
        let n = start_time.len() & end_time.len();

        let mut len_pfx = vec![0; n + 1];
        for i in 0..n {
            len_pfx[i + 1] = len_pfx[i] + end_time[i] - start_time[i];
        }

        let k = k as usize;
        (k - 1..n).fold(0, |xfree, p| {
            let r = if p == n - 1 {
                event_time
            } else {
                start_time[p + 1]
            };
            let l = if p == k - 1 { 0 } else { end_time[p - k] };

            xfree.max(r - l - (len_pfx[p + 1] - len_pfx[p + 1 - k]))
        }) as _
    }
}

#[cfg(test)]
mod tests;
