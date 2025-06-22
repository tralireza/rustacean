//! # String :: Rusty

#![feature(test)]

extern crate test;

/// 38m Count and Say
struct Sol38;

impl Sol38 {
    /// 1 <= n <= 30
    pub fn count_and_say(n: i32) -> String {
        let mut s = "1".to_string();
        for _ in 0..n - 1 {
            let mut enc = vec![];

            let (mut count, mut prv) = (0, '^');
            for chr in s.chars().chain("$".chars()) {
                if chr == prv {
                    count += 1;
                } else {
                    enc.push((count, prv));
                    (count, prv) = (1, chr);
                }
            }

            println!("-> {:?}", enc);

            let mut t = String::new();
            for (count, chr) in enc.iter().skip(1) {
                t += &format!("{}{}", count, chr);
            }

            s = t;
        }

        println!("** {}", s);

        s
    }
}

/// 65h Valid Number
struct Sol65;

impl Sol65 {
    pub fn is_number(s: String) -> bool {
        // State Transition Table :: State/Input -> State
        // ['0-9', '+|-', '.', 'e', ' ']
        let stt = [
            [0, 0, 0, 0, 0], // State: 0 (*Bad* State)
            [3, 2, 4, 0, 1], // State: 1 (Start/End)
            [3, 0, 4, 0, 0], // State: 2
            [3, 0, 5, 6, 9], // State: 3 (End)
            [5, 0, 0, 0, 0], // State: 4
            [5, 0, 0, 6, 9], // State: 5 (End)
            [8, 7, 0, 0, 0], // State: 6
            [8, 0, 0, 0, 0], // State: 7
            [8, 0, 0, 0, 9], // State: 8 (End)
            [0, 0, 0, 0, 9], // State: 9 (End)
        ];

        let mut cstate = 1; // Current State := <- Start State

        for chr in s.chars() {
            cstate = match chr {
                '0'..='9' => stt[cstate][0],
                '+' | '-' => stt[cstate][1],
                '.' => stt[cstate][2],
                'e' | 'E' => stt[cstate][3],
                ' ' => stt[cstate][4],
                _ => return false, // Bad: input
            }
        }

        matches!(cstate, 3 | 5 | 8 | 9)
    }

    fn is_number_enum(s: String) -> bool {
        #[derive(Debug)]
        enum State {
            Start,
            Sign,
            Dot,
            Int,
            IntDot,
            Decimal,
            E,
            ESign,
            Exp,
        }

        use State::*;

        let mut cstate = Start;
        for chr in s.chars() {
            cstate = match (&cstate, chr) {
                (Start, '+' | '-') => Sign,
                (Start | Sign, '.') => Dot,
                (Start | Sign, '0'..='9') => Int,
                (Dot | IntDot, '0'..='9') => Decimal,
                (Int | Decimal | Exp, '0'..='9') => cstate, // No-Transition
                (Int, '.') => IntDot,
                (Int | Decimal | IntDot, 'e' | 'E') => E,
                (E, '+' | '-') => ESign,
                (E | ESign, '0'..='9') => Exp,

                _ => return false,
            }
        }

        matches!(cstate, Int | IntDot | Decimal | Exp)
    }
}

/// 466h Count The Repetition
struct Sol466 {}

impl Sol466 {
    /// 1 <= L1, L2 <= 100
    pub fn get_max_repetitions(s1: String, n1: i32, s2: String, n2: i32) -> i32 {
        let (mut counts, mut marks) = (vec![0; s2.len() + 1], vec![0; s2.len() + 1]);

        let s2: Vec<_> = s2.chars().collect();

        let (mut count, mut mark) = (0, 0);
        for i in 0..n1 as usize {
            for chr in s1.chars() {
                if chr == s2[mark] {
                    mark += 1;
                }
                if mark == s2.len() {
                    mark = 0;
                    count += 1;
                }
            }

            (counts[i], marks[i]) = (count, mark);
            println!("-> {i} {counts:?} {marks:?}");

            for k in 0..i {
                if marks[k] == mark {
                    let prv = counts[k];
                    let pattern = (counts[i] - counts[k]) * ((n1 - 1 - k as i32) / (i - k) as i32);
                    let rest = counts[k + (n1 as usize - 1 - k) % (i - k)] - counts[k];

                    return (prv + pattern + rest) / n2;
                }
            }
        }

        counts[n1 as usize - 1] / n2
    }
}

/// 917 Reverse Only Letters
struct Sol917;

impl Sol917 {
    pub fn reverse_only_letters(s: String) -> String {
        let mut ichr = s.chars().rev().filter(|c| c.is_alphabetic());

        s.chars()
            .flat_map(|c| (!c.is_alphabetic()).then_some(c).or_else(|| ichr.next()))
            .collect()
    }
}

/// 1061m Lexicographically Smallest Equivalent String
struct Sol1061 {}

impl Sol1061 {
    pub fn smallest_equivalent_string(s1: String, s2: String, base_str: String) -> String {
        let mut djset: Vec<_> = (0..26).collect();

        let mut union = |v1, v2| {
            let (v1, v2) = (find(v1, &mut djset), find(v2, &mut djset));
            match v1.cmp(&v2) {
                std::cmp::Ordering::Greater => djset[v1] = v2,
                std::cmp::Ordering::Less => djset[v2] = v1,
                _ => {}
            }
        };

        fn find(v: usize, djset: &mut [usize]) -> usize {
            if djset[v] != v {
                djset[v] = find(djset[v], djset);
            }
            djset[v]
        }

        for (chr1, chr2) in s1.as_bytes().iter().zip(s2.as_bytes().iter()) {
            union((chr1 - b'a') as usize, (chr2 - b'a') as usize);
        }

        println!("-> {djset:?}");

        base_str
            .as_bytes()
            .iter()
            .map(|&chr| (find((chr - b'a') as usize, &mut djset) as u8 + b'a') as char)
            .collect()
    }
}

/// 1154 Days of the Year
struct Sol1154;

impl Sol1154 {
    pub fn day_of_year(date: String) -> i32 {
        let mut days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        let mut vs = vec![];
        for w in date.split('-') {
            vs.push(w.parse::<i32>());
        }

        println!(" -> {:?}", vs);

        let mut dy = 0;
        if let (Ok(year), Ok(month), Ok(day)) = (&vs[0], &vs[1], &vs[2]) {
            dy += day;
            if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                days[1] += 1;
            }

            for m in 0..month - 1 {
                dy += days[m as usize];
            }
        }

        dy
    }
}

/// 1163h Last Substring in Lexicographical Order
struct Sol1163 {}

impl Sol1163 {
    pub fn last_substring(s: String) -> String {
        let s: Vec<_> = s.chars().collect();
        let n = s.len();

        let (mut i, mut j) = (0, 1);
        while j < n {
            let mut k = 0;
            while j + k < n && s[i + k] == s[j + k] {
                k += 1;
            }

            if j + k < n && s[i + k] < s[j + k] {
                (i, j) = (j, (j + 1).max(i + k + 1));
            } else {
                j += k + 1;
            }
        }

        s.iter().skip(i).collect()
    }
}

/// 2138 Divide a String Into Group of Size K
struct Sol2138 {}

impl Sol2138 {
    pub fn divide_string(s: String, k: i32, fill: char) -> Vec<String> {
        let mut divs = vec![];

        let k = k as usize;
        for start in (0..=s.len() - k).step_by(k) {
            divs.push(s[start..start + k].to_string());
        }

        if s.len() % k != 0 {
            let mut last = s[s.len() / k * k..].to_string();
            for _ in 0..k - s.len() % k {
                last.push(fill);
            }
            divs.push(last);
        }

        divs
    }
}

/// 3403h Find the Lexicographically Largest String From the Box I
struct Sol3403 {}

impl Sol3403 {
    pub fn answer_string(word: String, num_friends: i32) -> String {
        if num_friends == 1 {
            return word;
        }

        let n = word.len();
        let mut answer = String::new();
        for i in 0..n {
            let s = &word[i..n.min(i + n - (num_friends as usize - 1))];
            if &answer[..] < s {
                answer = s.to_string();
            }
        }

        answer
    }
}

/// 3442 Maximum Difference Between Even and Odd Frequency I
struct Sol3442 {}

impl Sol3442 {
    pub fn max_difference(s: String) -> i32 {
        use std::collections::HashMap;

        let mut freqs = HashMap::new();
        for chr in s.chars() {
            freqs.entry(chr).and_modify(|f| *f += 1).or_insert(1);
        }

        println!("-> {freqs:?}");

        if let (Some(omax), Some(emin)) = (
            freqs.values().filter(|&f| f & 1 == 1).max(),
            freqs.values().filter(|&f| f & 1 == 0).min(),
        ) {
            return omax - emin;
        }

        0
    }
}

#[cfg(test)]
mod tests;
