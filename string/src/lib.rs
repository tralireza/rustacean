//! # String :: Rusty

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

#[cfg(test)]
mod tests;
