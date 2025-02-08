//! # String :: Rusty

/// 65h Valid Number
struct Sol65;

impl Sol65 {
    pub fn is_number(s: String) -> bool {
        // ['0-9', '+|-', '.', 'e', ' ']
        let states = [
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

        let mut cur = 1;
        for chr in s.chars() {
            cur = match chr {
                '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' => states[cur][0],
                '+' | '-' => states[cur][1],
                '.' => states[cur][2],
                'e' | 'E' => states[cur][3],
                ' ' => states[cur][4],
                _ => return false, // Bad: input
            }
        }

        match cur {
            3 | 5 | 8 | 9 => true,
            _ => false,
        }
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
mod tests {
    use super::*;

    #[test]
    fn test_65() {
        assert!(Sol65::is_number("0".to_string()));
        assert!(!Sol65::is_number("e".to_string()));
        assert!(!Sol65::is_number(".".to_string()));

        assert!(Sol65::is_number("2e0".to_string()));
    }

    #[test]
    fn test_917() {
        assert_eq!(
            Sol917::reverse_only_letters("a-bC-dEf-ghIj".to_string()),
            "j-Ih-gfE-dCba".to_string()
        );
    }

    #[test]
    fn test_1154() {
        assert_eq!(Sol1154::day_of_year("2019-01-09".to_string()), 9);
        assert_eq!(Sol1154::day_of_year("2019-02-10".to_string()), 41);
    }
}
