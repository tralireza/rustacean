//! # String :: Rusty

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
        let mut days = vec![31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        let mut vs = vec![];
        for w in date.split('-') {
            vs.push(w.parse::<i32>());
        }

        println!(" -> {:?}", vs);

        let mut dy = 0;
        match (&vs[0], &vs[1], &vs[2]) {
            (Ok(year), Ok(month), Ok(day)) => {
                dy += day;
                if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                    days[1] += 1;
                }

                for m in 0..month - 1 {
                    dy += days[m as usize];
                }
            }
            _ => (),
        }

        dy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
