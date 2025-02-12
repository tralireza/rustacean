//! # Rust :: Stack

/// 1910m Remove All Occurrences of a Substring
struct Sol1910;

impl Sol1910 {
    pub fn remove_occurrences(s: String, part: String) -> String {
        let part = part.chars().collect::<Vec<_>>();

        let mut stack = vec![];
        for chr in s.chars() {
            stack.push(chr);

            let mut count = 0;
            for &tchr in stack.iter().rev().take(part.len()) {
                if part[part.len() - 1 - count] == tchr {
                    count += 1;
                    continue;
                }
                break;
            }

            if count == part.len() {
                stack.truncate(stack.len() - part.len());
            }
        }

        stack.iter().collect()
    }
}

/// 3174 Clear Digits
struct Sol3174;

impl Sol3174 {
    pub fn clear_digits(s: String) -> String {
        let mut stack: Vec<char> = vec![];

        for chr in s.chars() {
            if chr.is_numeric() {
                match stack.last() {
                    Some(tchr) if tchr.is_alphabetic() => {
                        stack.pop();
                        continue;
                    }
                    _ => (),
                }
            }

            stack.push(chr);
        }

        stack.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1910() {
        assert_eq!(
            Sol1910::remove_occurrences("daabcbaabcbc".to_string(), "abc".to_string()),
            "dab".to_string()
        );
        assert_eq!(
            Sol1910::remove_occurrences("axxxxyyyyb".to_string(), "xy".to_string()),
            "ab".to_string()
        );
    }

    #[test]
    fn test_3174() {
        assert_eq!(Sol3174::clear_digits("abc".to_string()), "abc".to_string());
        assert_eq!(Sol3174::clear_digits("cb34".to_string()), "".to_string());
    }
}
