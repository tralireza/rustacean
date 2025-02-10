//! # Rust :: Stack

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
    fn test_3174() {
        assert_eq!(Sol3174::clear_digits("abc".to_string()), "abc".to_string());
        assert_eq!(Sol3174::clear_digits("cb34".to_string()), "".to_string());
    }
}
