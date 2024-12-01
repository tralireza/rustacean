struct Solution1346 {}

impl Solution1346 {
    // 1346 Check If N and Its Double Exist
    pub fn check_if_exist(arr: Vec<i32>) -> bool {
        let mut mvs = std::collections::HashMap::new();

        for n in arr {
            match mvs.get(&n) {
                Some(&f) => mvs.insert(n, f + 1),
                _ => mvs.insert(n, 1),
            };
        }

        for (&n, &f) in mvs.iter() {
            println!("{n} -> {f}");

            if n == 0 {
                if f > 1 {
                    return true;
                }
                continue;
            }
            if let Some(_) = mvs.get(&(2 * n)) {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1346() {
        assert_eq!(Solution1346::check_if_exist(vec![10, 2, 5, 3]), true)
    }
}
