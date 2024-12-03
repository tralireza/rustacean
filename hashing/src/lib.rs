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
        assert_eq!(Solution1346::check_if_exist(vec![10, 2, 5, 3]), true);
        assert_eq!(Solution1346::check_if_exist(vec![3, 1, 7, 11]), false);
    }

    #[test]
    fn test_constructs() {
        for (_x, _y) in (0..8).zip(0..8) {}

        let h = [1, 1, 3, 9, 42, 89, 113, 127];
        for n in &h {
            let rst = match n {
                0 => "",
                1..7 => "",
                9 | 10 | 11 => "",
                42 | 113 => "Y",
                _ => "",
            };
            if rst == "Y" {
                println!("{} => {}", rst, n)
            }
        }
    }

    #[test]
    fn test_speed() {
        use std::time::{Duration, Instant};

        let mut count = 0;
        let limit = Duration::new(1, 0);
        let start = Instant::now();

        while (Instant::now() - start) < limit {
            count += 1
        }

        println!(" -> {count}")
    }
}
