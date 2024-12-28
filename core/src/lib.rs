//! Rust Core

// 2558 Take Gifts From the Richest Pile
struct Solution2558 {}

impl Solution2558 {
    pub fn pick_gifts(gifts: Vec<i32>, k: i32) -> i64 {
        use std::collections::BinaryHeap;

        let mut pq = BinaryHeap::new();
        for n in gifts {
            pq.push(n);
        }

        for _ in 0..k {
            if let Some(g) = pq.pop() {
                pq.push(f64::sqrt(g as f64) as i32);
            }
        }

        let mut gtotal = 0i64;
        for g in pq {
            gtotal += g as i64;
        }

        gtotal
    }
}

/// 342 Power of Four
struct Solution342;

impl Solution342 {
    pub fn is_power_of_four(n: i32) -> bool {
        if n == 0 {
            return false;
        }

        let mut n = n;
        while n % 4 == 0 {
            n /= 4;
        }

        n == 1
    }
}

/// 3396 Minimum Number of Operations to Make Elements in Array Distinct
struct Solution3396;

impl Solution3396 {
    pub fn minimum_operations(nums: Vec<i32>) -> i32 {
        let mut freq = vec![0; 101];
        nums.iter().for_each(|n| freq[*n as usize] += 1);

        let mut ops = 0;
        let mut i = 0;

        loop {
            println!(" -> {:?}", freq);

            let mut dups = false;
            for f in &freq {
                if f > &1 {
                    dups = true;
                    break;
                }
            }
            if !dups {
                break;
            }

            ops += 1;
            (0..3).for_each(|_| {
                if i < nums.len() {
                    freq[nums[i] as usize] -= 1;
                    i += 1;
                }
            });
        }

        ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution342() {
        assert!(Solution342::is_power_of_four(16));
        assert!(!Solution342::is_power_of_four(5));
        assert!(Solution342::is_power_of_four(1));
    }

    #[test]
    fn test_solution2558() {
        assert_eq!(Solution2558::pick_gifts(vec![25, 64, 9, 4, 100], 4), 29);
        println!("--");
        assert_eq!(Solution2558::pick_gifts(vec![1, 1, 1, 1], 4), 4);
    }

    #[test]
    fn test_solution3396() {
        assert_eq!(
            Solution3396::minimum_operations(vec![1, 2, 3, 4, 2, 3, 3, 5, 7]),
            2
        );
        println!();
        assert_eq!(Solution3396::minimum_operations(vec![4, 5, 6, 4, 4]), 2);
        println!();
        assert_eq!(Solution3396::minimum_operations(vec![6, 7, 8, 9]), 0);
    }

    #[test]
    fn test_hexdump() {
        use std::io::Read;

        const BYTES_PER_LINE: usize = 16;

        let mut f = std::fs::File::open("./src/lib.rs").expect("can't open file");

        let mut pos = 0;
        let mut bfr = [0; BYTES_PER_LINE];
        let mut pline = String::with_capacity(BYTES_PER_LINE);

        while let Ok(_) = f.read_exact(&mut bfr) {
            print!("0x{:08x}\t", pos);

            for byte in &bfr {
                match byte {
                    0x00 => print!(".  "),
                    0xff => print!("## "),
                    _ => print!("{:02x} ", byte),
                }

                match byte {
                    0x20..=0x7e => pline.push(*byte as char),
                    _ => pline.push('.' as char),
                }
            }

            println!("\t{}", pline);
            pline.clear();

            pos += BYTES_PER_LINE;
        }
    }

    #[test]
    fn test_io() {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        match File::open("src/lib.rs") {
            Ok(f) => {
                let rdr = BufReader::new(f);
                for line in rdr.lines() {
                    match line {
                        Ok(line) => println!(" -> {}", line),
                        Err(_) => (),
                    }
                }
            }
            Err(e) => println!(" -> Err: {}", e),
        }
    }

    #[derive(Debug, PartialEq)]
    enum MFileState {
        Open,
        Closed,
    }

    use std::fmt;
    use std::fmt::Display;

    impl Display for MFileState {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match *self {
                MFileState::Open => write!(f, "OPEN"),
                MFileState::Closed => write!(f, "CLOSED"),
            }
        }
    }

    #[derive(Debug)]
    struct MockFile {
        name: String,
        state: MFileState,
    }

    trait Read {
        fn read(self: &Self, bfr: &mut Vec<u8>) -> Result<usize, String>;
    }

    impl Read for MockFile {
        fn read(self: &MockFile, bfr: &mut Vec<u8>) -> Result<usize, String> {
            bfr.push(1);
            Ok(1)
        }
    }

    impl Display for MockFile {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "<{} ({})>", self.name, self.state)
        }
    }

    impl MockFile {
        fn new(name: &str) -> MockFile {
            MockFile {
                name: String::from(name),
                state: MFileState::Closed,
            }
        }
    }

    #[test]
    fn test_mockfile() {
        let f = MockFile::new("file.mock");
        let mut bfr = vec![];

        match f.read(&mut bfr) {
            Ok(bytes) => println!("{} -> {}", f, bytes),
            Err(_) => (),
        }

        println!("-> {:?}", f);
        println!("-> {}", f);
    }
}
