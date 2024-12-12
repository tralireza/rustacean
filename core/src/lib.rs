use num::complex::Complex;

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

struct Mandelbrot {}

impl Mandelbrot {
    pub fn calculate(
        iters: usize,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        width: usize,
        height: usize,
    ) -> Vec<Vec<usize>> {
        let mut rows: Vec<_> = Vec::with_capacity(height);
        for y in 0..height {
            let mut row: Vec<usize> = Vec::with_capacity(width);
            for x in 0..width {
                let xp = x as f64 / width as f64;
                let yp = y as f64 / height as f64;
                let cx = xmin + (xmax - xmin) * xp;
                let cy = ymin + (ymax - ymin) * yp;
                row.push(Mandelbrot::mpoint(cx, cy, iters));
            }
            rows.push(row);
        }

        rows
    }

    fn mpoint(cx: f64, cy: f64, iters: usize) -> usize {
        let mut z = Complex { re: 0.0, im: 0.0 };
        let c = Complex::new(cx, cy);
        for i in 0..iters {
            if z.norm() > 2.0 {
                return i;
            }
            z = z * z + c;
        }

        iters
    }

    pub fn render(mset: Vec<Vec<usize>>) {
        for row in mset {
            let mut l = String::with_capacity(row.len());
            for r in row {
                let chr = match r {
                    0..=2 => ' ',
                    3..=5 => '.',
                    6..=10 => '•',
                    11..=30 => '+',
                    31..=100 => '*',
                    101..=200 => 'x',
                    201..=400 => '$',
                    401..=700 => '#',
                    _ => '%',
                };
                l.push(chr);
            }
            println!("-> {}", l);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution2558() {
        assert_eq!(Solution2558::pick_gifts(vec![25, 64, 9, 4, 100], 4), 29);
        println!("--");
        assert_eq!(Solution2558::pick_gifts(vec![1, 1, 1, 1], 4), 4);
    }

    #[test]
    fn test_mandelbrot() {
        let mset = Mandelbrot::calculate(1000, -2.0, 1.0, -1.0, 1.0, 100, 24);
        Mandelbrot::render(mset);
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

    #[derive(Debug)]
    struct MockFile;

    trait Read {
        fn read(self: &Self, bfr: &mut Vec<u8>) -> Result<usize, String>;
    }

    impl Read for MockFile {
        fn read(self: &MockFile, bfr: &mut Vec<u8>) -> Result<usize, String> {
            bfr.push(1);
            Ok(1)
        }
    }

    #[test]
    fn test_mockfile() {
        let f = MockFile{};
        let mut bfr = vec!();

        match f.read(&mut bfr) {
            Ok(bytes) => println!("{:?} -> {}", f, bytes),
            Err(_) => (),
        }
    }
}
