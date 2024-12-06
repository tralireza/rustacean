use num::complex::Complex;

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
}
