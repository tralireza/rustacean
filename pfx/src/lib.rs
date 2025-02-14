//! # Rust :: Prefix Sum

/// 1352m Product of the Last K Numbers
#[derive(Debug)]
struct ProductOfNumbers {
    length: usize,
    pds: Vec<i32>,
}

impl ProductOfNumbers {
    fn new() -> Self {
        ProductOfNumbers {
            length: 0,
            pds: vec![1],
        }
    }

    fn add(&mut self, num: i32) {
        match num {
            0 => {
                self.pds = vec![1];
                self.length = 0;
            }
            _ => {
                self.pds.push(num * self.pds[self.length]);
                self.length += 1;
            }
        }

        println!("-> {} {:?}", num, self);
    }

    fn get_product(&self, k: i32) -> i32 {
        match self.length < k as usize {
            true => 0,
            _ => self.pds[self.length] / self.pds[self.length - k as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1352() {
        let mut o = ProductOfNumbers::new();
        for n in [3, 0, 2, 5, 4] {
            o.add(n);
        }
        for (k, p) in [(2, 20), (3, 40), (4, 0)] {
            assert_eq!(o.get_product(k), p);
        }
        o.add(8);
        assert_eq!(o.get_product(2), 32);
    }
}
