//! # Rust :: Prefix Sum

/// 363h Max Sum of Rectangle No Larger Than K
struct Sol363 {}

impl Sol363 {
    pub fn max_sum_submatrix(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let mut pfxsum = vec![vec![0; matrix[0].len() + 1]; matrix.len()];
        let mut xsum = i32::MIN;

        for r in 0..matrix.len() {
            println!("-> {pfxsum:?}");

            for c in 0..matrix[r].len() {
                pfxsum[r][c + 1] = pfxsum[r][c] + matrix[r][c];
                xsum = xsum.max(pfxsum[r][c + 1]).min(k);
            }
        }

        xsum
    }
}

/// 1352m Product of the Last K Numbers
#[derive(Debug)]
struct ProductOfNumbers {
    pds: Vec<i32>,
}

impl ProductOfNumbers {
    fn new() -> Self {
        ProductOfNumbers { pds: vec![1] }
    }

    fn add(&mut self, num: i32) {
        match num {
            0 => {
                self.pds = vec![1];
            }
            _ => {
                self.pds.push(num * self.pds[self.pds.len() - 1]);
            }
        }

        println!("-> {} {:?}", num, self);
    }

    fn get_product(&self, k: i32) -> i32 {
        match self.pds.len() <= k as usize {
            true => 0,
            _ => self.pds[self.pds.len() - 1] / self.pds[self.pds.len() - 1 - k as usize],
        }
    }
}

/// 2845m Count of Interesting Subarrays
struct Sol2845;

impl Sol2845 {
    pub fn count_interesting_subarrays(nums: Vec<i32>, modulo: i32, k: i32) -> i64 {
        use std::collections::HashMap;

        let mut counts = HashMap::new();
        counts.entry(0).or_insert(1);

        let mut psum = 0;
        nums.iter().fold(0, |mut count, n| {
            if n % modulo == k {
                psum += 1;
            }

            match counts.get(&((psum - k + modulo) % modulo)) {
                Some(v) => count += v,
                _ => (),
            }

            counts
                .entry(psum % modulo)
                .and_modify(|f| *f += 1)
                .or_insert(1);

            println!("-> {:?}", counts);

            count
        })
    }
}

#[cfg(test)]
mod tests;
