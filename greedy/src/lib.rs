//! # Rust Greedy

/// 781m Rabbits in Forest
struct Sol781;

impl Sol781 {
    /// 1 <= N <= 1000, 0 <= N_i < 10000
    pub fn num_rabbits(answers: Vec<i32>) -> i32 {
        let freq = answers.iter().fold([0; 1000], |mut freq, &f| {
            freq[f as usize] += 1;
            freq
        });

        freq.iter()
            .enumerate()
            .map(|(n, f)| (n as i32 + 1, f))
            .fold(0, |count, (n, f)| count + (f + n - 1) / n * n)
    }
}

/// 1007m Minimum Domino Rotations For Equal Row
struct Sol1007;

impl Sol1007 {
    pub fn min_domino_rotations(tops: Vec<i32>, bottoms: Vec<i32>) -> i32 {
        let mut r = i32::MAX;

        'LOOP: for n in 1..=6 {
            let mut top = 0;
            let mut bottom = 0;

            for (&t, &b) in tops.iter().zip(bottoms.iter()) {
                if t != n && b != n {
                    continue 'LOOP;
                }

                if t != n {
                    top += 1;
                }
                if b != n {
                    bottom += 1;
                }
            }

            r = r.min(top.min(bottom));
        }

        if r < i32::MAX {
            return r;
        }
        -1
    }
}

#[cfg(test)]
mod tests;
