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

        freq.iter().enumerate().fold(0, |count, (n, f)| {
            count + ((f + n as i32) / (n as i32 + 1)) * (n as i32 + 1)
        })
    }
}

#[cfg(test)]
mod tests;
