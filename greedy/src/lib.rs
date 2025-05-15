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

/// 2900 Longest Unequal Adjacent Groups Subsequence I
struct Sol2900;

impl Sol2900 {
    /// 1 <= |words, groups| <= 100
    pub fn get_longest_subsequence(mut words: Vec<String>, groups: Vec<i32>) -> Vec<String> {
        words.reverse();
        groups
            .iter()
            .skip(1)
            .fold(
                (vec![words.pop().unwrap()], groups[0]),
                |(mut ls, cur_group), &g| {
                    if cur_group == g {
                        words.pop();
                        (ls, g)
                    } else {
                        ls.push(words.pop().unwrap());
                        (ls, g)
                    }
                },
            )
            .0
    }
}

/// 2918m Minimum Equal Sum of Two Arrays After Replacing Zeros
struct Sol2918;

impl Sol2918 {
    pub fn min_sum(nums1: Vec<i32>, nums2: Vec<i32>) -> i64 {
        let folder = |(sum, zeros), n| match n == 0 {
            true => (sum + 1, zeros + 1),
            _ => (sum + n as i64, zeros),
        };

        let (sum1, zeros1) = nums1.into_iter().fold((0, 0), folder);
        let (sum2, zeros2) = nums2.into_iter().fold((0, 0), folder);

        if sum1 > sum2 && zeros2 == 0 || sum2 > sum1 && zeros1 == 0 {
            return -1;
        }
        sum1.max(sum2)
    }
}

#[cfg(test)]
mod tests;
