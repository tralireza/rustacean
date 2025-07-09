//! # Array

#![feature(test)]

extern crate test;

/// 73m Set Matrix Zeroes
struct Sol73 {}

impl Sol73 {
    /// -2^31 <= M_ij <= 2^31-1
    /// 1 <= M, N <= 200
    pub fn set_zeroes(matrix: &mut [Vec<i32>]) {
        let (rows, cols) = (matrix.len(), matrix[0].len());

        let zero_row0 = matrix[0].contains(&0);
        let zero_col0 = matrix.iter().any(|row| row[0] == 0);

        for r in 1..rows {
            for c in 1..cols {
                if matrix[r][c] == 0 {
                    (matrix[0][c], matrix[r][0]) = (0, 0);
                }
            }
        }

        for r in 1..rows {
            for c in 1..cols {
                if matrix[0][c] == 0 || matrix[r][0] == 0 {
                    matrix[r][c] = 0;
                }
            }
        }

        if zero_col0 {
            for row in matrix.iter_mut() {
                row[0] = 0;
            }
        }
        if zero_row0 {
            for v in matrix[0].iter_mut() {
                *v = 0;
            }
        }
    }
}

/// 747 Largest Number At Least Twice of Others
struct Sol747 {}

impl Sol747 {
    /// 0 <= N_i <= 100
    pub fn dominant_index(nums: Vec<i32>) -> i32 {
        let xv = nums.iter().max().unwrap_or(&-1);

        if nums.iter().filter(|&n| n != xv).all(|n| 2 * n <= *xv) {
            return nums.iter().position(|n| n == xv).unwrap() as i32;
        }

        -1
    }
}

/// 766 Toeplitz Matrix
struct Sol766 {}

impl Sol766 {
    pub fn is_toeplitz_matrix(matrix: Vec<Vec<i32>>) -> bool {
        matrix
            .iter()
            .zip(matrix.iter().skip(1))
            .all(|(tr, br)| tr.iter().zip(br.iter().skip(1)).all(|(lv, rv)| lv == rv))
    }
}

/// 798h Smallest Rotation with Highest Score
struct Sol798 {}

impl Sol798 {
    /// 1 <= L <= 10^5
    /// 0 <= N_i < L
    pub fn best_rotation(nums: Vec<i32>) -> i32 {
        let n = nums.len();

        let mut scores = vec![0; n];
        for (i, &p) in nums.iter().enumerate() {
            let (left, right) = ((n + i - p as usize + 1) % n, (i + 1) % n);
            scores[left] -= 1;
            scores[right] += 1;
            if left > right {
                scores[0] -= 1;
            }
        }

        let (mut cur, mut best) = (0, -(n as i32));
        scores.into_iter().enumerate().fold(0, |x, (i, score)| {
            cur += score;
            if cur > best {
                best = cur;
                i
            } else {
                x
            }
        }) as _
    }
}

/// 1184 Distance Between Bus Stops
struct Sol1184;

impl Sol1184 {
    pub fn distance_between_bus_stops(distance: Vec<i32>, start: i32, destination: i32) -> i32 {
        let (mut src, mut dst) = (start as usize, destination as usize);
        if src > dst {
            (src, dst) = (dst, src);
        }

        distance[src..dst]
            .iter()
            .sum::<i32>()
            .min(distance[dst..].iter().chain(distance[..src].iter()).sum())
    }
}

/// 1394 Find Lucky Integer in an Array
struct Sol1394;

impl Sol1394 {
    /// 1 <= N, N_i <= 500
    pub fn find_lucky(arr: Vec<i32>) -> i32 {
        let mut freqs = [0; 500 + 1];
        for n in arr {
            freqs[n as usize] += 1;
        }

        freqs
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, &f)| f > 0)
            .find(|(n, f)| *f == n)
            .map(|(n, _)| n as i32)
            .unwrap_or(-1)
    }
}

/// 1534 Count Good Triplets
struct Sol1534;

impl Sol1534 {
    /// O(N^3)
    pub fn count_good_triplets(arr: Vec<i32>, a: i32, b: i32, c: i32) -> i32 {
        let mut count = 0;
        for (i, x) in arr.iter().enumerate() {
            for (j, y) in arr.iter().enumerate().skip(i + 1) {
                if (x - y).abs() <= a {
                    for z in arr.iter().skip(j + 1) {
                        if (y - z).abs() <= b && (z - x).abs() <= c {
                            count += 1;
                        }
                    }
                }
            }
        }

        count
    }
}

/// 1550 Three Consecutive Odds
struct Sol1550;

impl Sol1550 {
    pub fn three_consecutive_odds(arr: Vec<i32>) -> bool {
        arr.windows(3)
            .inspect(|v| println!("-> {:?}", v))
            .any(|v| v[0] & v[1] & v[2] & 1 == 1)
    }
}

/// 1752 Check If Array Is Sorted and Rotated
struct Sol1752;

impl Sol1752 {
    pub fn check(nums: Vec<i32>) -> bool {
        println!(" * {:?}", nums);
        println!(
            ":: {}",
            nums.windows(2).fold(
                if nums[0] < nums[nums.len() - 1] { 1 } else { 0 },
                |r, v| if v[1] < v[0] { r + 1 } else { r }
            ) <= 1
        );

        let mut nums = nums;
        let Some(pinv) = nums.windows(2).position(|v| v[1] < v[0]) else {
            return true;
        };

        nums.rotate_left(pinv + 1);
        nums.windows(2).all(|v| v[0] <= v[1])
    }
}

/// 1800 Maximum Possible Ascending Subarray Sum
struct Sol1800;

impl Sol1800 {
    pub fn max_ascending_sum(nums: Vec<i32>) -> i32 {
        nums.windows(2)
            .fold((nums[0], nums[0]), |mut kadan, v| {
                if v[1] > v[0] {
                    kadan.0 += v[1];
                } else {
                    kadan.0 = v[1];
                }
                kadan.1 = kadan.1.max(kadan.0);

                kadan
            })
            .1
    }
}

/// 1920 Build Array from Permutation
struct Sol1920;

impl Sol1920 {
    /// 1 <= N <= 1000, 0 <= N_i < N
    pub fn build_array(nums: Vec<i32>) -> Vec<i32> {
        println!("** {:?}", nums);

        fn in_place(nums: Vec<i32>) -> Vec<i32> {
            let mut nums = nums;
            for i in 0..nums.len() {
                nums[i] += 1000 * (nums[nums[i] as usize] % 1000);
            }
            for n in nums.iter_mut() {
                *n /= 1000;
            }

            nums
        }
        println!(":: {:?}", in_place(nums.to_vec()));

        nums.iter().fold(vec![], |mut v, &n| {
            v.push(nums[n as usize]);
            v
        })
    }
}

/// 2200 Find All K-Distant Indices in an Array
struct Sol2200 {}

impl Sol2200 {
    pub fn find_k_distant_indices(nums: Vec<i32>, key: i32, k: i32) -> Vec<i32> {
        let k = k as usize;
        let mut l = 0;
        nums.iter()
            .enumerate()
            .filter(|(_, &n)| n == key)
            .fold(vec![], |mut kdists, (r, _)| {
                for i in l.max(r.saturating_sub(k))..=(r + k).min(nums.len() - 1) {
                    kdists.push(i as i32);
                }
                l = r + k + 1;

                kdists
            })
    }
}

/// 2016 Maximum Difference Between Increasing Elements
struct Sol2016 {}

impl Sol2016 {
    pub fn maximum_difference(nums: Vec<i32>) -> i32 {
        let mut vmin = nums[0];
        let mut xdiff = -1;

        for n in nums.into_iter().skip(1) {
            if n > vmin {
                xdiff = xdiff.max(n - vmin);
            }
            vmin = vmin.min(n);
        }

        xdiff
    }
}

/// 2017m Grid Game
struct Sol2017;

impl Sol2017 {
    pub fn grid_game(grid: Vec<Vec<i32>>) -> i64 {
        let mut top = grid[0].iter().map(|&n| n as i64).sum::<i64>();
        let mut bottom = 0;

        (0..grid[0].len()).fold(i64::MAX, |mx, i| {
            top -= grid[0][i] as i64;
            bottom += grid[1][i] as i64;
            mx.min(top.max(bottom - grid[1][i] as i64))
        })
    }
}

/// 2033m Minimum Operations to Make a Uni-Value Grid
struct Sol2033;

impl Sol2033 {
    pub fn min_operations(grid: Vec<Vec<i32>>, x: i32) -> i32 {
        let mut nums = vec![];
        for r in 0..grid.len() {
            for c in 0..grid[0].len() {
                nums.push(grid[r][c]);
            }
        }

        nums.sort_unstable();
        let median = nums[nums.len() / 2];

        println!("-> {:?}", (&nums, median));

        let r = median % x;
        let mut ops = 0;
        for n in nums {
            if n % x != r {
                return -1;
            }
            ops += (n - median).abs() / x;
        }

        ops
    }
}

/// 2094 Finding 3-Digit Even Numbers
struct Sol2094;

impl Sol2094 {
    pub fn find_even_numbers(digits: Vec<i32>) -> Vec<i32> {
        let mut freq = [0; 10];
        digits.iter().for_each(|&d| freq[d as usize] += 1);

        println!("-> {:?}", freq);

        let mut evens = vec![];
        for h in 1..=9 {
            if freq[h] == 0 {
                continue;
            }
            freq[h] -= 1;
            for t in 0..=9 {
                if freq[t] == 0 {
                    continue;
                }
                freq[t] -= 1;
                for o in (0..=8).step_by(2) {
                    if freq[o] > 0 {
                        evens.push((100 * h + 10 * t + o) as i32);
                    }
                }
                freq[t] += 1;
            }
            freq[h] += 1;
        }

        evens
    }
}

/// 2099 Find Subsequence of Length K With the Largest Sum
struct Sol2099 {}

impl Sol2099 {
    pub fn max_subsequence(nums: Vec<i32>, k: i32) -> Vec<i32> {
        use std::cmp::Reverse;

        let mut sorted: Vec<_> = nums.iter().enumerate().collect();
        sorted.sort_by_key(|(_, &n)| Reverse(n));
        println!("-> {sorted:?}");

        sorted[0..k as usize].sort();
        println!("-> {sorted:?}");

        sorted[0..k as usize].iter().map(|(_, &n)| n).collect()
    }
}

/// 2145m Count the Hidden Sequences
struct Sol2145;

impl Sol2145 {
    /// 1 <= N <= 10^5, -10^5 <= N_i <= 10^5
    pub fn number_of_arrays(differences: Vec<i32>, lower: i32, upper: i32) -> i32 {
        let (mut x, mut n) = (0, 0);
        let mut v = 0;
        for diff in differences {
            v += diff;
            x = x.max(v);
            n = n.min(v);

            if x - n > upper - lower {
                return 0;
            }
        }

        upper - lower - (x - n) + 1
    }
}

/// 2176 Count Equal and Divisible Pairs in an Array
struct Sol2176;

impl Sol2176 {
    pub fn count_pairs(nums: Vec<i32>, k: i32) -> i32 {
        let mut count = 0;
        for (i, &x) in nums.iter().enumerate().take(nums.len() - 1) {
            for (j, &y) in nums.iter().enumerate().skip(i + 1) {
                if x == y && (i * j).is_multiple_of(k as usize) {
                    count += 1;
                }
            }
        }

        count
    }
}

/// 2780m Minimum Index of a Valid Split
struct Sol2780;

impl Sol2780 {
    pub fn minimum_index(nums: Vec<i32>) -> i32 {
        let mut copy = nums.to_vec();
        copy.sort_unstable();

        let (mut dominent, mut frq) = (0, 0);
        copy.iter().fold((0, 0), |(d, mut f), &n| {
            if d == n {
                f += 1;
            } else {
                f = 1;
            }

            if f > frq {
                frq = f;
                dominent = n;
            }

            (n, f)
        });

        println!("-> {:?}", (dominent, frq));

        let mut f = 0;
        for (i, &n) in nums.iter().enumerate() {
            if n == dominent {
                f += 1;
            }

            if f > i.div_ceil(2) && (frq - f) > (nums.len() - i - 1) / 2 {
                return i as i32;
            }
        }

        -1
    }

    /// Majority Voting Algorithm: Boyer-Moore
    #[expect(non_snake_case)]
    fn Boyer_Moore(nums: Vec<i32>) -> i32 {
        nums.iter()
            .fold((nums[0], 0), |(mut majority, mut frq), &n| {
                if n == majority {
                    frq += 1;
                } else {
                    frq -= 1;
                }

                if frq == 0 {
                    majority = n;
                }

                (majority, frq)
            })
            .0
    }
}

/// 2873 Maximum Value of an Ordered Triplet I
struct Sol2873;

impl Sol2873 {
    pub fn maximum_triplet_value(nums: Vec<i32>) -> i64 {
        let mut value = -1;
        for i in 0..nums.len() {
            for j in i + 1..nums.len() {
                for k in j + 1..nums.len() {
                    value = value.max((nums[i] - nums[j]) as i64 * nums[k] as i64);
                }
            }
        }

        value.max(0)
    }
}

/// 2874m Maximum Value of an Ordered Triplet II
struct Sol2874;

impl Sol2874 {
    pub fn maximum_triplet_value(nums: Vec<i32>) -> i64 {
        let mut value = 0;
        let (mut lmax, mut diff) = (0, 0);
        for n in nums {
            value = value.max(diff as i64 * n as i64);

            diff = diff.max(lmax - n);
            lmax = lmax.max(n);
        }

        value
    }
}

/// 2894 Divisible and Non-divisible Sums Difference
struct Sol2894 {}

impl Sol2894 {
    pub fn difference_of_sums(n: i32, m: i32) -> i32 {
        2 * (1..=n).filter(|&v| v % m != 0).sum::<i32>() - n * (n + 1) / 2
    }
}

/// 2942 Find Words Containing Character
struct Sol2942 {}

impl Sol2942 {
    pub fn find_words_containing(words: Vec<String>, x: char) -> Vec<i32> {
        println!(
            ":: {:?}",
            words
                .iter()
                .enumerate()
                .filter_map(|(i, word)| if word.contains(x) {
                    Some(i as i32)
                } else {
                    None
                })
                .collect::<Vec<_>>()
        );

        words
            .iter()
            .enumerate()
            .filter(|(_, word)| word.contains(x))
            .map(|(i, _)| i as i32)
            .collect()
    }
}

/// 2966m Divide Array Into Arrays With Max Difference
struct Sol2966 {}
impl Sol2966 {
    pub fn divide_array(mut nums: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        nums.chunks(3)
            .map(|chk| {
                if chk[2] - chk[0] > k {
                    None
                } else {
                    Some(chk.to_vec())
                }
            })
            .collect::<Option<Vec<Vec<i32>>>>()
            .unwrap_or_default()
    }
}

/// 3105 Longest Strictly Increasing or Strictly Decreasing Subarray
struct Sol3105;

impl Sol3105 {
    pub fn longest_monotonic_subarray(nums: Vec<i32>) -> i32 {
        let (mut asc, mut desc) = (0, 0);
        nums.windows(2).fold([0, 0], |mut r, v| {
            if v[0] > v[1] {
                r[0] += 1;
                desc = desc.max(r[0]);
            } else {
                r[0] = 0;
            }
            if v[1] > v[0] {
                r[1] += 1;
                asc = asc.max(r[1]);
            } else {
                r[1] = 0;
            }
            println!("{:?}", r);
            r
        });

        println!(":: {:?}", asc.max(desc) + 1);

        match (asc, desc) {
            (0, 0) => 1,
            _ => asc.max(desc) + 1,
        }
    }
}

/// 3151 Special Array I
struct Sol3151;

impl Sol3151 {
    pub fn is_array_special(nums: Vec<i32>) -> bool {
        nums.windows(2)
            .fold(true, |r, v| if (v[0] ^ v[1]) & 1 == 0 { false } else { r })
    }
}

/// 3169m Count Days Without Meetings
struct Sol3169;

impl Sol3169 {
    pub fn count_days(days: i32, meetings: Vec<Vec<i32>>) -> i32 {
        use std::collections::BTreeMap;

        let mut meetings = meetings;
        for meeting in meetings.iter_mut() {
            meeting[1] += 1;
        }

        let mut sweep = BTreeMap::new();
        for meeting in &meetings {
            sweep
                .entry(&meeting[0])
                .and_modify(|f| *f += 1)
                .or_insert(1);
            sweep
                .entry(&meeting[1])
                .and_modify(|f| *f -= 1)
                .or_insert(-1);
        }

        let days = days + 1;
        sweep.entry(&days).or_insert(0);

        println!("-> {:?}", (std::any::type_name_of_val(&sweep), &sweep));

        let (mut cur_meeting, mut cur_day) = (0, 1);
        sweep.iter().fold(0, |mut rst, (&day, &diff)| {
            if cur_meeting == 0 {
                rst += day - cur_day;
            }
            cur_meeting += diff;
            cur_day = *day;

            rst
        })
    }
}

/// 3355m Zero Array Transformation I
struct Sol3355 {}

impl Sol3355 {
    pub fn is_zero_array(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> bool {
        let mut diffs = vec![0; nums.len() + 1];
        for query in queries {
            diffs[query[0] as usize] += 1;
            diffs[query[1] as usize + 1] -= 1;
        }

        let mut diffs_sum = vec![];
        let mut sum = 0;
        for diff in diffs {
            sum += diff;
            diffs_sum.push(sum);
        }

        println!("-> {diffs_sum:?}");

        nums.into_iter()
            .zip(diffs_sum)
            .all(|(num, diff_sum)| num <= diff_sum)
    }
}

/// 3392 Count Subarrays of Length Three With a Condition
struct Sol3392;

impl Sol3392 {
    pub fn count_subarrays(nums: Vec<i32>) -> i32 {
        println!(
            ":: {}",
            nums.to_vec().windows(3).fold(0, |count, v| {
                if 2 * (v[0] + v[2]) == v[1] {
                    count + 1
                } else {
                    count
                }
            })
        );

        let mut count = 0;
        for i in 2..nums.len() {
            if 2 * (nums[i - 2] + nums[i]) == nums[i - 1] {
                count += 1;
            }
        }

        count
    }
}

/// 3394m Check if Grid can be Cut into Sections
struct Sol3394;

impl Sol3394 {
    #[expect(unused_variables)]
    pub fn check_valid_cuts(n: i32, mut rectangles: Vec<Vec<i32>>) -> bool {
        let mut check = |offset| {
            rectangles.sort_unstable_by_key(|v| v[offset]);

            let mut gaps = 0;
            rectangles
                .iter()
                .skip(1)
                .fold(rectangles[0][offset + 2], |end: i32, rectangle| {
                    if end <= rectangle[offset] {
                        gaps += 1;
                    }

                    end.max(rectangle[offset + 2])
                });

            gaps >= 2
        };

        check(0) || check(1)
    }
}

/// 3423 Maximum Difference Between Adjacent Elements in a Circular Array
struct Sol3423 {}

impl Sol3423 {
    /// 2 <= N <= 100
    pub fn max_adjacent_distance(nums: Vec<i32>) -> i32 {
        println!(
            ":: {}",
            nums.iter()
                .zip(nums.iter().cycle().skip(1))
                .map(|(a, b)| (a - b).abs())
                .max()
                .unwrap()
        );

        nums.windows(2)
            .fold((nums[0] - nums[nums.len() - 1]).abs(), |r, w| {
                ((w[0] - w[1]).abs()).max(r)
            })
    }
}

#[cfg(test)]
mod tests;
