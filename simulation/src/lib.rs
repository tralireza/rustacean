//! # Rust :: Simulation

/// 498m Diagonal Traverse
struct Sol498 {}

impl Sol498 {
    /// 1 <= R, C <= 10^4
    pub fn find_diagonal_order(mat: Vec<Vec<i32>>) -> Vec<i32> {
        let (rows, cols) = (mat.len(), mat[0].len());

        let mut dwalk = vec![];
        let mut forward = true;

        for r in 0..rows {
            let mut diag = vec![];
            for (d, row) in mat
                .iter()
                .rev()
                .skip(rows - r - 1)
                .take(rows.min(cols))
                .enumerate()
            {
                diag.push(row[d]);
            }

            if !forward {
                diag.reverse();
            }
            forward = !forward;

            dwalk.extend(diag);
        }

        for c in 1..cols {
            let mut diag = vec![];
            for (d, row) in mat.iter().rev().take(rows.min(cols - c)).enumerate() {
                diag.push(row[c + d]);
            }

            if !forward {
                diag.reverse();
            }
            forward = !forward;

            dwalk.extend(diag);
        }

        dwalk
    }
}

/// 2154 Keep Multiplying Found Values by Two
struct Sol2154 {}

impl Sol2154 {
    pub fn find_final_value(nums: Vec<i32>, original: i32) -> i32 {
        use std::collections::HashSet;
        use std::iter::successors;

        let mut hs = HashSet::new();
        for n in nums {
            hs.insert(n);
        }

        successors(Some(original), |o| {
            if hs.contains(o) { Some(o << 1) } else { None }
        })
        .last()
        .unwrap()
    }
}

/// 2161m Partition Array According to Given Pivot
struct Sol2161;

impl Sol2161 {
    pub fn pivot_array(nums: Vec<i32>, pivot: i32) -> Vec<i32> {
        let mut rst = vec![pivot; nums.len()];

        let (mut left, mut right) = (0, nums.len() - 1);
        for (l, r) in (0..nums.len()).zip((0..nums.len()).rev()) {
            if nums[l] < pivot {
                rst[left] = nums[l];
                left += 1;
            }
            if nums[r] > pivot {
                rst[right] = nums[r];
                right -= 1;
            }
        }

        println!(":: {:?}", rst);

        rst
    }
}

/// 2243 Calculate Digit Sum of a String
struct Sol2243 {}

impl Sol2243 {
    pub fn digit_sum(s: String, k: i32) -> String {
        let k = k as usize;

        println!(":? {:?}", {
            let mut s = s.clone();
            while s.len() > k {
                s = s
                    .as_bytes()
                    .chunks(k)
                    .map(|chars| {
                        chars
                            .iter()
                            .map(|chr| (chr - b'0') as u32)
                            .sum::<u32>()
                            .to_string()
                    })
                    .collect::<Vec<String>>()
                    .join("");
            }
            s
        });

        let mut source: Vec<_> = s.as_bytes().iter().map(|n| (n - b'0') as u16).collect();

        while source.len() > k {
            let mut target = vec![];
            for chunk in source.chunks(k) {
                let mut dsum: u16 = chunk.iter().sum();
                if dsum == 0 {
                    target.push(0);
                } else {
                    let mut digits = vec![];
                    while dsum > 0 {
                        digits.push(dsum % 10);
                        dsum /= 10;
                    }
                    digits.reverse();
                    target.extend_from_slice(&digits);
                }
            }

            source = target;
        }

        String::from_utf8(source.iter().map(|&n| n as u8 + b'0').collect()).unwrap()
    }
}

/// 2303 Calculate Amount Paid in Taxes
struct Sol2303 {}

impl Sol2303 {
    pub fn calculate_tax(brackets: Vec<Vec<i32>>, mut income: i32) -> f64 {
        let mut rates = vec![vec![0, 0]];
        rates.extend_from_slice(&brackets);

        rates
            .windows(2)
            .map(|w| {
                let diff = w[1][0] - w[0][0];
                let tax = diff.min(income) as f64 * w[1][1] as f64 / 100.0;
                income -= diff.min(income);

                tax
            })
            .sum()
    }
}

/// 2402h Meeting Room III
struct Sol2402 {}

impl Sol2402 {
    pub fn most_booked(n: i32, mut meetings: Vec<Vec<i32>>) -> i32 {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut frees = BinaryHeap::new();
        for room in 0..n {
            frees.push((Reverse(room), 0));
        }
        println!("-> {frees:?}");

        let mut ongoings = BinaryHeap::new();

        meetings.sort_unstable();
        for meeting in meetings {
            while let Some(&(Reverse(end), Reverse(room), count)) = ongoings.peek() {
                if end <= meeting[0] as i64 {
                    ongoings.pop();
                    frees.push((Reverse(room), count));
                } else {
                    break;
                }
            }

            if let Some((Reverse(room), count)) = frees.pop() {
                ongoings.push((Reverse(meeting[1] as i64), Reverse(room), count + 1));
            } else if let Some((Reverse(end), Reverse(room), count)) = ongoings.pop() {
                ongoings.push((
                    Reverse(end + (meeting[1] - meeting[0]) as i64),
                    Reverse(room),
                    count + 1,
                ));
            }
        }

        println!("-> O:   {ongoings:?}");
        println!("-> F:   {frees:?}");

        let mut x = vec![];
        for (Reverse(room), count) in frees.drain() {
            x.push((Reverse(count), room));
        }
        for (_, Reverse(room), count) in ongoings.drain() {
            x.push((Reverse(count), room));
        }
        x.sort();

        println!("-> {x:?}");

        x[0].1
    }
}

/// 2460 Apply Operations to an Array
struct Sol2460;

impl Sol2460 {
    pub fn apply_operations(nums: Vec<i32>) -> Vec<i32> {
        let mut nums = nums;
        for i in 0..nums.len() - 1 {
            if nums[i] == nums[i + 1] {
                nums[i] *= 2;
                nums[i + 1] = 0;
            }
        }

        let mut vcopy = nums.to_vec();
        vcopy.sort_by_key(|&v| if v > 0 { 0 } else { 1 });

        println!("-> {:?}", vcopy);

        for z in 0..nums.len() {
            if nums[z] == 0 {
                let mut swap = z;
                while nums[swap] == 0 {
                    swap += 1;
                    if swap >= nums.len() {
                        return nums;
                    }
                }
                (nums[z], nums[swap]) = (nums[swap], nums[z]);
            }
        }

        nums
    }
}

/// 2739 Total Distance Traveled
struct Sol2739 {}

impl Sol2739 {
    pub fn distance_traveled(main_tank: i32, additional_tank: i32) -> i32 {
        let mut distance = 0;

        let (mut fuel, mut reserve) = (main_tank, additional_tank);
        loop {
            if fuel >= 5 {
                distance += 50;
                fuel -= 5;
                if reserve > 0 {
                    reserve -= 1;
                    fuel += 1;
                }
            } else {
                break distance + fuel * 10;
            }
        }
    }
}

/// 3446m Sort Matrix by Diagonals
struct Sol3446 {}

impl Sol3446 {
    /// 1 <= N <= 10
    pub fn sort_matrix(mut grid: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let n = grid.len();

        for diag in 0..n {
            let mut rdiag: Vec<_> = (0..n - diag).map(|d| grid[diag + d][d]).collect();
            rdiag.sort_by(|a, b| b.cmp(a));
            for d in 0..n - diag {
                grid[diag + d][d] = rdiag[d];
            }
        }

        for diag in 1..n {
            let mut rdiag: Vec<_> = (0..n - diag).map(|d| grid[d][diag + d]).collect();
            rdiag.sort();
            for d in 0..n - diag {
                grid[d][diag + d] = rdiag[d];
            }
        }

        grid
    }
}

#[cfg(test)]
mod tests;
