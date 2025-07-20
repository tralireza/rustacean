//! # Rust :: Simulation

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
            if hs.contains(o) {
                Some(o << 1)
            } else {
                None
            }
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

#[cfg(test)]
mod tests;
