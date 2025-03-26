//! # Array

/// 1184 Distance Between Bus Stops
struct Sol1184;

impl Sol1184 {
    pub fn distance_between_bus_stops(distance: Vec<i32>, start: i32, destination: i32) -> i32 {
        let (mut src, mut dst) = (start as usize, destination as usize);
        if src > dst {
            (src, dst) = (dst, src);
        }

        distance[src..dst].iter().fold(0, |r, v| r + v).min(
            distance[dst..]
                .iter()
                .chain(distance[..src].iter())
                .fold(0, |r, v| r + v),
        )
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
                    kadan.0 = kadan.0 + v[1];
                } else {
                    kadan.0 = v[1];
                }
                kadan.1 = kadan.1.max(kadan.0);

                kadan
            })
            .1
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

/// 3394m Check if Grid can be Cut into Sections
struct Sol3394;

impl Sol3394 {
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

#[cfg(test)]
mod tests;
