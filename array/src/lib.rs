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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1184() {
        assert_eq!(
            Sol1184::distance_between_bus_stops(vec![1, 2, 3, 4], 0, 1),
            1
        );
        assert_eq!(
            Sol1184::distance_between_bus_stops(vec![1, 2, 3, 4], 0, 2),
            3
        );
        assert_eq!(
            Sol1184::distance_between_bus_stops(vec![1, 2, 3, 4], 0, 3),
            4
        );
    }

    #[test]
    fn test_1752() {
        assert_eq!(Sol1752::check(vec![3, 4, 5, 1, 2]), true);
        assert_eq!(Sol1752::check(vec![2, 1, 3, 4]), false);
        assert_eq!(Sol1752::check(vec![1, 2, 3]), true);
    }

    #[test]
    fn test_1800() {
        assert_eq!(Sol1800::max_ascending_sum(vec![10, 20, 30, 5, 10, 50]), 65);
        assert_eq!(Sol1800::max_ascending_sum(vec![10, 20, 30, 40, 50]), 150);
        assert_eq!(
            Sol1800::max_ascending_sum(vec![12, 17, 15, 13, 10, 11, 12]),
            33
        );
    }

    #[test]
    fn test_2017() {
        assert_eq!(Sol2017::grid_game(vec![vec![2, 5, 4], vec![1, 5, 1]]), 4);
        assert_eq!(Sol2017::grid_game(vec![vec![3, 3, 1], vec![8, 5, 2]]), 4);
        assert_eq!(
            Sol2017::grid_game(vec![vec![1, 3, 1, 15], vec![1, 3, 3, 1]]),
            7
        );
    }

    #[test]
    fn test_3105() {
        assert_eq!(Sol3105::longest_monotonic_subarray(vec![1, 4, 3, 3, 2]), 2);
        assert_eq!(Sol3105::longest_monotonic_subarray(vec![3, 3, 3, 3]), 1);
        assert_eq!(Sol3105::longest_monotonic_subarray(vec![3, 2, 1]), 3);
    }

    #[test]
    fn test_3151() {
        assert_eq!(Sol3151::is_array_special(vec![1]), true);
        assert_eq!(Sol3151::is_array_special(vec![2, 1, 4]), true);
        assert_eq!(Sol3151::is_array_special(vec![4, 3, 1, 6]), false);
    }
}
