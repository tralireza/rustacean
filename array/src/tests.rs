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
fn test_1534() {
    for (rst, arr, a, b, c) in [
        (4, vec![3, 0, 1, 1, 9, 7], 7, 2, 3),
        (0, vec![1, 1, 2, 2, 3], 0, 0, 1),
    ] {
        assert_eq!(Sol1534::count_good_triplets(arr, a, b, c), rst);
    }
}

#[test]
fn test_1550() {
    for (rst, arr) in [
        (false, vec![2, 6, 4, 1]),
        (true, vec![1, 2, 34, 3, 4, 5, 7, 23, 12]),
        (false, vec![1]),
        (true, vec![1, 3, 5]),
        (false, vec![1, 2, 3]),
    ] {
        println!("*** {:?}", arr);
        assert_eq!(Sol1550::three_consecutive_odds(arr), rst);
    }
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
fn test_1920() {
    for (rst, nums) in [
        (vec![0, 1, 2, 4, 5, 3], vec![0, 2, 1, 5, 3, 4]),
        (vec![4, 5, 0, 1, 2, 3], vec![5, 0, 1, 2, 3, 4]),
    ] {
        assert_eq!(Sol1920::build_array(nums), rst);
    }
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
fn test_2033() {
    for (rst, grid, x) in [
        (4, vec![vec![2, 4], vec![6, 8]], 2),
        (5, vec![vec![1, 5], vec![2, 3]], 1),
        (-1, vec![vec![1, 2], vec![3, 4]], 2),
    ] {
        assert_eq!(Sol2033::min_operations(grid, x), rst);
    }
}

#[test]
fn test_2145() {
    for (rst, differences, lower, upper) in [
        (2, vec![1, -3, 4], 1, 6),
        (4, vec![3, -4, 5, 1, -2], -4, 5),
        (0, vec![4, -7, 2], 3, 6),
    ] {
        assert_eq!(Sol2145::number_of_arrays(differences, lower, upper), rst);
    }
}

#[test]
fn test_2176() {
    for (rst, nums, k) in [(4, vec![3, 1, 2, 2, 2, 1, 3], 2), (0, vec![1, 2, 3, 4], 1)] {
        assert_eq!(Sol2176::count_pairs(nums, k), rst);
    }
}

#[test]
fn test_2780() {
    for (rst, nums) in [
        (2, vec![1, 2, 2, 2]),
        (4, vec![2, 1, 3, 1, 1, 1, 7, 1, 2, 1]),
        (-1, vec![3, 3, 3, 3, 7, 2, 2]),
    ] {
        println!(
            "-> [ Boyer-Moore ] majority/dominant: {}",
            Sol2780::Boyer_Moore(nums.to_vec())
        );
        assert_eq!(Sol2780::minimum_index(nums), rst);
    }
}

#[test]
fn test_2873() {
    for (rst, nums) in [
        (77, vec![12, 6, 1, 2, 7]),
        (133, vec![1, 10, 3, 4, 19]),
        (0, vec![1, 2, 3]),
    ] {
        assert_eq!(Sol2873::maximum_triplet_value(nums), rst);
    }
}

#[test]
fn test_2874() {
    for (rst, nums) in [
        (77, vec![12, 6, 1, 2, 7]),
        (133, vec![1, 10, 3, 4, 19]),
        (0, vec![1, 2, 3]),
    ] {
        assert_eq!(Sol2874::maximum_triplet_value(nums), rst);
    }
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

#[test]
fn test_3169() {
    for (rst, days, mut meetings) in [
        (2, 10, vec![vec![5, 7], vec![1, 3], vec![9, 10]]),
        (1, 5, vec![vec![2, 4], vec![1, 3]]),
        (0, 6, vec![vec![1, 6]]),
    ] {
        use std::cmp::Ordering::*;
        meetings.sort_unstable_by(|x, y| match x[0].cmp(&y[0]) {
            Equal => x[1].cmp(&y[1]),
            _ => x[0].cmp(&y[0]),
        });
        println!("-> {:?}", meetings);

        assert_eq!(Sol3169::count_days(days, meetings), rst);
    }
}

#[test]
fn test_3394() {
    for (rst, n, rectangles) in [
        (
            true,
            5,
            vec![
                vec![1, 0, 5, 2],
                vec![0, 2, 2, 4],
                vec![3, 2, 5, 3],
                vec![0, 4, 4, 5],
            ],
        ),
        (
            true,
            4,
            vec![
                vec![0, 0, 1, 1],
                vec![2, 0, 3, 4],
                vec![0, 2, 2, 3],
                vec![3, 0, 4, 3],
            ],
        ),
        (
            false,
            4,
            vec![
                vec![0, 2, 2, 4],
                vec![1, 0, 3, 2],
                vec![2, 2, 3, 4],
                vec![3, 0, 4, 2],
                vec![3, 2, 4, 4],
            ],
        ),
    ] {
        assert_eq!(Sol3394::check_valid_cuts(n, rectangles), rst);
    }
}
