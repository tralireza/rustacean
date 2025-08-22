use super::*;

#[test]
fn test_73() {
    for (rst, mut matrix) in [
        (
            vec![vec![1, 0, 1], vec![0, 0, 0], vec![1, 0, 1]],
            vec![vec![1, 1, 1], vec![1, 0, 1], vec![1, 1, 1]],
        ),
        (
            vec![vec![0, 0, 0, 0], vec![0, 4, 5, 0], vec![0, 3, 1, 0]],
            vec![vec![0, 1, 2, 0], vec![3, 4, 5, 2], vec![1, 3, 1, 5]],
        ),
    ] {
        Sol73::set_zeroes(&mut matrix);
        assert_eq!(rst, matrix);
    }
}

#[test]
fn test_747() {
    for (rst, nums) in [(1, vec![3, 6, 1, 0]), (-1, vec![1, 2, 3, 4])] {
        println!("* {nums:?}");
        assert_eq!(Sol747::dominant_index(nums), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_766() {
    for (rst, matrix) in [
        (
            true,
            vec![vec![1, 2, 3, 4], vec![5, 1, 2, 3], vec![9, 5, 1, 2]],
        ),
        (false, vec![vec![1, 2], vec![2, 2]]),
    ] {
        println!("* {matrix:?}");
        assert_eq!(Sol766::is_toeplitz_matrix(matrix), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_798() {
    for (rst, nums) in [(3, vec![2, 3, 1, 4, 0]), (0, vec![1, 3, 0, 2, 4])] {
        println!("* {nums:?}");
        assert_eq!(Sol798::best_rotation(nums), rst);
        println!(":: {rst:?}");
    }
}

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
fn test_1394() {
    for (rst, arr) in [
        (2, vec![2, 2, 3, 4]),
        (3, vec![1, 2, 2, 3, 3, 3]),
        (-1, vec![2, 2, 2, 3, 3]),
    ] {
        println!("* {arr:?}");
        assert_eq!(Sol1394::find_lucky(arr), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1450() {
    for (rst, start_time, end_time, query_time) in [
        (1, vec![1, 2, 3], vec![3, 2, 7], 4),
        (1, vec![4], vec![4], 4),
    ] {
        println!("* {start_time:?} {end_time:?} {query_time}");
        assert_eq!(Sol1450::busy_student(start_time, end_time, query_time), rst);
        println!(":: {rst:?}");
    }
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

#[bench]
fn bench_1550_windows(b: &mut test::Bencher) {
    let arr = vec![1, 2, 34, 3, 4, 5, 7, 23, 12];
    fn windows(arr: &Vec<i32>) -> bool {
        arr.windows(3).any(|v| v[0] & v[1] & v[2] & 1 == 1)
    }

    b.iter(|| test::black_box(windows(&arr)));
}

#[bench]
fn bench_1550_kadane(b: &mut test::Bencher) {
    let arr = vec![1, 2, 34, 3, 4, 5, 7, 23, 12];
    fn kadane(arr: &Vec<i32>) -> bool {
        let mut counter = 0;
        for &n in arr.iter() {
            if n & 1 == 1 {
                counter += 1;
                if counter == 3 {
                    return true;
                }
            } else {
                counter = 0;
            }
        }
        false
    }

    b.iter(|| test::black_box(kadane(&arr)));
}

#[test]
fn test_1752() {
    assert_eq!(Sol1752::check(vec![3, 4, 5, 1, 2]), true);
    assert_eq!(Sol1752::check(vec![2, 1, 3, 4]), false);
    assert_eq!(Sol1752::check(vec![1, 2, 3]), true);
}

#[test]
fn test_1796() {
    for (rst, s) in [(2, "dfa12321afd"), (-1, "abc1111")] {
        println!("* {s:?}");
        assert_eq!(Sol1796::second_highest(s.to_string()), rst);
        println!(":: {rst:?}");
    }
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
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2016() {
    for (rst, nums) in [
        (4, vec![7, 1, 5, 4]),
        (-1, vec![9, 4, 3, 2]),
        (9, vec![1, 5, 2, 10]),
    ] {
        assert_eq!(Sol2016::maximum_difference(nums), rst);
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
fn test_2094() {
    for (rst, digits) in [
        (
            vec![102, 120, 130, 132, 210, 230, 302, 310, 312, 320],
            vec![2, 1, 3, 0],
        ),
        (vec![222, 228, 282, 288, 822, 828, 882], vec![2, 2, 8, 8, 2]),
        (vec![], vec![3, 7, 5]),
    ] {
        assert_eq!(Sol2094::find_even_numbers(digits), rst);
    }
}

#[test]
fn test_2099() {
    for (rst, nums, k) in [
        (vec![3, 3], vec![2, 1, 3, 3], 2),
        (vec![1, 3, 4], vec![1, -2, 3, 4], 3),
        (vec![3, 4], vec![3, 4, 3, 3], 2),
    ] {
        println!("* {nums:?} {k}");
        assert_eq!(Sol2099::max_subsequence(nums, k), rst);
        println!(":: {rst:?}");
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
fn test_2190() {
    for (rst, nums, key) in [
        (100, vec![1, 100, 200, 1, 100], 1),
        (2, vec![2, 2, 2, 2, 3], 2),
    ] {
        println!("* {nums:?} {key}");
        assert_eq!(Sol2190::most_frequent(nums, key), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2200() {
    for (rst, nums, key, k) in [
        (vec![1, 2, 3, 4, 5, 6], vec![3, 4, 9, 1, 3, 9, 5], 9, 1),
        (vec![0, 1, 2, 3, 4], vec![2, 2, 2, 2, 2], 2, 2),
    ] {
        assert_eq!(Sol2200::find_k_distant_indices(nums, key, k), rst);
    }
}

#[test]
fn test_2210() {
    for (rst, nums) in [(3, vec![2, 4, 1, 1, 6, 5]), (0, vec![6, 6, 5, 5, 4, 1])] {
        println!("* {nums:?}");
        assert_eq!(Sol2210::count_hill_valley(nums), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2248() {
    for (rst, nums) in [(
        vec![3, 4],
        vec![vec![3, 1, 2, 4, 5], vec![1, 2, 3, 4], vec![3, 4, 5, 6]],
    )] {
        println!("* {nums:?}");
        assert_eq!(Sol2248::intersection(nums), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2341() {
    for (rst, nums) in [
        (vec![3, 1], vec![1, 3, 2, 1, 3, 2, 2]),
        (vec![1, 0], vec![1, 1]),
        (vec![0, 1], vec![0]),
        (vec![0, 5], vec![1, 2, 3, 4, 5]), // 18/128
    ] {
        println!("* {nums:?}");
        assert_eq!(Sol2341::number_of_pairs(nums), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2419() {
    for (rst, nums) in [(2, vec![1, 2, 3, 3, 2, 2]), (1, vec![1, 2, 3, 4])] {
        println!("* {nums:?}");
        assert_eq!(Sol2419::longest_subarray(nums), rst);
        println!(":: {rst:?}");
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
fn test_2894() {
    for (rst, n, m) in [(19, 10, 3), (15, 5, 6), (-15, 5, 1)] {
        assert_eq!(Sol2894::difference_of_sums(n, m), rst);
    }
}

#[test]
fn test_2942() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (rst, words, x) in [
        (vec![0, 1], vec![s!("leet"), s!("code")], 'e'),
        (
            vec![0, 2],
            vec![s!("abc"), s!("bcd"), s!("aaaa"), s!("cbc")],
            'a',
        ),
        (
            vec![],
            vec![s!("abc"), s!("bcd"), s!("aaaa"), s!("cbc")],
            'z',
        ),
    ] {
        assert_eq!(Sol2942::find_words_containing(words, x), rst);
    }
}

#[test]
fn test_2966() {
    for (rst, nums, k) in [
        (
            vec![vec![1, 1, 3], vec![3, 4, 5], vec![7, 8, 9]],
            vec![1, 3, 4, 8, 7, 9, 3, 5, 1],
            2,
        ),
        (vec![], vec![2, 4, 2, 2, 5, 2], 2),
        (
            vec![
                vec![2, 2, 12],
                vec![4, 8, 5],
                vec![5, 9, 7],
                vec![7, 8, 5],
                vec![5, 9, 10],
                vec![11, 12, 2],
            ],
            vec![4, 2, 9, 8, 2, 12, 7, 12, 10, 5, 8, 5, 5, 7, 9, 2, 5, 11],
            14,
        ),
    ] {
        println!("* {nums:?} {k}");
        assert_eq!(Sol2966::divide_array(nums, k), rst);
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
fn test_3195() {
    for (rst, grid) in [
        (6, vec![vec![0, 1, 0], vec![1, 0, 1]]),
        (1, vec![vec![1, 0], vec![0, 0]]),
    ] {
        println!("* {grid:?}");
        assert_eq!(Sol3195::minimum_area(grid), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3355() {
    for (rst, nums, queries) in [
        (true, vec![1, 0, 1], vec![vec![0, 2]]),
        (false, vec![4, 3, 2, 1], vec![vec![1, 3], vec![0, 2]]),
    ] {
        assert_eq!(Sol3355::is_zero_array(nums, queries), rst);
    }
}

#[test]
fn test_3392() {
    for (rst, nums) in [(1, vec![1, 2, 1, 4, 1]), (0, vec![1, 1, 1])] {
        assert_eq!(Sol3392::count_subarrays(nums), rst);
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

#[test]
fn test_3423() {
    for (rst, nums) in [(3, vec![1, 2, 4]), (5, vec![-5, -10, -5])] {
        println!("* {nums:?}");
        assert_eq!(Sol3423::max_adjacent_distance(nums), rst);
        println!(":: {rst}");
    }
}
