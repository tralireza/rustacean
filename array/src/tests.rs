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
