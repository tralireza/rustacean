use super::*;

#[test]
fn test_407() {
    for (rst, height_map) in [
        (
            4,
            vec![
                vec![1, 4, 3, 1, 3, 2],
                vec![3, 2, 1, 3, 2, 4],
                vec![2, 3, 3, 2, 3, 1],
            ],
        ),
        (
            10,
            vec![
                vec![3, 3, 3, 3, 3],
                vec![3, 2, 2, 2, 3],
                vec![3, 2, 1, 2, 3],
                vec![3, 2, 2, 2, 3],
                vec![3, 3, 3, 3, 3],
            ],
        ),
    ] {
        assert_eq!(Sol407::trap_rain_water(height_map), rst);
    }
}

#[test]
fn test_1046() {
    for (rst, stones) in [(1, vec![2, 7, 4, 1, 8, 1]), (1, vec![1])] {
        println!("* {stones:?}");
        assert_eq!(Sol1046::last_stone_weight(stones), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3066() {
    for (rst, nums, k) in [
        (2, vec![2, 11, 10, 1, 3], 10),
        (4, vec![1, 1, 2, 4, 9], 20),
        (2, vec![999999999, 999999999, 999999999], 1000000000),
    ] {
        assert_eq!(Sol3066::min_operations(nums, k), rst);
    }
}

#[test]
fn test_3362() {
    for (rst, nums, queries) in [
        (1, vec![2, 0, 2], vec![vec![0, 2], vec![0, 2], vec![1, 1]]),
        (
            2,
            vec![1, 1, 1, 1],
            vec![vec![1, 3], vec![0, 2], vec![1, 3], vec![1, 2]],
        ),
        (-1, vec![1, 2, 3, 4], vec![vec![0, 3]]),
        (
            4,
            vec![1, 2],
            vec![
                vec![1, 1],
                vec![0, 0],
                vec![1, 1],
                vec![1, 1],
                vec![0, 1],
                vec![0, 0],
            ],
        ),
        (
            2,
            vec![0, 0, 1, 1, 0],
            vec![vec![3, 4], vec![0, 2], vec![2, 3]],
        ),
        (
            -1,
            vec![0, 0, 3],
            vec![vec![0, 2], vec![1, 1], vec![0, 0], vec![0, 0]],
        ),
    ] {
        println!("* {nums:?}");
        assert_eq!(Sol3362::max_removal(nums, queries), rst);
        println!(":: {rst}");
    }
}
