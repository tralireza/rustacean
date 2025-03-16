use super::*;

#[test]
fn test_153() {
    assert_eq!(Sol153::find_min(vec![3, 4, 5, 1, 2]), 1);
    assert_eq!(Sol153::find_min(vec![4, 5, 6, 7, 0, 1, 2]), 0);
    assert_eq!(Sol153::find_min(vec![11, 13, 15, 17]), 11);
}

#[test]
fn test_154() {
    assert_eq!(Sol154::find_min(vec![1, 3, 5]), 1);
    assert_eq!(Sol154::find_min(vec![2, 2, 2, 0, 1]), 0);
}

#[test]
fn test_315() {
    for (rst, nums) in [
        (vec![2, 1, 1, 0], vec![5, 2, 6, 1]),
        (vec![0], vec![-1]),
        (vec![0, 0], vec![-1, -1]),
    ] {
        assert_eq!(Sol315::count_smaller(nums.to_vec()), rst);
        assert_eq!(Sol315::bit_count_smaller(nums), rst);
    }
}

#[test]
fn test_704() {
    for (rst, target, nums) in [
        (4, 9, vec![-1, 0, 3, 5, 9, 12]),
        (-1, 2, vec![-1, 0, 3, 5, 9, 12]),
        (0, -1, vec![-1, 0, 3, 5, 9, 12]),
    ] {
        println!(":: {}", rst);
        assert_eq!(Sol704::search(nums, target), rst);
    }
}

#[test]
fn test_2226() {
    for (rst, candies, k) in [(5, vec![5, 8, 6], 3), (0, vec![2, 5], 11)] {
        assert_eq!(Sol2226::maximum_candies(candies, k), rst);
    }
}

#[test]
fn test_2529() {
    for (rst, nums) in [
        (3, vec![-2, -1, -1, 1, 2, 3]),
        (3, vec![-3, -2, -1, 0, 0, 1, 2]),
        (4, vec![5, 20, 66, 1314]),
    ] {
        assert_eq!(Sol2529::maximum_count(nums), rst);
    }
}

#[test]
fn test_2560() {
    for (rst, nums, k) in [(5, vec![2, 3, 5, 9], 2), (2, vec![2, 7, 9, 3, 1], 2)] {
        println!("** {:?}", (&nums, k));
        assert_eq!(Sol2560::min_capability(nums, k), rst);
    }
}

#[test]
fn test_3356() {
    for (rst, nums, queries) in [
        (
            2,
            vec![2, 0, 2],
            vec![vec![0, 2, 1], vec![0, 2, 1], vec![1, 1, 3]],
        ),
        (-1, vec![4, 3, 2, 1], vec![vec![1, 3, 2], vec![0, 2, 1]]),
    ] {
        assert_eq!(Sol3356::min_zero_array(nums, queries), rst);
    }
}
