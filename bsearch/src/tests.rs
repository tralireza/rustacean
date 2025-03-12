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
fn test_2529() {
    for (rst, nums) in [
        (3, vec![-2, -1, -1, 1, 2, 3]),
        (3, vec![-3, -2, -1, 0, 0, 1, 2]),
        (4, vec![5, 20, 66, 1314]),
    ] {
        assert_eq!(Sol2529::maximum_count(nums), rst);
    }
}
