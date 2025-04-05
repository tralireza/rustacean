use super::*;

#[test]
fn test_218() {
    assert_eq!(
        Sol218::get_skyline(vec![
            vec![2, 9, 10],
            vec![3, 7, 15],
            vec![5, 12, 12],
            vec![15, 20, 10],
            vec![19, 24, 8]
        ]),
        vec![
            vec![2, 10],
            vec![3, 15],
            vec![7, 12],
            vec![12, 0],
            vec![15, 10],
            vec![20, 8],
            vec![24, 0]
        ]
    );
    assert_eq!(
        Sol218::get_skyline(vec![vec![0, 2, 3], vec![2, 5, 3]]),
        vec![vec![0, 3], vec![5, 0]]
    );

    assert_eq!(
        Sol218::get_skyline(vec![vec![2, 9, 10], vec![12, 15, 10]]),
        vec![vec![2, 10], vec![9, 0], vec![12, 10], vec![15, 0]]
    );
}

#[test]
fn test_1863() {
    for (rst, nums) in [
        (6, vec![1, 3]),
        (28, vec![5, 1, 6]),
        (480, vec![3, 4, 5, 6, 7, 8]),
    ] {
        assert_eq!(Sol1863::subset_xor_sum(nums), rst);
    }
}
