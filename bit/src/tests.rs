use super::*;

#[test]
fn test_493() {
    for (rst, nums) in [
        (2, vec![1, 3, 2, 3, 1]),
        (3, vec![2, 4, 3, 5, 1]),
        (
            0,
            vec![
                2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647,
            ],
        ), // 136/140
    ] {
        println!("* {nums:?}");
        assert_eq!(Sol493::reverse_pairs(nums), rst);
    }
}

#[test]
fn test_218() {
    for (rst, buildings) in [
        (
            vec![
                vec![2, 10],
                vec![3, 15],
                vec![7, 12],
                vec![12, 0],
                vec![15, 10],
                vec![20, 8],
                vec![24, 0],
            ],
            vec![
                vec![2, 9, 10],
                vec![3, 7, 15],
                vec![5, 12, 12],
                vec![15, 20, 10],
                vec![19, 24, 8],
            ],
        ),
        (
            vec![vec![0, 3], vec![5, 0]],
            vec![vec![0, 2, 3], vec![2, 5, 3]],
        ),
        (
            vec![vec![2, 10], vec![9, 0], vec![12, 10], vec![15, 0]],
            vec![vec![2, 9, 10], vec![12, 15, 10]],
        ),
    ] {
        assert_eq!(Sol218::get_skyline(buildings), rst);
    }
}

#[test]
fn test_315() {
    for (rst, nums) in [
        (vec![2, 1, 1, 0], vec![5, 2, 6, 1]),
        (vec![0], vec![-1]),
        (vec![0, 0], vec![-1, -1]),
    ] {
        assert_eq!(Sol315::count_smaller(nums), rst);
    }
}

#[test]
fn test_2179() {
    for (rst, nums1, nums2) in [
        (1, vec![2, 0, 1, 3], vec![0, 1, 2, 3]),
        (4, vec![4, 0, 1, 3, 2], vec![4, 1, 0, 2, 3]),
    ] {
        assert_eq!(Sol2179::good_triplets(nums1, nums2), rst);
    }
}
