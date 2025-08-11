use super::*;

#[test]
fn test_868() {
    for (rst, n) in [(2, 22), (0, 8), (2, 5)] {
        assert_eq!(Sol868::binary_gap(n), rst);
    }
}

#[test]
fn test_1863() {
    for (rst, nums) in [
        (6, vec![1, 3]),
        (28, vec![5, 1, 6]),
        (480, vec![3, 4, 5, 6, 7, 8]),
    ] {
        for f in [Sol1863::subset_xor_sum, Sol1863::subset_xor_sum_bitwise] {
            assert_eq!(f(nums.to_vec()), rst);
        }
    }
}

#[test]
fn test_2438() {
    for (rst, n, queries) in [
        (vec![2, 4, 64], 15, vec![vec![0, 1], vec![2, 2], vec![0, 3]]),
        (vec![2], 2, vec![vec![0, 0]]),
    ] {
        println!("* {n} {queries:?}");
        assert_eq!(Sol2438::product_queries(n, queries), rst);
        println!(":: {rst:?}");
    }
}
