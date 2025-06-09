use super::*;

#[test]
fn test_363() {
    for (rst, matrix, k) in [
        (2, vec![vec![1, 0, 1], vec![0, -2, 3]], 2),
        (3, vec![vec![2, 2, -1]], 3),
    ] {
        assert_eq!(Sol363::max_sum_submatrix(matrix, k), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_1352() {
    let mut o = ProductOfNumbers::new();
    for n in [3, 0, 2, 5, 4] {
        o.add(n);
    }
    for (k, p) in [(2, 20), (3, 40), (4, 0)] {
        assert_eq!(o.get_product(k), p);
    }
    o.add(8);
    assert_eq!(o.get_product(2), 32);
}

#[test]
fn test_2845() {
    for (rst, nums, modulo, k) in [(3, vec![3, 2, 4], 2, 1), (2, vec![3, 1, 9, 6], 3, 0)] {
        assert_eq!(Sol2845::count_interesting_subarrays(nums, modulo, k), rst);
    }
}
