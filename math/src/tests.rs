use super::*;

#[test]
fn test_29() {
    for (n, d, r) in [(10, 3, 3), (7, -3, -2), (-2147483648, -1, 2147483647)] {
        assert_eq!(Sol29::divide(n, d), r);
    }
}

#[test]
fn test_838() {
    for (rst, dominoes) in [
        ("RR.L".to_string(), "RR.L".to_string()),
        ("LL.RR.LLRRLL..".to_string(), ".L.R...LR..L..".to_string()),
    ] {
        assert_eq!(Sol838::push_dominoes(dominoes), rst);
    }
}

#[test]
fn test_908() {
    assert_eq!(Sol908::smallest_range_i(vec![1], 0), 0);
    assert_eq!(Sol908::smallest_range_i(vec![0, 10], 2), 6);
    assert_eq!(Sol908::smallest_range_i(vec![1, 3, 6], 3), 0);
}

#[test]
fn test_970() {
    for (rst, x, y, bound) in [
        (vec![2, 3, 4, 5, 7, 9, 10], 2, 3, 10),
        (vec![2, 4, 6, 8, 10, 14], 3, 5, 15),
        (vec![2, 3, 5, 9], 2, 1, 10),
    ] {
        use std::collections::HashSet;
        let set: HashSet<i32> = rst.into_iter().collect();
        assert!(Sol970::powerful_integers(x, y, bound)
            .iter()
            .all(|x| set.contains(&x)));
    }
}

#[test]
fn test_989() {
    assert_eq!(
        Sol989::add_to_array_form(vec![1, 2, 0, 0], 34),
        vec![1, 2, 3, 4]
    );
    assert_eq!(Sol989::add_to_array_form(vec![2, 7, 4], 181), vec![4, 5, 5]);
    assert_eq!(
        Sol989::add_to_array_form(vec![2, 1, 5], 806),
        vec![1, 0, 2, 1]
    );
}

#[test]
fn test_1780() {
    for f in [
        Sol1780::check_powers_of_three,
        Sol1780::check_powers_of_three_recursive,
    ] {
        assert_eq!(f(12), true);
        assert_eq!(f(91), true);
        assert_eq!(f(21), false);
    }
}

#[test]
fn test_2338() {
    for (rst, n, max_value) in [(10, 2, 5), (11, 5, 3), (27, 2, 10), (510488787, 184, 389)] {
        assert_eq!(Sol2338::ideal_arrays(n, max_value), rst);
    }
}

#[test]
fn test_2523() {
    for (rst, left, right) in [(vec![11, 13], 10, 19), (vec![-1, -1], 4, 6)] {
        assert_eq!(Sol2523::closest_primes(left, right), rst);
    }
}

#[test]
fn test_2578() {
    for (n, r) in [(1, 1), (2, 5)] {
        assert_eq!(Sol2578::colored_cells(n), r);
    }
}

#[test]
fn test_2818() {
    for (rst, nums, k) in [
        (81, vec![8, 3, 9, 3, 8], 2),
        (4788, vec![19, 12, 14, 6, 10, 18], 3),
        (256720975, vec![3289, 2832, 14858, 22011], 6),
        (630596200, vec![6, 1, 13, 10, 1, 17, 6], 27),
    ] {
        assert_eq!(Sol2818::maximum_score(nums, k), rst);
    }
}

#[test]
fn test_2843() {
    for (rst, low, high) in [(9, 1, 100), (4, 1200, 1230)] {
        assert_eq!(Sol2843::count_symmetric_integers(low, high), rst);
    }
}

#[test]
fn test_3272() {
    for (rst, n, k) in [(27, 3, 5), (2, 1, 4), (2468, 5, 6), (9, 2, 1)] {
        assert_eq!(Sol3272::count_good_integers(n, k), rst)
    }
}
