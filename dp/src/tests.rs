use super::*;

#[test]
fn test_115() {
    assert_eq!(
        Sol115::num_distinct("rabbbit".to_string(), "rabbit".to_string()),
        3
    );
    assert_eq!(
        Sol115::num_distinct("babgbag".to_string(), "bag".to_string()),
        5
    );
}

#[test]
fn test_132() {
    assert_eq!(Sol132::min_cut("aab".to_string()), 1);
    assert_eq!(Sol132::min_cut("a".to_string()), 0);
    assert_eq!(Sol132::min_cut("ab".to_string()), 1);
}

#[test]
fn test_174() {
    assert_eq!(
        Sol174::calculate_minimum_hp(vec![vec![-2, -3, 3], vec![-5, -10, 1], vec![10, 30, -5]]),
        7
    );
    assert_eq!(Sol174::calculate_minimum_hp(vec![vec![0]]), 1);
}

#[test]
fn test_312() {
    for (rst, nums) in [(167, vec![3, 1, 5, 8]), (10, vec![1, 5])] {
        assert_eq!(Sol312::max_coins(nums), rst);
    }
}

#[test]
fn test_516() {
    assert_eq!(Sol516::longest_palindrome_subseq("bbbab".to_string()), 4);
    assert_eq!(Sol516::longest_palindrome_subseq("cbbd".to_string()), 2);
}

#[test]
fn test_873() {
    assert_eq!(
        Sol873::len_longest_fib_subseq(vec![1, 2, 3, 4, 5, 6, 7, 8]),
        5
    );
    assert_eq!(
        Sol873::len_longest_fib_subseq(vec![1, 3, 7, 11, 12, 14, 18]),
        3
    );
}

#[test]
fn test_1092() {
    for f in [
        Sol1092::shortest_common_supersequence,
        Sol1092::scs_recursive,
    ] {
        assert_eq!(
            f("abac".to_string(), "cab".to_string()),
            "cabac".to_string()
        );
        assert_eq!(
            f("aaaaaaaa".to_string(), "aaaaaaaa".to_string()),
            "aaaaaaaa".to_string()
        );
        println!("--");
    }
}

#[test]
fn test_1524() {
    for f in [Sol1524::num_of_subarrays, Sol1524::num_of_subarrays_psum] {
        assert_eq!(f(vec![1, 3, 5]), 4);
        assert_eq!(f(vec![2, 4, 6]), 0);
        assert_eq!(f(vec![1, 2, 3, 4, 5, 6, 7]), 16);
        println!("--");
    }
}

#[test]
fn test_1749() {
    assert_eq!(Sol1749::max_absolute_sum(vec![1, -3, 2, 3, -4]), 5);
    assert_eq!(Sol1749::max_absolute_sum(vec![2, -5, 1, -4, 3, -2]), 8);
}

#[test]
fn test_2140() {
    for (rst, questions) in [
        (5, vec![vec![3, 2], vec![4, 3], vec![4, 4], vec![2, 5]]),
        (
            7,
            vec![vec![1, 1], vec![2, 2], vec![3, 3], vec![4, 4], vec![5, 5]],
        ),
    ] {
        assert_eq!(Sol2140::most_points(questions), rst);
    }
}

#[test]
fn test_2836() {
    // 1 <= Receiver.Length <= 10^5,  1 <= k <= 10^10
    assert_eq!(Sol2836::get_max_function_value(vec![2, 0, 1], 4), 6);
    assert_eq!(Sol2836::get_max_function_value(vec![1, 1, 1, 2, 3], 3), 10);
}
