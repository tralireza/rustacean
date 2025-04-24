use super::*;

#[test]
fn test_1358() {
    for (rst, s) in [
        (10, "abcabc".to_string()),
        (3, "aaacb".to_string()),
        (1, "abc".to_string()),
    ] {
        assert_eq!(Sol1358::number_of_substrings(s), rst);
    }
}

#[test]
fn test_2379() {
    for (rst, blocks, k) in [
        (3, "WBBWWBBWBW".to_string(), 7),
        (0, "WBWBBBW".to_string(), 2),
    ] {
        assert_eq!(Sol2379::minimum_recolors(blocks, k), rst);
    }
}

#[test]
fn test_2401() {
    for (rst, nums) in [(3, vec![1, 3, 8, 48, 10]), (1, vec![3, 1, 5, 11, 13])] {
        assert_eq!(Sol2401::longest_nice_subarray(nums), rst);
    }
}

#[test]
fn test_2537() {
    for (rst, nums, k) in [
        (1, vec![1, 1, 1, 1, 1], 10),
        (4, vec![3, 1, 4, 3, 2, 2, 4], 2),
    ] {
        assert_eq!(Sol2537::count_good(nums, k), rst);
    }
}

#[test]
fn test_2799() {
    for (rst, nums) in [(4, vec![1, 3, 1, 2, 2]), (10, vec![10, 10, 10, 10])] {
        assert_eq!(Sol2799::count_complete_subarrays(nums), rst);
    }
}

#[test]
fn test_3191() {
    for (rst, nums) in [(3, vec![0, 1, 1, 1, 0, 0]), (-1, vec![0, 1, 1, 1])] {
        assert_eq!(Sol3191::min_operations(nums), rst);
    }
}

#[test]
fn test_3208() {
    for (rst, colors, k) in [
        (3, vec![0, 1, 0, 1, 0], 3),
        (2, vec![0, 1, 0, 0, 1, 0, 1], 6),
        (0, vec![1, 1, 0, 1], 4),
    ] {
        assert_eq!(Sol3208::number_of_alternating_groups(colors, k), rst);
    }
}

#[test]
fn test_3306() {
    for (rst, word, k) in [
        (0, "aeioqq".to_string(), 1),
        (1, "aeiou".to_string(), 0),
        (3, "ieaouqqieaouqq".to_string(), 1),
    ] {
        assert_eq!(Sol3306::count_of_substrings(word, k), rst);
    }
}
