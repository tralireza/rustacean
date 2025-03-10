use super::*;

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
