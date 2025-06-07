use super::*;

#[test]
fn test_135() {
    for (rst, ratings) in [(5, vec![1, 0, 2]), (4, vec![1, 2, 2])] {
        println!("* {ratings:?}");
        assert_eq!(Sol135::candy(ratings), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_781() {
    for (rst, answers) in [(5, vec![1, 1, 2]), (11, vec![10, 10, 10])] {
        assert_eq!(Sol781::num_rabbits(answers), rst);
    }
}

#[test]
fn test_1007() {
    for (rst, tops, bottoms) in [
        (2, vec![2, 1, 2, 4, 2, 2], vec![5, 2, 6, 2, 3, 2]),
        (-1, vec![3, 5, 1, 2, 3], vec![3, 6, 3, 3, 4]),
    ] {
        assert_eq!(Sol1007::min_domino_rotations(tops, bottoms), rst);
    }
}

#[test]
fn test_2131() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (rst, words) in [
        (6, vec![s!("lc"), s!("cl"), s!("gg")]),
        (
            8,
            vec![s!("ab"), s!("ty"), s!("yt"), s!("lc"), s!("cl"), s!("ab")],
        ),
        (2, vec![s!("cc"), s!("ll"), s!("xx")]),
    ] {
        println!("* {words:?}");
        assert_eq!(Sol2131::longest_palindrome(words), rst);
    }
}

#[test]
fn test_2434() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (rst, s) in [
        (s!("azz"), s!("zza")),
        (s!("abc"), s!("bac")),
        (s!("addb"), s!("bdda")),
        (s!("eekstrlpmomwzqummz"), s!("mmuqezwmomeplrtskz")),
    ] {
        println!("* {s}");
        assert_eq!(Sol2434::robot_with_string(s), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_2900() {
    for (rst, words, groups) in [
        (
            vec!["e".to_string(), "b".to_string()],
            vec!["e".to_string(), "a".to_string(), "b".to_string()],
            vec![0, 0, 1],
        ),
        (
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![1, 0, 1, 1],
        ),
    ] {
        assert_eq!(Sol2900::get_longest_subsequence(words, groups), rst);
    }
}

#[test]
fn test_2918() {
    for (rst, nums1, nums2) in [
        (12, vec![3, 2, 0, 1, 0], vec![6, 5, 0]),
        (-1, vec![2, 0, 2, 0], vec![1, 4]),
        (
            139,
            vec![0, 16, 28, 12, 10, 15, 25, 24, 6, 0, 0],
            vec![20, 15, 19, 5, 6, 29, 25, 8, 12],
        ),
    ] {
        assert_eq!(Sol2918::min_sum(nums1, nums2), rst);
    }
}
