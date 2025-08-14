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
fn test_329() {
    for (rst, matrix) in [
        (4, vec![vec![9, 9, 4], vec![6, 6, 8], vec![2, 1, 1]]),
        (4, vec![vec![3, 4, 5], vec![3, 2, 6], vec![2, 2, 1]]),
        (1, vec![vec![1]]),
    ] {
        println!("* {matrix:?}");
        assert_eq!(Sol329::longest_increasing_path(matrix), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_368() {
    for (rst, nums) in [
        (vec![1, 2], vec![1, 2, 3]),
        (vec![1, 2, 4, 8], vec![1, 2, 4, 8]),
        (
            vec![9, 18, 90, 180, 360, 720],
            vec![5, 9, 18, 54, 108, 540, 90, 180, 360, 720],
        ),
        (vec![1], vec![1]),
    ] {
        println!("** {:?}", nums.to_vec());
        assert_eq!(Sol368::largest_divisible_subset(nums), rst);
    }
}

#[test]
fn test_377() {
    for (rst, nums, target) in [(7, vec![1, 2, 3], 4), (0, vec![9], 3)] {
        assert_eq!(Sol377::combination_sum4(nums, target), rst);
    }
}

#[test]
fn test_516() {
    for (rst, s) in [(4, "bbbab".to_string()), (2, "cbbd".to_string())] {
        assert_eq!(Sol516::longest_palindrome_subseq(s), rst);
    }
}

#[test]
fn test_639() {
    for (rst, s) in [
        (9, "*".to_string()),
        (18, "1*".to_string()),
        (15, "2*".to_string()),
        (3, "126".to_string()),
    ] {
        assert_eq!(Sol639::num_decodings(s), rst);
    }
}

#[test]
fn test_730() {
    for (rst, s) in [
        (6, "bccb".to_string()),
        (
            104860361,
            "abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba".to_string(),
        ),
    ] {
        println!("* {s}");
        assert_eq!(Sol730::count_palindromic_subsequences(s), rst);
    }
}

#[test]
fn test_790() {
    for (rst, n) in [(5, 3), (1, 1), (11, 4)] {
        assert_eq!(Sol790::num_tilings(n), rst);
    }
}

#[test]
fn test_808() {
    for (rst, n) in [
        (0.625, 50),
        (0.71875, 100),
        (0.625, 1),       // 39/43
        (1.0, 660295675), // 20/43
    ] {
        println!("* {n}");
        assert_eq!(Sol808::soup_servings(n), rst);
        println!(":: {rst:?}");
    }
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
fn test_1751() {
    for (rst, events, k) in [
        (7, vec![vec![1, 2, 4], vec![3, 4, 3], vec![2, 3, 1]], 2),
        (10, vec![vec![1, 2, 4], vec![3, 4, 3], vec![2, 3, 10]], 2),
        (
            9,
            vec![vec![1, 1, 1], vec![2, 2, 2], vec![3, 3, 3], vec![4, 4, 4]],
            3,
        ),
    ] {
        println!("* {events:?} {k}");
        assert_eq!(Sol1751::max_values(events, k), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1857() {
    for (rst, colors, edges) in [
        (
            3,
            "abaca".to_string(),
            vec![vec![0, 1], vec![0, 2], vec![2, 3], vec![3, 4]],
        ),
        (-1, "a".to_string(), vec![vec![0, 0]]),
        (
            3,
            "hhqhuqhqff".to_string(),
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![2, 3],
                vec![3, 4],
                vec![3, 5],
                vec![5, 6],
                vec![2, 7],
                vec![6, 7],
                vec![7, 8],
                vec![3, 8],
                vec![5, 8],
                vec![8, 9],
                vec![3, 9],
                vec![6, 9],
            ],
        ),
        (1, "c".to_string(), vec![]),
        (1, "abcz".to_string(), vec![]),
    ] {
        println!("* {edges:?}");
        assert_eq!(Sol1857::largest_path_value(colors, edges), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_1931() {
    for (rst, m, n) in [(3, 1, 1), (6, 1, 2), (580986, 5, 5), (408208448, 5, 1000)] {
        assert_eq!(Sol1931::color_the_grid(m, n), rst);
        println!(":: {rst}");
    }
}

#[bench]
fn bench_1931_hashmap(b: &mut test::Bencher) {
    use std::collections::HashMap;

    let mut dp_cur = HashMap::with_capacity(27);
    for mask in [1, 4, 5, 9, 11, 23, 24, 25, 26] {
        dp_cur.insert(mask, 1);
    }

    b.iter(|| {
        test::black_box(for mask in [1, 2, 4, 11, 19, 20, 25, 26] {
            let _ = dp_cur.get(&mask);
        });
    });
}

#[bench]
fn bench_1931_array(b: &mut test::Bencher) {
    let mut dp_cur = vec![0; 27];
    for mask in [1, 4, 5, 9, 11, 23, 24, 25, 26] {
        dp_cur[mask] = 1;
    }

    b.iter(|| {
        test::black_box(for mask in [1, 2, 4, 11, 19, 20, 25, 26] {
            let _ = dp_cur[mask];
        });
    });
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
        for f in [Sol2140::most_points, Sol2140::recursive] {
            assert_eq!(f(questions.to_vec()), rst);
        }
    }
}

#[test]
fn test_2787() {
    for (rst, n, x) in [(1, 10, 2), (2, 4, 1)] {
        println!("* {n} {x}");
        assert_eq!(Sol2787::number_of_ways(n, x), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2836() {
    // 1 <= Receiver.Length <= 10^5,  1 <= k <= 10^10
    assert_eq!(Sol2836::get_max_function_value(vec![2, 0, 1], 4), 6);
    assert_eq!(Sol2836::get_max_function_value(vec![1, 1, 1, 2, 3], 3), 10);
}

#[test]
fn test_2901() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (rst, words, groups) in [
        (
            vec![s!("bab"), s!("dab")],
            vec![s!("bab"), s!("dab"), s!("cab")],
            vec![1, 2, 2],
        ),
        (
            vec![s!("a"), s!("b"), s!("c"), s!("d")],
            vec![s!("a"), s!("b"), s!("c"), s!("d")],
            vec![1, 2, 3, 4],
        ),
    ] {
        println!("* {words:?}");
        assert_eq!(
            Sol2901::get_words_in_longest_subsequence(words, groups),
            rst
        );
    }
}

#[test]
fn test_2999() {
    for (rst, start, finish, limit, s) in [
        (5, 1, 6000, 4, "124".to_string()),
        (2, 15, 215, 6, "10".to_string()),
        (0, 1000, 2000, 4, "3000".to_string()),
        (16135677999, 697662853, 11109609599885, 6, "5".to_string()),
    ] {
        assert_eq!(
            Sol2999::number_of_powerful_int(start, finish, limit, s),
            rst
        );
    }
}

#[test]
fn test_3068() {
    for (rst, nums, k, edges) in [
        (6, vec![1, 2, 1], 3, vec![vec![0, 1], vec![0, 2]]),
        (9, vec![2, 3], 7, vec![vec![0, 1]]),
        (
            42,
            vec![7, 7, 7, 7, 7, 7],
            3,
            vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![0, 4]],
        ),
    ] {
        assert_eq!(Sol3068::maximum_value_sum(nums, k, edges), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3335() {
    for (rst, s, t) in [(7, "abcyy".to_string(), 2), (5, "azbk".to_string(), 1)] {
        assert_eq!(Sol3335::length_after_transformations(s, t), rst);
    }
}

#[test]
fn test_3337() {
    for (rst, s, t, nums) in [
        (
            7,
            "abcyy".to_string(),
            2,
            vec![
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            ],
        ),
        (
            8,
            "azbk".to_string(),
            1,
            vec![
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            ],
        ),
        (
            417796858,
            "x".to_string(),
            16,
            vec![
                6, 6, 8, 1, 9, 9, 10, 3, 9, 4, 8, 5, 2, 8, 10, 2, 6, 8, 2, 3, 3, 7, 2, 6, 4, 2,
            ],
        ),
    ] {
        assert_eq!(Sol3337::length_after_transformations(s, t, nums), rst);
    }
}

#[test]
fn test_3363() {
    for (rst, fruits) in [
        (
            100,
            vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 8, 7],
                vec![9, 10, 11, 12],
                vec![13, 14, 15, 16],
            ],
        ),
        (4, vec![vec![1, 1], vec![1, 1]]),
    ] {
        println!("* {fruits:?}");
        assert_eq!(Sol3363::max_collected_fruits(fruits), rst);
        println!(":: {rst:?}");
    }
}
