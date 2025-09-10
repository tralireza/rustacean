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
fn test_630() {
    for (rst, courses) in [
        (
            3,
            vec![
                vec![100, 200],
                vec![200, 1300],
                vec![1000, 1250],
                vec![2000, 3200],
            ],
        ),
        (1, vec![vec![1, 2]]),
        (0, vec![vec![3, 2], vec![4, 3]]),
        (2, vec![vec![1, 2], vec![2, 3]]),
    ] {
        assert_eq!(Sol630::schedule_course(courses), rst);
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
fn test_1717() {
    for (rst, s, x, y) in [(19, "cdbcbbaaabab", 4, 5), (20, "aabbaaxybbaabb", 5, 4)] {
        println!("* {s:?} {x} {y}");
        assert_eq!(Sol1717::maximum_gain(s.to_string(), x, y), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1733() {
    for (rst, n, languages, friendships) in [
        (
            1,
            2,
            vec![vec![1], vec![2], vec![1, 2]],
            vec![[1, 2], [1, 3], [2, 3]],
        ),
        (
            2,
            3,
            vec![vec![2], vec![1, 3], vec![1, 2], vec![3]],
            vec![[1, 4], [1, 2], [3, 4], [2, 3]],
        ),
    ] {
        println!("* {n} {languages:?} {friendships:?}");
        assert_eq!(
            Sol1733::minimum_teachings(
                n,
                languages,
                friendships
                    .into_iter()
                    .map(|a| a.into_iter().collect())
                    .collect()
            ),
            rst
        );
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1353() {
    for (rst, nums) in [
        (3, vec![vec![1, 2], vec![2, 3], vec![3, 4]]),
        (4, vec![vec![1, 2], vec![2, 3], vec![3, 4], vec![1, 2]]),
    ] {
        println!("* {nums:?}");
        assert_eq!(Sol1353::max_events(nums), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2014() {
    for (rst, s, k) in [
        ("let".to_string(), "letsleetcode".to_string(), 2),
        ("".to_string(), "ab".to_string(), 2),
    ] {
        println!("* {s:?} {k}");
        assert_eq!(Sol2014::longest_subsequence_repeated_k(s, k), rst);
        println!(":: {rst:?}");
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
fn test_2294() {
    for (rst, nums, k) in [
        (2, vec![3, 6, 1, 2, 5], 2),
        (2, vec![1, 2, 3], 1),
        (3, vec![2, 2, 4, 5], 0),
    ] {
        assert_eq!(Sol2294::partition_array(nums, k), rst);
    }
}

#[test]
fn test_2311() {
    for (rst, s, k) in [
        (5, "1001010".to_string(), 5),
        (6, "00101001".to_string(), 1),
    ] {
        println!("* {s:?} {k}");
        assert_eq!(Sol2311::longest_subsequence(s, k), rst);
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

#[bench]
fn bench_2434_hashmap(b: &mut test::Bencher) {
    b.iter(|| test::black_box(Sol2434::robot_with_string("eekstrlpmomwzqummz".to_string())));
}

#[bench]
fn bench_2434_array(b: &mut test::Bencher) {
    b.iter(|| test::black_box(robot_with_string("eekstrlpmomwzqummz".to_string())));

    fn robot_with_string(s: String) -> String {
        let mut freqs = [0; 26];
        for chr in s.as_bytes().iter() {
            freqs[(chr - b'a') as usize] += 1;
        }

        let mut prints = vec![];

        let mut marker = 0;
        let mut stack = vec![];
        for &chr in s.as_bytes().iter() {
            stack.push(chr);
            freqs[(chr - b'a') as usize] -= 1;

            while marker < 25 && freqs[marker] == 0 {
                marker += 1;
            }

            while !stack.is_empty() && marker + b'a' as usize >= stack[stack.len() - 1] as usize {
                if let Some(chr) = stack.pop() {
                    prints.push(chr as char);
                }
            }
        }

        prints.iter().collect()
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

#[test]
fn test_3170() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (rst, s) in [
        (s!("aab"), s!("aaba*")),
        (s!("abc"), s!("abc")),
        (s!("yz"), s!("xyz*")),
    ] {
        println!("* {s}");
        assert_eq!(Sol3170::clear_stars(s), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_3439() {
    for (rst, event_time, k, start_time, end_time) in [
        (2, 5, 1, vec![1, 3], vec![2, 5]),
        (6, 10, 1, vec![0, 2, 9], vec![1, 4, 10]),
        (0, 5, 2, vec![0, 1, 2, 3, 4], vec![1, 2, 3, 4, 5]),
    ] {
        println!("* {event_time} {k} {start_time:?} {end_time:?}");
        assert_eq!(
            Sol3439::max_free_time(event_time, k, start_time, end_time),
            rst
        );
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3440() {
    for (rst, event_time, start_time, end_time) in [
        (2, 5, vec![1, 3], vec![2, 5]),
        (6, 10, vec![0, 2, 9], vec![1, 4, 10]),
        (6, 10, vec![0, 3, 7, 9], vec![1, 4, 8, 10]),
        (0, 5, vec![0, 1, 2, 3, 4], vec![1, 2, 3, 4, 5]),
    ] {
        println!("* {event_time} {start_time:?} {end_time:?}");
        assert_eq!(
            Sol3440::max_free_time(event_time, start_time, end_time),
            rst
        );
        println!(":: {rst:?}");
    }
}
