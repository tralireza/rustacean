use super::*;

#[test]
fn test_38() {
    for (rst, n) in [
        ("1211".to_string(), 4),
        ("1".to_string(), 1),
        ("111221".to_string(), 5),
    ] {
        assert_eq!(Sol38::count_and_say(n), rst);
    }
}

#[test]
fn test_65() {
    for f in [Sol65::is_number, Sol65::is_number_enum] {
        for (rst, s) in [
            (true, "0".to_string()),
            (false, "e".to_string()),
            (false, ".".to_string()),
            (true, "2e0".to_string()),
        ] {
            assert_eq!(f(s), rst);
        }
    }
}

#[test]
fn test_466() {
    for (rst, s1, n1, s2, n2) in [
        (2, "abc".to_string(), 4, "ab".to_string(), 2),
        (1, "acb".to_string(), 1, "acb".to_string(), 1),
        (12, "aaa".to_string(), 20, "aaaaa".to_string(), 1),
    ] {
        println!("* {s1}:{n1} {s2}:{n2}");
        assert_eq!(Sol466::get_max_repetitions(s1, n1, s2, n2), rst);
    }
}

#[test]
fn test_917() {
    assert_eq!(
        Sol917::reverse_only_letters("a-bC-dEf-ghIj".to_string()),
        "j-Ih-gfE-dCba".to_string()
    );
}

#[test]
fn test_1061() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (rst, s1, s2, base_str) in [
        (s!("makkek"), s!("parker"), s!("morris"), s!("parser")),
        (s!("hdld"), s!("hello"), s!("world"), s!("hold")),
        (
            s!("aauaaaaada"),
            s!("leetcode"),
            s!("programs"),
            s!("sourcecode"),
        ),
    ] {
        println!("* {s1} ~ {s2}");
        assert_eq!(Sol1061::smallest_equivalent_string(s1, s2, base_str), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_1154() {
    assert_eq!(Sol1154::day_of_year("2019-01-09".to_string()), 9);
    assert_eq!(Sol1154::day_of_year("2019-02-10".to_string()), 41);
}

#[test]
fn test_1163() {
    for (rst, s) in [
        ("bab".to_string(), "abab".to_string()),
        ("tcode".to_string(), "leetcode".to_string()),
    ] {
        println!("* {s}");
        assert_eq!(Sol1163::last_substring(s), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_3403() {
    for (rst, word, num_friends) in [
        ("dbc".to_string(), "dbca".to_string(), 2),
        ("g".to_string(), "gggg".to_string(), 4),
        ("gh".to_string(), "gh".to_string(), 1), // 694/785
    ] {
        assert_eq!(Sol3403::answer_string(word, num_friends), rst);
    }
}

#[test]
fn test_3442() {
    for (rst, s) in [(3, "aaaaabbc".to_string()), (1, "abcabcab".to_string())] {
        assert_eq!(Sol3442::max_difference(s), rst);
    }
}
