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
fn test_804() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (rst, words) in [
        (2, vec![s!("gin"), s!("zen"), s!("gig"), s!("msg")]),
        (1, vec!["a".to_string()]),
    ] {
        println!("* {words:?}");
        assert_eq!(Sol804::unique_morse_representations(words), rst);
        println!(":: {rst:?}");
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
fn test_1078() {
    for (rst, text, first, second) in [
        (
            vec!["girl", "student"],
            "alice is a good girl she is a good student",
            "a",
            "good",
        ),
        (vec!["we", "rock"], "we will we will rock you", "we", "will"),
    ] {
        println!("* {text:?} {first:?} {second:?}");
        assert_eq!(
            Sol1078::find_ocurrences(text.to_string(), first.to_string(), second.to_string()),
            rst
        );
        println!(":: {rst:?}");
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
fn test_1876() {
    for (rst, s) in [(1, "xyzzaz"), (4, "aababcabc")] {
        println!("* {s:?}");
        assert_eq!(Sol1876::count_good_substrings(s.to_string()), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2138() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (rst, s, k, fill) in [
        (
            vec![s!("abc"), s!("def"), s!("ghi")],
            "abcdefghi".to_string(),
            3,
            'x',
        ),
        (
            vec![s!("abc"), s!("def"), s!("ghi"), s!("jxx")],
            "abcdefghij".to_string(),
            3,
            'x',
        ),
    ] {
        println!("* {s:?} {k} {fill:?}");
        assert_eq!(Sol2138::divide_string(s, k, fill), rst);
        println!(":: {rst:?}");
    }

    println!("---");
    let chrs: Vec<_> = "abcdefghij".to_string().chars().collect();
    println!("-> {} :: {chrs:?}", std::any::type_name_of_val(&chrs));

    let chks: Vec<_> = chrs.chunks(3).collect();
    println!("-> {} :: {chks:?}", std::any::type_name_of_val(&chks));

    let divs: Vec<_> = chks
        .iter()
        .map(|chk| chk.iter().collect::<String>())
        .collect();
    println!("-> {} :: {divs:?}", std::any::type_name_of_val(&divs));
    println!("---");
}

#[bench]
fn bench_2138(b: &mut test::Bencher) {
    b.iter(|| test::black_box(Sol2138::divide_string("abcdefghij".to_string(), 3, 'x')));
}

#[bench]
fn bench_2138_rusty(b: &mut test::Bencher) {
    fn rusty(s: String, k: i32, fill: char) -> Vec<String> {
        let chrs: Vec<_> = s.chars().collect();
        let chks: Vec<_> = chrs.chunks(3).collect();
        let mut divs: Vec<_> = chks
            .iter()
            .map(|chk| chk.iter().collect::<String>())
            .collect();

        if let Some(last) = divs.last_mut() {
            *last += &fill.to_string().repeat(k as usize - last.len());
        }

        divs
    }

    b.iter(|| test::black_box(rusty("abcdefghij".to_string(), 3, 'x')));
}

#[test]
fn test_2273() {
    for (rst, words) in [
        (vec!["abba", "cd"], vec!["abba", "baba", "bbaa", "cd", "cd"]),
        (vec!["a", "b", "c", "d", "e"], vec!["a", "b", "c", "d", "e"]),
    ] {
        println!("* {words:?}");
        assert_eq!(
            Sol2273::remove_anagrams(words.into_iter().map(|w| w.to_string()).collect()),
            rst
        );
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2315() {
    for (rst, s) in [(5, "yo|uar|e**|b|e***au|tifu|l"), (0, "iamprogrammer")] {
        println!("* {s:?}");
        assert_eq!(Sol2315::count_asterisks(s.to_string()), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3330() {
    for (rst, word) in [
        (5, "abbcccc".to_string()),
        (1, "abcd".to_string()),
        (4, "aaaa".to_string()),
    ] {
        println!("* {word:?}");
        assert_eq!(Sol3330::possible_string_count(word), rst);
        println!(":: {rst:?}");
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
