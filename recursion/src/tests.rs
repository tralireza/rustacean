use super::*;

#[test]
fn test_37() {
    Sol37::solve_sudoku(&mut vec![
        vec!['5', '3', '.', '.', '7', '.', '.', '.', '.'],
        vec!['6', '.', '.', '1', '9', '5', '.', '.', '.'],
        vec!['.', '9', '8', '.', '.', '.', '.', '6', '.'],
        vec!['8', '.', '.', '.', '6', '.', '.', '.', '3'],
        vec!['4', '.', '.', '8', '.', '3', '.', '.', '1'],
        vec!['7', '.', '.', '.', '2', '.', '.', '.', '6'],
        vec!['.', '6', '.', '.', '.', '.', '2', '8', '.'],
        vec!['.', '.', '.', '4', '1', '9', '.', '.', '5'],
        vec!['.', '.', '.', '.', '8', '.', '.', '7', '9'],
    ]);
}

#[bench]
fn bench_37(b: &mut test::Bencher) {
    b.iter(|| test_37());
}

#[test]
fn test_50() {
    for (rst, x, n) in [
        (1024., 2., 10),
        (9.26100, 2.1, 3),
        (0.25, 2., -2),
        (0., 2., -2147483648),
    ] {
        assert!((Sol50::my_pow(x, n) - rst).abs() < 0.00001);
    }
}

#[test]
fn test_60() {
    assert_eq!(Sol60::get_permutation(3, 3), "213".to_string());
    assert_eq!(Sol60::get_permutation(4, 9), "2314".to_string());
    assert_eq!(Sol60::get_permutation(3, 1), "123".to_string());
}

#[test]
fn test_282() {
    for (rst, num, target) in [
        (
            vec!["1+2+3".to_string(), "1*2*3".to_string()],
            "123".to_string(),
            6,
        ),
        (
            vec!["2+3*2".to_string(), "2*3+2".to_string()],
            "232".to_string(),
            8,
        ),
        (vec![], "3456237490".to_string(), 9191),
    ] {
        assert_eq!(Sol282::add_operators(num, target), rst);
    }
}

#[test]
fn test_301() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (mut rst, s) in [
        (vec![s!("(())()"), s!("()()()")], s!("()())()")),
        (vec![s!("(a())()"), s!("(a)()()")], s!("(a)())()")),
        (vec![s!("")], s!(")(")),
        (vec![s!("((aaaaa))")], s!("((((((((((((((((((aaaaa))")),
    ] {
        let mut calculated = Sol301::remove_invalid_parentheses(s);
        println!(":: {calculated:?}");

        calculated.sort();
        rst.sort();

        assert_eq!(calculated, rst);
    }
}

#[test]
fn test_386() {
    for (rst, n) in [
        (vec![1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9], 13),
        (vec![1, 2], 2),
        (
            vec![
                1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 3, 4, 5, 6, 7, 8, 9,
            ],
            21,
        ),
    ] {
        println!("* {n}");
        assert_eq!(Sol386::lexical_order(n), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1079() {
    assert_eq!(Sol1079::num_tile_possibilities("AAB".to_string()), 8);
    assert_eq!(Sol1079::num_tile_possibilities("AAABBC".to_string()), 188);
    assert_eq!(Sol1079::num_tile_possibilities("V".to_string()), 1);

    assert_eq!(
        Sol1079::num_tile_possibilities("ABCDEFG".to_string()),
        13699
    );
}

#[test]
fn test_1718() {
    // 1 <= n <= 20
    assert_eq!(
        Sol1718::construct_distanced_sequence(3),
        vec![3, 1, 2, 3, 2]
    );
    assert_eq!(
        Sol1718::construct_distanced_sequence(5),
        vec![5, 3, 1, 4, 3, 5, 2, 4, 2]
    );
}

#[bench]
fn bench_1718(b: &mut test::Bencher) {
    b.iter(|| Sol1718::construct_distanced_sequence(20));
}

#[test]
fn test_1922() {
    for (rst, n) in [(5, 1), (400, 4), (564908303, 50)] {
        assert_eq!(Sol1922::count_good_numbers(n), rst);
    }
}

#[test]
fn test_2044() {
    for (rst, nums) in [(2, vec![3, 1]), (7, vec![2, 2, 2]), (6, vec![3, 2, 1, 5])] {
        println!("* {nums:?}");
        assert_eq!(Sol2044::count_max_or_subsets(nums), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2375() {
    assert_eq!(
        Sol2375::smallest_number("IIIDIDDD".to_string()),
        "123549876".to_string()
    );
    assert_eq!(
        Sol2375::smallest_number("DDD".to_string()),
        "4321".to_string()
    );
}

#[test]
fn test_2698() {
    // 1 <= n <= 1000
    assert_eq!(Sol2698::punishment_number(10), 182);
    assert_eq!(Sol2698::punishment_number(37), 1478);
}
