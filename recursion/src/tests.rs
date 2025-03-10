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
