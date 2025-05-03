use super::*;

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
