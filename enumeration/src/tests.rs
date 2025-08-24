use super::*;

#[test]
fn test_3197() {
    for (rst, grid) in [
        (5, vec![vec![1, 0, 1], vec![1, 1, 1]]),
        (5, vec![vec![1, 0, 1, 0], vec![0, 1, 0, 1]]),
    ] {
        println!("* {grid:?}");
        assert_eq!(Sol3197::minimum_sum(grid), rst);
        println!(":: {rst:?}");
    }
}
