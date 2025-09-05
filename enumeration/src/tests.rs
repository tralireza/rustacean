use super::*;

#[test]
fn test_2749() {
    for (rst, num1, num2) in [(3, 3, -2), (-1, 5, 7)] {
        println!("* {num1} {num2}");
        assert_eq!(Sol2749::make_the_integer_zero(num1, num2), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3025() {
    for (rst, points) in [
        (0, vec![[1, 1], [2, 2], [3, 3]]),
        (2, vec![[6, 2], [4, 4], [2, 6]]),
        (2, vec![[3, 1], [1, 3], [1, 1]]),
        (0, vec![[0, 0], [2, 5]]), // 148/955
        (1, vec![[0, 3], [6, 1]]), // 150/955
    ] {
        let points: Vec<_> = points
            .into_iter()
            .map(|a| a.into_iter().collect())
            .collect();

        println!("* {points:?}");
        assert_eq!(Sol3025::number_of_pairs(points), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3027() {
    for (rst, points) in [
        (0, vec![[1, 1], [2, 2], [3, 3]]),
        (2, vec![[6, 2], [4, 4], [2, 6]]),
        (2, vec![[3, 1], [1, 3], [1, 1]]),
    ] {
        let points = points
            .into_iter()
            .map(|a| a.into_iter().collect())
            .collect();

        println!("* {points:?}");
        assert_eq!(Sol3027::number_of_pairs(points), rst);
        println!(":: {rst:?}");
    }
}

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
