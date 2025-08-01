use super::*;

#[test]
fn test_75() {
    for (rst, mut nums) in [
        (vec![0, 0, 1, 1, 2, 2], vec![2, 0, 2, 1, 1, 0]),
        (vec![0, 1, 2], vec![2, 0, 1]),
        (vec![1, 2], vec![1, 2]),
        (vec![2], vec![2]),
    ] {
        Sol75::sort_colors(&mut nums);
        assert_eq!(nums, rst);
    }
}

#[test]
fn test_220() {
    for (rst, nums, idiff, vdiff) in [
        (true, vec![1, 2, 3, 1], 3, 0),
        (false, vec![1, 5, 9, 1, 5, 9], 2, 3),
        (false, vec![4, 2], 2, 1),
        (true, vec![4, 1, 6, 3], 4, 1),
    ] {
        assert_eq!(
            Sol220::contains_nearby_almost_duplicate(nums, idiff, vdiff),
            rst
        );
    }
}

#[test]
fn test_905() {
    assert_eq!(
        Sol905::sort_array_by_parity(vec![3, 1, 2, 4]),
        vec![4, 2, 3, 1]
    );
    assert_eq!(Sol905::sort_array_by_parity(vec![0]), vec![0]);
}

#[test]
fn test_2410() {
    for (rst, players, trainers) in [
        (2, vec![4, 7, 9], vec![8, 2, 5, 8]),
        (1, vec![1, 1, 1], vec![10]),
    ] {
        println!("* {players:?} {trainers:?}");
        assert_eq!(Sol2410::match_players_and_trainers(players, trainers), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2551() {
    for (rst, weights, k) in [(4, vec![1, 3, 5, 1], 2), (0, vec![1, 3], 2)] {
        assert_eq!(Sol2551::put_marbles(weights, k), rst);
    }
}

#[test]
fn test_2948() {
    assert_eq!(
        Sol2948::lexicographically_smallest_array(vec![1, 5, 3, 9, 8], 2),
        vec![1, 3, 5, 8, 9]
    );
    assert_eq!(
        Sol2948::lexicographically_smallest_array(vec![1, 7, 6, 18, 2, 1], 3),
        vec![1, 6, 7, 18, 1, 2]
    );
    assert_eq!(
        Sol2948::lexicographically_smallest_array(vec![1, 7, 28, 19, 10], 2),
        vec![1, 7, 28, 19, 10]
    );
}
