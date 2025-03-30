use super::*;

#[test]
fn test_599() {
    assert_eq!(
        Sol599::find_restaurant(
            vec![
                "Shogun".to_string(),
                "Tapioca Express".to_string(),
                "Burger King".to_string(),
                "KFC".to_string()
            ],
            vec![
                "Piatti".to_string(),
                "The Grill at Torrey Pines".to_string(),
                "Hungry Hunter Steakhouse".to_string(),
                "Shogun".to_string()
            ]
        ),
        vec!["Shogun".to_string()]
    );
    assert_eq!(
        Sol599::find_restaurant(
            vec!["happy".to_string(), "sad".to_string(), "good".to_string(),],
            vec!["sad".to_string(), "happy".to_string(), "good".to_string(),]
        ),
        vec!["sad".to_string(), "happy".to_string()]
    );
}

#[test]
fn test_763() {
    for (rst, s) in [
        (vec![9, 7, 8], "ababcbacadefegdehijhklij".to_string()),
        (vec![10], "eccbbbbdec".to_string()),
    ] {
        assert_eq!(Sol763::partition_labels(s), rst);
    }
}

#[test]
fn test_1726() {
    assert_eq!(Sol1726::tuple_same_product(vec![2, 3, 4, 6]), 8);
    assert_eq!(Sol1726::tuple_same_product(vec![1, 2, 4, 5, 10]), 16);
}

#[test]
fn test_1790() {
    assert_eq!(
        Sol1790::are_almost_equal("bank".to_string(), "kanb".to_string()),
        true
    );
    assert_eq!(
        Sol1790::are_almost_equal("attack".to_string(), "defend".to_string()),
        false
    );
    assert_eq!(
        Sol1790::are_almost_equal("kelb".to_string(), "kelb".to_string()),
        true
    );

    assert_eq!(
        Sol1790::are_almost_equal("qgqeg".to_string(), "gqgeq".to_string()),
        false
    );
}

#[test]
fn test_2206() {
    for (rst, nums) in [(true, vec![3, 2, 3, 2, 2, 2]), (false, vec![1, 2, 3, 4])] {
        assert_eq!(Sol2206::divide_array(nums), rst);
    }
}

#[test]
fn test_2342() {
    assert_eq!(Sol2342::maximum_sum(vec![18, 43, 36, 13, 7]), 54);
    assert_eq!(Sol2342::maximum_sum(vec![10, 12, 19, 14]), -1);

    assert_eq!(
        Sol2342::maximum_sum(vec![
            279, 169, 463, 252, 94, 455, 423, 315, 288, 64, 494, 337, 409, 283, 283, 477, 248, 8,
            89, 166, 188, 186, 128
        ]),
        872
    );
}

#[test]
fn test_2349() {
    let mut nc = NumberContainers::new();
    assert_eq!(nc.find(10), -1);
    for i in [2, 1, 3, 5] {
        nc.change(i, 10);
    }
    assert_eq!(nc.find(10), 1);
    nc.change(1, 20);
    assert_eq!(nc.find(10), 2);
}

#[test]
fn test_2364() {
    assert_eq!(Sol2364::count_bad_pairs(vec![4, 1, 3, 3]), 5);
    assert_eq!(Sol2364::count_bad_pairs(vec![1, 2, 3, 4, 5]), 0);
}

#[test]
fn test_2570() {
    assert_eq!(
        Sol2570::merge_arrays(
            vec![vec![1, 2], vec![2, 3], vec![4, 5]],
            vec![vec![1, 4], vec![3, 2], vec![4, 1]]
        ),
        vec![vec![1, 6], vec![2, 3], vec![3, 2], vec![4, 6]]
    );
    assert_eq!(
        Sol2570::merge_arrays(
            vec![vec![2, 4], vec![3, 6], vec![5, 5]],
            vec![vec![1, 3], vec![4, 3]]
        ),
        vec![vec![1, 3], vec![2, 4], vec![3, 6], vec![4, 3], vec![5, 5]]
    );
}

#[test]
fn test_2661() {
    assert_eq!(
        Sol2661::first_complete_index(vec![1, 3, 4, 2], vec![vec![1, 4], vec![2, 3]]),
        2
    );
    assert_eq!(
        Sol2661::first_complete_index(
            vec![2, 8, 7, 4, 1, 3, 5, 6, 9],
            vec![vec![3, 2, 5], vec![1, 4, 6], vec![8, 7, 9]]
        ),
        3
    );
}

#[test]
fn test_2965() {
    for (rst, grid) in [
        (vec![2, 4], vec![vec![1, 3], vec![2, 2]]),
        (
            vec![9, 5],
            vec![vec![9, 1, 7], vec![8, 9, 2], vec![3, 4, 6]],
        ),
    ] {
        assert_eq!(Sol2965::find_missing_and_repeated_values(grid), rst);
    }
}

#[test]
fn test_3160() {
    assert_eq!(
        Sol3160::query_results(4, vec![vec![1, 4], vec![2, 5], vec![1, 3], vec![3, 4]]),
        vec![1, 2, 2, 3]
    );
    assert_eq!(
        Sol3160::query_results(
            4,
            vec![vec![0, 1], vec![1, 2], vec![2, 2], vec![3, 4], vec![4, 5]]
        ),
        vec![1, 2, 2, 3, 4]
    );
}
