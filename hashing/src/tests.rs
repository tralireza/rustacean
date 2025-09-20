use super::*;

#[test]
fn test_336() {
    macro_rules! s {
        ($s:expr) => {
            $s.to_string()
        };
    }

    for (rst, words) in [
        (
            vec![vec![0, 1], vec![1, 0], vec![3, 2], vec![2, 4]],
            vec![s!("abcd"), s!("dcba"), s!("lls"), s!("s"), s!("sssll")],
        ),
        (
            vec![vec![0, 1], vec![1, 0]],
            vec![s!("bat"), s!("tab"), s!("cat")],
        ),
        (vec![vec![0, 1], vec![1, 0]], vec![s!("a"), s!("")]),
        (
            vec![vec![0, 3], vec![3, 0], vec![2, 3], vec![3, 2]],
            vec![s!("a"), s!("abc"), s!("aba"), s!("")],
        ),
    ] {
        println!("* {words:?}");
        let mut count = 0;
        for pair in Sol336::palindrome_pairs(words) {
            assert!(rst.contains(&pair));
            count += 1;
        }
        assert_eq!(rst.len(), count);
        println!(":: {rst:?}");
    }
}

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
fn test_869() {
    for (rst, n) in [(true, 1), (false, 10)] {
        println!("* {n}");
        assert_eq!(Sol869::reordered_power_of2(n), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_898() {
    for (rst, arr) in [(1, vec![0]), (3, vec![1, 1, 2]), (6, vec![1, 2, 4])] {
        println!("* {arr:?}");
        assert_eq!(Sol898::subarray_bitwise_o_rs(arr), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1128() {
    for (rst, dominoes) in [
        (1, vec![vec![1, 2], vec![2, 1], vec![3, 4], vec![5, 6]]),
        (
            3,
            vec![vec![1, 2], vec![1, 2], vec![1, 1], vec![1, 2], vec![2, 2]],
        ),
    ] {
        assert_eq!(Sol1128::num_equiv_domino_pairs(dominoes), rst);
    }
}

#[test]
fn test_1399() {
    for (rst, n) in [(4, 13), (2, 2)] {
        assert_eq!(Sol1399::count_largest_group(n), rst);
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
fn test_1865() {
    let mut o = Sol1865::new(vec![1, 1, 2, 2, 2, 3], vec![1, 4, 5, 2, 5, 4]);
    println!("* {o:?}");

    assert_eq!(o.count(7), 8);
    o.add(3, 2);

    assert_eq!(o.count(8), 2);
    assert_eq!(o.count(4), 1);

    o.add(0, 1);
    o.add(1, 1);

    assert_eq!(o.count(7), 11);
    println!(":: {o:?}");
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
fn test_2363() {
    for (rst, items1, items2) in [
        (
            vec![vec![1, 6], vec![3, 9], vec![4, 5]],
            vec![vec![1, 1], vec![4, 5], vec![3, 8]],
            vec![vec![3, 1], vec![1, 5]],
        ),
        (
            vec![vec![1, 4], vec![2, 4], vec![3, 4]],
            vec![vec![1, 1], vec![3, 2], vec![2, 3]],
            vec![vec![2, 1], vec![3, 2], vec![1, 3]],
        ),
        (
            vec![vec![1, 7], vec![2, 4], vec![7, 1]],
            vec![vec![1, 3], vec![2, 2]],
            vec![vec![7, 1], vec![2, 2], vec![1, 4]],
        ),
    ] {
        println!("* {items1:?} {items2:?}");
        assert_eq!(Sol2363::merge_similar_items(items1, items2), rst);
        println!(":: {rst:?}");
    }
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
fn test_3085() {
    for (rst, word, k) in [
        (3, "aabcaba".to_string(), 0),
        (2, "dabdcbdcdcd".to_string(), 2),
        (1, "aaabaaa".to_string(), 2),
    ] {
        println!("-> '{word}' {k}");
        assert_eq!(Sol3085::minimum_deletions(word, k), rst);
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

#[test]
fn test_3375() {
    for (rst, nums, k) in [
        (2, vec![5, 2, 5, 4, 5], 2),
        (-1, vec![2, 1, 2], 2),
        (4, vec![9, 7, 5, 3], 1),
    ] {
        assert_eq!(Sol3375::min_operations(nums, k), rst);
    }
}

#[test]
fn test_3484() {
    let mut o = Spreadsheet3484::new(3);
    println!("-> {o:?}");

    assert_eq!(o.get_value("=5+7".to_string()), 12);
    o.set_cell("A1".to_string(), 10);
    println!("-> {o:?}");

    assert_eq!(o.get_value("=A1+6".to_string()), 16);
    o.set_cell("B2".to_string(), 15);
    println!("-> {o:?}");

    assert_eq!(o.get_value("=A1+B2".to_string()), 25);
    o.reset_cell("A1".to_string());
    println!("-> {o:?}");

    assert_eq!(o.get_value("=A1+B2".to_string()), 15);
}

#[test]
fn test_3508() {
    let mut o = Router3508::new(3);

    for [src, dst, ts] in [[1, 4, 90], [2, 5, 90], [1, 4, 90], [3, 5, 95], [4, 5, 105]] {
        o.add_packet(src, dst, ts);
    }
    println!("-> {o:?}");

    assert_eq!(o.forward_packet(), vec![2, 5, 90]);

    o.add_packet(5, 2, 110);
    println!("-> {o:?}");

    assert_eq!(o.get_count(5, 100, 110), 1);
}
