use super::*;

#[test]
fn test_29() {
    for (n, d, r) in [(10, 3, 3), (7, -3, -2), (-2147483648, -1, 2147483647)] {
        assert_eq!(Sol29::divide(n, d), r);
    }
}

#[test]
fn test_335() {
    for (rst, distance) in [
        (true, vec![2, 1, 1, 2]),
        (false, vec![1, 2, 3, 4]),
        (true, vec![1, 1, 1, 2, 1]),
        (true, vec![1, 1, 1, 1]),
    ] {
        assert_eq!(Sol335::is_self_crossing(distance), rst);
    }
}

#[test]
fn test_587() {
    for (rst, trees) in [
        (
            vec![[1, 1], [2, 0], [2, 4], [3, 3], [4, 2]],
            vec![[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]],
        ),
        (vec![[1, 2], [2, 2], [4, 2]], vec![[1, 2], [2, 2], [4, 2]]),
        (vec![[1, 5]], vec![[1, 5]]), //87/88
    ] {
        println!("* {trees:?}");
        assert_eq!(Sol587::outer_trees(trees), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_837() {
    for (rst, n, k, max_pts) in [(1.0, 10, 1, 10), (0.6, 6, 1, 10), (0.73278, 21, 17, 10)] {
        println!("* {n} {k} {max_pts}");
        assert!(Sol837::new21_game(n, k, max_pts) - rst <= 1e-5);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_838() {
    for (rst, dominoes) in [
        ("RR.L".to_string(), "RR.L".to_string()),
        ("LL.RR.LLRRLL..".to_string(), ".L.R...LR..L..".to_string()),
    ] {
        assert_eq!(Sol838::push_dominoes(dominoes), rst);
    }
}

#[test]
fn test_892() {
    for (rst, grid) in [
        (34, vec![vec![1, 2], vec![3, 4]]),
        (32, vec![vec![1, 1, 1], vec![1, 0, 1], vec![1, 1, 1]]),
    ] {
        println!("* {grid:?}");
        assert_eq!(Sol892::surface_area(grid), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_908() {
    assert_eq!(Sol908::smallest_range_i(vec![1], 0), 0);
    assert_eq!(Sol908::smallest_range_i(vec![0, 10], 2), 6);
    assert_eq!(Sol908::smallest_range_i(vec![1, 3, 6], 3), 0);
}

#[test]
fn test_970() {
    for (rst, x, y, bound) in [
        (vec![2, 3, 4, 5, 7, 9, 10], 2, 3, 10),
        (vec![2, 4, 6, 8, 10, 14], 3, 5, 15),
        (vec![2, 3, 5, 9], 2, 1, 10),
    ] {
        use std::collections::HashSet;
        let set: HashSet<i32> = rst.into_iter().collect();
        assert!(Sol970::powerful_integers(x, y, bound)
            .iter()
            .all(|x| set.contains(&x)));
    }
}

#[test]
fn test_989() {
    for (rst, num, k) in [
        (vec![1, 2, 3, 4], vec![1, 2, 0, 0], 34),
        (vec![4, 5, 5], vec![2, 7, 4], 181),
        (vec![1, 0, 2, 1], vec![2, 1, 5], 806),
    ] {
        println!("* {num:?} {k}");
        assert_eq!(Sol989::add_to_array_form(num, k), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1295() {
    for (rst, nums) in [
        (2, vec![12, 345, 2, 6, 7896]),
        (1, vec![555, 901, 482, 1771]),
    ] {
        assert_eq!(Sol1295::find_numbers(nums), rst);
    }
}

#[test]
fn test_1432() {
    for (rst, num) in [
        (888, 555),
        (8, 9),
        (820000, 123456),
        (888, 111),         // 205/211
        (8808050, 1101057), // 207/211
    ] {
        println!("* {num}");
        assert_eq!(Sol1432::max_diff(num), rst);
    }
}

#[test]
fn test_1780() {
    for f in [
        Sol1780::check_powers_of_three,
        Sol1780::check_powers_of_three_recursive,
    ] {
        assert_eq!(f(12), true);
        assert_eq!(f(91), true);
        assert_eq!(f(21), false);
    }
}

#[test]
fn test_2081() {
    for (rst, k, n) in [(25, 2, 5), (499, 3, 7), (20379000, 7, 17)] {
        println!("* {k} {n}");
        assert_eq!(Sol2081::k_mirror(k, n), rst);
    }
}

#[test]
fn test_2338() {
    for (rst, n, max_value) in [(10, 2, 5), (11, 5, 3), (27, 2, 10), (510488787, 184, 389)] {
        assert_eq!(Sol2338::ideal_arrays(n, max_value), rst);
    }
}

#[test]
fn test_2523() {
    for (rst, left, right) in [(vec![11, 13], 10, 19), (vec![-1, -1], 4, 6)] {
        assert_eq!(Sol2523::closest_primes(left, right), rst);
    }
}

#[test]
fn test_2566() {
    for (rst, num) in [(99009, 11891), (99, 90), (999, 999), (9, 0)] {
        println!("* {num}");
        assert_eq!(Sol2566::min_max_difference(num), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_2578() {
    for (n, r) in [(1, 1), (2, 5)] {
        assert_eq!(Sol2578::colored_cells(n), r);
    }
}

#[test]
fn test_2818() {
    for (rst, nums, k) in [
        (81, vec![8, 3, 9, 3, 8], 2),
        (4788, vec![19, 12, 14, 6, 10, 18], 3),
        (256720975, vec![3289, 2832, 14858, 22011], 6),
        (630596200, vec![6, 1, 13, 10, 1, 17, 6], 27),
    ] {
        assert_eq!(Sol2818::maximum_score(nums, k), rst);
    }
}

#[test]
fn test_2843() {
    for (rst, low, high) in [(9, 1, 100), (4, 1200, 1230)] {
        assert_eq!(Sol2843::count_symmetric_integers(low, high), rst);
    }
}

#[test]
fn test_2929() {
    for (rst, n, limit) in [(3, 5, 2), (10, 3, 3)] {
        println!("* {n} {limit}");
        assert_eq!(Sol2929::distribute_candies(n, limit), rst);
    }
}

#[test]
fn test_3021() {
    for (rst, n, m) in [(3, 3, 2), (0, 1, 1)] {
        println!("* {n}, {m}");
        assert_eq!(Sol3021::flower_game(n, m), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3024() {
    for (rst, nums) in [
        ("equilateral".to_string(), vec![3, 3, 3]),
        ("scalene".to_string(), vec![3, 4, 5]),
    ] {
        assert_eq!(Sol3024::triangle_type(nums), rst);
    }
}

#[test]
fn test_3272() {
    for (rst, n, k) in [(27, 3, 5), (2, 1, 4), (2468, 5, 6), (9, 2, 1)] {
        assert_eq!(Sol3272::count_good_integers(n, k), rst)
    }
}

#[test]
fn test_3304() {
    for (rst, k) in [('b', 5), ('c', 10)] {
        println!("* {k}");
        assert_eq!(Sol3304::kth_character(k), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3307() {
    for (rst, k, operations) in [('a', 5, vec![0, 0, 0]), ('b', 10, vec![0, 1, 0, 1])] {
        println!("* {k} {operations:?}");
        assert_eq!(Sol3307::kth_character(k, operations), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3405() {
    for (rst, n, m, k) in [(4, 3, 2, 1), (6, 4, 2, 2), (2, 5, 2, 0)] {
        assert_eq!(Sol3405::count_good_arrays(n, m, k), rst);
    }
}

#[test]
fn test_3443() {
    for (rst, s, k) in [(3, "NWSE".to_string(), 1), (6, "NSWWEW".to_string(), 3)] {
        assert_eq!(Sol3443::max_distance(s, k), rst);
    }
}
