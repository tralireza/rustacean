use super::*;

#[test]
fn test_153() {
    assert_eq!(Sol153::find_min(vec![3, 4, 5, 1, 2]), 1);
    assert_eq!(Sol153::find_min(vec![4, 5, 6, 7, 0, 1, 2]), 0);
    assert_eq!(Sol153::find_min(vec![11, 13, 15, 17]), 11);
}

#[test]
fn test_154() {
    assert_eq!(Sol154::find_min(vec![1, 3, 5]), 1);
    assert_eq!(Sol154::find_min(vec![2, 2, 2, 0, 1]), 0);
}

#[test]
fn test_315() {
    for (rst, nums) in [
        (vec![2, 1, 1, 0], vec![5, 2, 6, 1]),
        (vec![0], vec![-1]),
        (vec![0, 0], vec![-1, -1]),
    ] {
        assert_eq!(Sol315::count_smaller(nums.to_vec()), rst);
        assert_eq!(Sol315::bit_count_smaller(nums), rst);
    }
}

#[test]
fn test_611() {
    for (rst, nums) in [
        (3, vec![2, 2, 3, 4]),
        (4, vec![4, 2, 3, 4]),
        (0, vec![1, 1, 3, 4]),                              // 17/241
        (10, vec![24, 3, 82, 22, 35, 84, 19]),              // 39/241
        (91, vec![61, 73, 62, 57, 46, 11, 33, 79, 79, 60]), // 127/241
    ] {
        println!("* {nums:?}");
        assert_eq!(Sol611::triangle_number(nums), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_704() {
    for (rst, target, nums) in [
        (4, 9, vec![-1, 0, 3, 5, 9, 12]),
        (-1, 2, vec![-1, 0, 3, 5, 9, 12]),
        (0, -1, vec![-1, 0, 3, 5, 9, 12]),
    ] {
        println!(":: {}", rst);
        assert_eq!(Sol704::search(nums, target), rst);
    }
}

#[test]
fn test_793() {
    for (rst, k) in [
        (5, 0),
        (0, 5),
        (5, 3),
        (5, 1000000000), // 43/44
    ] {
        println!("* {k}");
        assert_eq!(Sol793::preimage_size_fzf(k), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1498() {
    for (rst, nums, target) in [
        (4, vec![3, 5, 6, 7], 9),
        (6, vec![3, 3, 6, 8], 10),
        (61, vec![2, 3, 3, 4, 6, 7], 12),
        (
            272187084,
            vec![
                14, 4, 6, 6, 20, 8, 5, 6, 8, 12, 6, 10, 14, 9, 17, 16, 9, 7, 14, 11, 14, 15, 13,
                11, 10, 18, 13, 17, 17, 14, 17, 7, 9, 5, 10, 13, 8, 5, 18, 20, 7, 5, 5, 15, 19, 14,
            ],
            22,
        ), // 20/63
    ] {
        println!("* {nums:?} {target}");
        assert_eq!(Sol1498::num_subseq(nums, target), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2040() {
    for (rst, nums1, nums2, k) in [
        (8, vec![2, 5], vec![3, 4], 2),
        (0, vec![-4, -2, 0, 3], vec![2, 4], 6),
        (-6, vec![-2, -1, 0, 1, 2], vec![-3, -1, 2, 4, 5], 3),
    ] {
        println!("* {nums1:?} {nums2:?} {k}",);
        assert_eq!(Sol2040::kth_smallest_product(nums1, nums2, k), rst);
    }
}

#[test]
fn test_2071() {
    for (rst, tasks, workers, pills, strength) in [
        (3, vec![3, 2, 1], vec![0, 3, 3], 1, 1),
        (1, vec![5, 4], vec![0, 0, 0], 1, 5),
        (2, vec![10, 15, 30], vec![0, 10, 10, 10, 10], 3, 10),
    ] {
        assert_eq!(
            Sol2071::max_task_assign(tasks, workers, pills, strength),
            rst
        );
    }
}

#[test]
fn test_2226() {
    for (rst, candies, k) in [(5, vec![5, 8, 6], 3), (0, vec![2, 5], 11)] {
        assert_eq!(Sol2226::maximum_candies(candies, k), rst);
    }
}

#[test]
fn test_2529() {
    for (rst, nums) in [
        (3, vec![-2, -1, -1, 1, 2, 3]),
        (3, vec![-3, -2, -1, 0, 0, 1, 2]),
        (4, vec![5, 20, 66, 1314]),
    ] {
        assert_eq!(Sol2529::maximum_count(nums), rst);
    }
}

#[test]
fn test_2560() {
    for (rst, nums, k) in [(5, vec![2, 3, 5, 9], 2), (2, vec![2, 7, 9, 3, 1], 2)] {
        println!("** {:?}", (&nums, k));
        assert_eq!(Sol2560::min_capability(nums, k), rst);
    }
}

#[test]
fn test_2563() {
    for (rst, nums, lower, upper) in [
        (6, vec![0, 1, 7, 4, 4, 5], 3, 6),
        (1, vec![1, 7, 9, 2, 5], 11, 11),
    ] {
        assert_eq!(Sol2563::count_fair_pairs(nums, lower, upper), rst);
    }
}

#[test]
fn test_2594() {
    for (rst, ranks, cars) in [(16, vec![4, 2, 3, 1], 10), (16, vec![5, 1, 8], 6)] {
        assert_eq!(Sol2594::repair_cars(ranks, cars), rst);
    }
}

#[test]
fn test_2616() {
    for (rst, nums, p) in [
        (1, vec![10, 1, 2, 7, 1, 3], 2),
        (0, vec![4, 2, 1, 2], 1),
        (1, vec![3, 4, 2, 3, 2, 1, 2], 3),
        (2, vec![2, 6, 2, 4, 2, 2, 0, 2], 4),
    ] {
        println!("* {nums:?} {p}");
        assert_eq!(Sol2616::minimize_max(nums, p), rst);
    }
}

#[test]
fn test_3356() {
    for (rst, nums, queries) in [
        (
            2,
            vec![2, 0, 2],
            vec![vec![0, 2, 1], vec![0, 2, 1], vec![1, 1, 3]],
        ),
        (-1, vec![4, 3, 2, 1], vec![vec![1, 3, 2], vec![0, 2, 1]]),
    ] {
        assert_eq!(Sol3356::min_zero_array(nums, queries), rst);
    }
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
