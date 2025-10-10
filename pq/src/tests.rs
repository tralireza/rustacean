use super::*;

#[test]
fn test_407() {
    for (rst, height_map) in [
        (
            4,
            vec![
                vec![1, 4, 3, 1, 3, 2],
                vec![3, 2, 1, 3, 2, 4],
                vec![2, 3, 3, 2, 3, 1],
            ],
        ),
        (
            10,
            vec![
                vec![3, 3, 3, 3, 3],
                vec![3, 2, 2, 2, 3],
                vec![3, 2, 1, 2, 3],
                vec![3, 2, 2, 2, 3],
                vec![3, 3, 3, 3, 3],
            ],
        ),
    ] {
        assert_eq!(Sol407::trap_rain_water(height_map), rst);
    }
}

#[test]
fn test_778() {
    for (rst, grid) in [
        (3, vec![vec![0, 2], vec![1, 3]]),
        (
            16,
            vec![
                vec![0, 1, 2, 3, 4],
                vec![24, 23, 22, 21, 5],
                vec![12, 13, 14, 15, 16],
                vec![11, 17, 18, 19, 20],
                vec![10, 9, 8, 7, 6],
            ],
        ),
    ] {
        println!("* {grid:?}");
        assert_eq!(Sol778::swim_in_water(grid), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1046() {
    for (rst, stones) in [(1, vec![2, 7, 4, 1, 8, 1]), (1, vec![1])] {
        println!("* {stones:?}");
        assert_eq!(Sol1046::last_stone_weight(stones), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1792() {
    for (rst, classes, extra_students) in [
        (0.78333, vec![[1, 2], [3, 5], [2, 2]], 2),
        (0.53485, vec![[2, 4], [3, 9], [4, 5], [2, 10]], 4),
    ] {
        let classes = classes
            .into_iter()
            .map(|a| a.into_iter().collect())
            .collect();

        println!("* {classes:?} {extra_students}");
        assert!((Sol1792::max_average_ratio(classes, extra_students) - rst).abs() < 1e-5);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_1912() {
    let mut o = MovieRentingSystem1912::new(
        3,
        [
            [0, 1, 5], // [shop, movie, price]
            [0, 2, 6],
            [0, 3, 7],
            [1, 1, 4],
            [1, 2, 7],
            [2, 1, 5],
        ]
        .into_iter()
        .map(|a| a.into_iter().collect())
        .collect(),
    );
    println!("-> {o:?}");

    assert_eq!(o.search(1), vec![1, 0, 2]);

    for (shop, movie) in [(0, 1), (1, 2)] {
        o.rent(shop, movie);
    }
    println!("-> {o:?}");

    assert_eq!(
        o.report(),
        [[0, 1], [1, 2]]
            .into_iter()
            .map(|a| a.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );

    o.drop(1, 2);
    println!("-> {o:?}");

    assert_eq!(o.search(2), vec![0, 1]);
}

#[test]
fn test_2231() {
    for (rst, num) in [
        (3412, 1234),
        (87655, 65875),
        (427, 247), // 62/238
    ] {
        println!("* {num}");
        assert_eq!(Sol2231::largest_integer(num), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_2353() {
    let mut o = FoodRatings2353::new(
        ["kimchi", "miso", "sushi", "moussaka", "ramen", "bulgogi"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        [
            "korean", "japanese", "japanese", "greek", "japanese", "korean",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect(),
        vec![9, 12, 8, 15, 14, 7],
    );
    println!("-> {o:?}");

    for (rst, cuisine) in [("kimchi", "korean"), ("ramen", "japanese")] {
        println!("* {cuisine:?}");
        assert_eq!(o.highest_rated(cuisine.to_string()), rst);
        println!(":: {rst:?}");
    }

    o.change_rating("sushi".to_string(), 16);
    println!("-> {o:?}");

    for (rst, cuisine) in [("sushi", "japanese")] {
        println!("* {cuisine:?}");
        assert_eq!(o.highest_rated(cuisine.to_string()), rst);
        println!(":: {rst:?}");
    }

    o.change_rating("ramen".to_string(), 16);
    println!("-> {o:?}");

    for (rst, cuisine) in [("ramen", "japanese")] {
        println!("* {cuisine:?}");
        assert_eq!(o.highest_rated(cuisine.to_string()), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3066() {
    for (rst, nums, k) in [
        (2, vec![2, 11, 10, 1, 3], 10),
        (4, vec![1, 1, 2, 4, 9], 20),
        (2, vec![999999999, 999999999, 999999999], 1000000000),
    ] {
        assert_eq!(Sol3066::min_operations(nums, k), rst);
    }
}

#[test]
fn test_3147() {
    for (rst, energy, k) in [(3, vec![5, 2, -10, -5, 1], 3), (-1, vec![-2, -3, -1], 2)] {
        println!("* {energy:?} {k}");
        assert_eq!(Sol3147::maximum_energy(energy, k), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3362() {
    for (rst, nums, queries) in [
        (1, vec![2, 0, 2], vec![vec![0, 2], vec![0, 2], vec![1, 1]]),
        (
            2,
            vec![1, 1, 1, 1],
            vec![vec![1, 3], vec![0, 2], vec![1, 3], vec![1, 2]],
        ),
        (-1, vec![1, 2, 3, 4], vec![vec![0, 3]]),
        (
            4,
            vec![1, 2],
            vec![
                vec![1, 1],
                vec![0, 0],
                vec![1, 1],
                vec![1, 1],
                vec![0, 1],
                vec![0, 0],
            ],
        ),
        (
            2,
            vec![0, 0, 1, 1, 0],
            vec![vec![3, 4], vec![0, 2], vec![2, 3]],
        ),
        (
            -1,
            vec![0, 0, 3],
            vec![vec![0, 2], vec![1, 1], vec![0, 0], vec![0, 0]],
        ),
    ] {
        println!("* {nums:?}");
        assert_eq!(Sol3362::max_removal(nums, queries), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_3408() {
    // [user, task, priority]
    let mut o = TaskManager3408::new(
        [[1, 101, 10], [2, 102, 20], [3, 103, 15]]
            .into_iter()
            .map(|a| a.into_iter().collect())
            .collect(),
    );
    println!("-> {o:?}");

    o.add(4, 104, 5);
    o.edit(102, 8);
    println!("-> {o:?}");

    assert_eq!(o.exec_top(), 3);

    o.rmv(101);
    o.add(5, 105, 15);
    println!("-> {o:?}");

    assert_eq!(o.exec_top(), 5);
}
