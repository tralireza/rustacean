use super::*;

#[test]
fn test_684() {
    for f in [
        Sol684::find_redundant_connection,
        Sol684::find_redundant_connection_graph,
    ] {
        assert_eq!(f(vec![vec![1, 2], vec![1, 3], vec![2, 3]]), vec![2, 3]);
        assert_eq!(
            f(vec![
                vec![1, 2],
                vec![2, 3],
                vec![3, 4],
                vec![1, 4],
                vec![1, 5]
            ]),
            vec![1, 4]
        );

        assert_eq!(
            f(vec![
                vec![7, 8],
                vec![2, 6],
                vec![2, 8],
                vec![1, 4],
                vec![9, 10],
                vec![1, 7],
                vec![3, 9],
                vec![6, 9],
                vec![3, 5],
                vec![3, 10]
            ]),
            vec![3, 10]
        );
    }
}

#[test]
fn test_695() {
    for (rst, grid) in [
        (
            6,
            vec![
                vec![0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                vec![0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                vec![0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            ],
        ),
        (0, vec![vec![0, 0, 0, 0, 0, 0, 0, 0]]),
    ] {
        assert_eq!(Sol695::max_area_of_island(grid), rst);
    }
}

#[test]
fn test_753() {
    for (rst, n, k) in [
        ("10".to_string(), 1, 2),
        ("01100".to_string(), 2, 2),
        ("9876543210".to_string(), 1, 10),
    ] {
        assert_eq!(Sol753::crack_safe(n, k), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_802() {
    assert_eq!(
        Sol802::eventual_safe_nodes(vec![
            vec![1, 2],
            vec![2, 3],
            vec![5],
            vec![0],
            vec![5],
            vec![],
            vec![]
        ]),
        vec![2, 4, 5, 6]
    );
    assert_eq!(
        Sol802::eventual_safe_nodes(vec![
            vec![1, 2, 3, 4],
            vec![1, 2],
            vec![3, 4],
            vec![0, 4],
            vec![]
        ]),
        vec![4]
    );
}

#[test]
fn test_827() {
    for (rst, grid) in [
        (3, vec![vec![1, 0], vec![0, 1]]),
        (4, vec![vec![1, 1], vec![1, 0]]),
        (4, vec![vec![1, 1], vec![1, 1]]),
    ] {
        println!("* {grid:?}");
        assert_eq!(Sol827::largest_island(grid), rst);
        println!(":: {rst}");
    }
}

#[test]
fn test_909() {
    for (rst, board) in [
        (
            4,
            vec![
                vec![-1, -1, -1, -1, -1, -1],
                vec![-1, -1, -1, -1, -1, -1],
                vec![-1, -1, -1, -1, -1, -1],
                vec![-1, 35, -1, -1, 13, -1],
                vec![-1, -1, -1, -1, -1, -1],
                vec![-1, 15, -1, -1, -1, -1],
            ],
        ),
        (1, vec![vec![-1, -1], vec![-1, 3]]),
        (-1, vec![vec![1, 1, -1], vec![1, 1, 1], vec![-1, 1, 1]]),
        (
            4,
            vec![
                vec![2, -1, -1, -1, -1],
                vec![-1, -1, -1, -1, -1],
                vec![-1, -1, -1, -1, -1],
                vec![-1, -1, -1, -1, -1],
                vec![-1, -1, -1, -1, -1],
            ],
        ),
        (
            2,
            vec![
                vec![-1, -1, 19, 10, -1],
                vec![2, -1, -1, 6, -1],
                vec![-1, 17, -1, 19, -1],
                vec![25, -1, 20, -1, -1],
                vec![-1, -1, -1, -1, 15],
            ],
        ), // 146/216
    ] {
        println!("* {board:?}");
        assert_eq!(Sol909::snakes_and_ladders(board), rst);
    }
}

#[test]
fn test_1267() {
    assert_eq!(Sol1267::count_servers(vec![vec![1, 0], vec![0, 1]]), 0);
    assert_eq!(Sol1267::count_servers(vec![vec![1, 0], vec![1, 1]]), 3);
    assert_eq!(
        Sol1267::count_servers(vec![
            vec![1, 1, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1]
        ]),
        4
    );
}

#[test]
fn test_1298() {
    for (rst, status, candies, keys, contained_boxes, initial_boxes) in [
        (
            16,
            vec![1, 0, 1, 0],
            vec![7, 5, 4, 100],
            vec![vec![], vec![], vec![1], vec![]],
            vec![vec![1, 2], vec![3], vec![], vec![]],
            vec![0],
        ),
        (
            6,
            vec![1, 0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1, 1],
            vec![vec![1, 2, 3, 4, 5], vec![], vec![], vec![], vec![], vec![]],
            vec![vec![1, 2, 3, 4, 5], vec![], vec![], vec![], vec![], vec![]],
            vec![0],
        ),
    ] {
        assert_eq!(
            Sol1298::max_candies(status, candies, keys, contained_boxes, initial_boxes),
            rst
        );
    }
}

#[test]
fn test_1368h() {
    for f in [Sol1368::min_cost, Sol1368::min_cost_bfs01] {
        assert_eq!(
            f(vec![
                vec![1, 1, 1, 1],
                vec![2, 2, 2, 2],
                vec![1, 1, 1, 1],
                vec![2, 2, 2, 2]
            ]),
            3
        );
        assert_eq!(f(vec![vec![1, 1, 3], vec![3, 2, 2], vec![1, 1, 4]]), 0);
        assert_eq!(f(vec![vec![1, 2], vec![4, 3]]), 1);
    }
}

#[test]
fn test_1462() {
    assert_eq!(
        Sol1462::check_if_prerequisite(2, vec![vec![1, 0]], vec![vec![0, 1], vec![1, 0]]),
        vec![false, true]
    );
    assert_eq!(
        Sol1462::check_if_prerequisite(2, vec![], vec![vec![1, 0], vec![0, 1]]),
        vec![false, false]
    );
    assert_eq!(
        Sol1462::check_if_prerequisite(
            3,
            vec![vec![1, 2], vec![0, 1], vec![2, 0]],
            vec![vec![1, 0], vec![1, 2]]
        ),
        vec![true, true]
    );
}

#[test]
fn test_1765() {
    assert_eq!(
        Sol1765::highest_peak(vec![vec![0, 1], vec![0, 0]]),
        vec![vec![1, 0], vec![2, 1]]
    );
    assert_eq!(
        Sol1765::highest_peak(vec![vec![0, 0, 1], vec![1, 0, 0], vec![0, 0, 0]]),
        vec![vec![1, 1, 0], vec![0, 1, 1], vec![1, 2, 2]]
    );
}

#[test]
fn test_1976() {
    for (rst, n, roads) in [
        (
            4,
            7,
            vec![
                vec![0, 6, 7],
                vec![0, 1, 2],
                vec![1, 2, 3],
                vec![1, 3, 3],
                vec![6, 3, 3],
                vec![3, 5, 1],
                vec![6, 5, 1],
                vec![2, 5, 1],
                vec![0, 4, 5],
                vec![4, 6, 2],
            ],
        ),
        (1, 2, vec![vec![1, 0, 10]]),
    ] {
        assert_eq!(Sol1976::count_paths(n, roads), rst);
    }
}

#[test]
fn test_2127() {
    for (rst, favorite) in [
        (3, vec![2, 2, 1, 2]),
        (3, vec![1, 2, 0]),
        (4, vec![3, 0, 1, 4, 1]),
    ] {
        assert_eq!(Sol2127::maximum_invitations(favorite), rst);
    }
}

#[test]
fn test_2359() {
    for (rst, edges, node1, node2) in [
        (2, vec![2, 2, 3, -1], 0, 1),
        (2, vec![1, 2, -1], 0, 2),
        (0, vec![1, 0], 0, 1),
        (4, vec![4, 3, 0, 5, 3, -1], 4, 0),
    ] {
        println!("* {edges:?}");
        assert_eq!(Sol2359::closest_meeting_node(edges, node1, node2), rst);
    }
}

#[test]
fn test_2467() {
    assert_eq!(
        Sol2467::most_profitable_path(
            vec![vec![0, 1], vec![1, 2], vec![1, 3], vec![3, 4]],
            3,
            vec![-2, 4, 2, -4, 6]
        ),
        6
    );
    assert_eq!(
        Sol2467::most_profitable_path(vec![vec![0, 1]], 1, vec![-7280, 2350]),
        -7280
    );
    assert_eq!(
        Sol2467::most_profitable_path(vec![vec![0, 1], vec![0, 2]], 2, vec![-3360, -5394, -1146]),
        -3360
    );
}

#[test]
fn test_2493() {
    assert_eq!(
        Sol2493::magnificent_sets(
            6,
            vec![
                vec![1, 2],
                vec![1, 4],
                vec![1, 5],
                vec![2, 6],
                vec![2, 3],
                vec![4, 6]
            ]
        ),
        4
    );
    assert_eq!(
        Sol2493::magnificent_sets(3, vec![vec![1, 2], vec![2, 3], vec![3, 1],]),
        -1
    );
}

#[test]
fn test_2503() {
    for (rst, grid, queries) in [
        (
            vec![5, 8, 1],
            vec![vec![1, 2, 3], vec![2, 5, 7], vec![3, 5, 1]],
            vec![5, 6, 2],
        ),
        (vec![0], vec![vec![5, 2, 1], vec![1, 1, 2]], vec![3]),
    ] {
        assert_eq!(Sol2503::max_points(grid, queries), rst);
    }
}

#[test]
fn test_2608() {
    assert_eq!(
        Sol2608::find_shortest_cycle(
            7,
            vec![
                vec![0, 1],
                vec![1, 2],
                vec![2, 0],
                vec![3, 4],
                vec![4, 5],
                vec![5, 6],
                vec![6, 3]
            ]
        ),
        3
    );
    assert_eq!(
        Sol2608::find_shortest_cycle(4, vec![vec![0, 1], vec![0, 2]]),
        -1
    );
}

#[test]
fn test_2658() {
    for f in [Sol2658::find_max_fish, Sol2658::find_max_fish_recursion] {
        assert_eq!(
            f(vec![
                vec![0, 2, 1, 0],
                vec![4, 0, 0, 3],
                vec![1, 0, 0, 4],
                vec![0, 3, 2, 0]
            ]),
            7
        );
        assert_eq!(
            f(vec![
                vec![1, 0, 0, 0],
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 1]
            ]),
            1
        );

        assert_eq!(f(vec![vec![4, 5, 5], vec![0, 10, 0],]), 24);
        assert_eq!(f(vec![vec![8, 6], vec![2, 6]]), 22);
    }
}

#[test]
fn test_2685() {
    for (rst, n, edges) in [
        (3, 6, vec![vec![0, 1], vec![0, 2], vec![1, 2], vec![3, 4]]),
        (
            1,
            6,
            vec![vec![0, 1], vec![0, 2], vec![1, 2], vec![3, 4], vec![3, 5]],
        ),
    ] {
        assert_eq!(Sol2685::count_complete_components(n, edges), rst);
    }
}

#[test]
fn test_3108() {
    for (rst, n, edges, query) in [
        (
            vec![1, -1],
            5,
            vec![vec![0, 1, 7], vec![1, 3, 7], vec![1, 2, 1]],
            vec![vec![0, 3], vec![3, 4]],
        ),
        (
            vec![0],
            3,
            vec![vec![0, 2, 7], vec![0, 1, 15], vec![1, 2, 6], vec![1, 2, 1]],
            vec![vec![1, 2]],
        ),
        (
            vec![0],
            7,
            vec![
                vec![3, 0, 2],
                vec![5, 4, 12],
                vec![6, 3, 7],
                vec![4, 2, 2],
                vec![6, 2, 2],
            ],
            vec![vec![6, 0]],
        ),
    ] {
        assert_eq!(Sol3108::minimum_cost(n, edges, query), rst);
    }
}

#[test]
fn test_3341() {
    for (rst, move_time) in [
        (6, vec![vec![0, 4], vec![4, 4]]),
        (3, vec![vec![0, 0, 0], vec![0, 0, 0]]),
        (3, vec![vec![0, 1], vec![1, 2]]),
        (60, vec![vec![15, 58], vec![67, 4]]),
    ] {
        println!("** {:?}", move_time);
        assert_eq!(Sol3341::min_time_to_reach(move_time), rst);
    }
}

#[test]
fn test_3342() {
    for (rst, move_time) in [
        (7, vec![vec![0, 4], vec![4, 4]]),
        (6, vec![vec![0, 0, 0, 0], vec![0, 0, 0, 0]]),
        (4, vec![vec![0, 1], vec![1, 2]]),
    ] {
        println!("** {:?}", move_time);
        assert_eq!(Sol3342::min_time_to_reach(move_time), rst);
    }
}

#[test]
fn test_3372() {
    for (rst, edges1, edges2, k) in [
        (
            vec![9, 7, 9, 8, 8],
            vec![vec![0, 1], vec![0, 2], vec![2, 3], vec![2, 4]],
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![2, 7],
                vec![1, 4],
                vec![4, 5],
                vec![4, 6],
            ],
            2,
        ),
        (
            vec![6, 3, 3, 3, 3],
            vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![0, 4]],
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            1,
        ),
    ] {
        println!("* {k}");
        assert_eq!(Sol3372::max_target_nodes(edges1, edges2, k), rst);
        println!(":: {rst:?}");
    }
}

#[test]
fn test_3373() {
    for (rst, edges1, edges2) in [
        (
            vec![8, 7, 7, 8, 8],
            vec![vec![0, 1], vec![0, 2], vec![2, 3], vec![2, 4]],
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![2, 7],
                vec![1, 4],
                vec![4, 5],
                vec![4, 6],
            ],
        ),
        (
            vec![3, 6, 6, 6, 6],
            vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![0, 4]],
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        ),
    ] {
        assert_eq!(Sol3373::max_target_nodes(edges1, edges2), rst);
        println!(":: {rst:?}");
    }
}
