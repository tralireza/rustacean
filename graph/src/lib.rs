//! # Graph (DFS, BFS, Topological Sort, Kahn's)

/// 802m Find Eventual Safe States
struct Sol802;

impl Sol802 {
    pub fn eventual_safe_nodes(graph: Vec<Vec<i32>>) -> Vec<i32> {
        let n = graph.len();

        let (mut rvs, mut ins) = (vec![vec![]; n], vec![0; n]);
        (0..n).for_each(|v| {
            graph[v].iter().for_each(|&u| {
                ins[v as usize] += 1;
                rvs[u as usize].push(v);
            });
        });

        println!(" -> {:?} -> {:?}", graph, rvs);
        println!(" -> {:?}", ins);

        let mut q: Vec<_> = ins
            .iter()
            .enumerate()
            .filter_map(|(v, &degree)| if degree == 0 { Some(v) } else { None })
            .collect();

        let mut rst = vec![false; n];
        while let Some(v) = q.pop() {
            rst[v] = true;

            for u in rvs[v].iter().cloned() {
                ins[u] -= 1;
                if ins[u] == 0 {
                    q.push(u);
                }
            }
        }

        rst.iter()
            .enumerate()
            .filter_map(|(v, &flag)| if flag { Some(v as i32) } else { None })
            .collect()
    }
}

/// 1267m Count Servers that Communicate
struct Sol1267;

impl Sol1267 {
    pub fn count_servers(grid: Vec<Vec<i32>>) -> i32 {
        let (rows, cols) = (grid.len(), grid[0].len());

        let rn: Vec<_> = (0..rows)
            .map(|r| (0..cols).filter(|&c| grid[r][c] == 1).count())
            .collect();
        let cn: Vec<_> = (0..cols)
            .map(|c| (0..rows).filter(|&r| grid[r][c] == 1).count())
            .collect();

        println!(" -> R: {:?}   C: {:?}", rn, cn);

        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .filter(|&(r, c)| rn[r] > 1 || cn[c] > 1)
            .map(|(r, c)| grid[r][c])
            .sum()
    }
}

/// 1368h Minimum Cost to Make at Least One Valid Path in a Grid
struct Sol1368;

impl Sol1368 {
    pub fn min_cost(grid: Vec<Vec<i32>>) -> i32 {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let (rows, cols) = (grid.len(), grid[0].len());
        let mut costs = vec![vec![i32::MAX; cols]; rows];

        let mut pq = BinaryHeap::new();
        pq.push(Reverse((0, 0i32, 0i32)));
        costs[0][0] = 0;

        println!(" -> {:?}", pq);
        while let Some(Reverse((w, r, c))) = pq.pop() {
            if costs[r as usize][c as usize] < w {
                continue;
            }

            print!(" -> {:?} :: ", (w, r, c));

            [(0, 1), (0, -1), (1, 0), (-1, 0)]
                .into_iter()
                .zip(1..=4)
                .inspect(|o| print!(" {:?}", o))
                .for_each(|((dx, dy), dir)| {
                    let w = w + (dir != grid[r as usize][c as usize]) as i32;
                    let (r, c) = (r + dx as i32, c + dy as i32);
                    if 0 <= r
                        && r < rows as i32
                        && 0 <= c
                        && c < cols as i32
                        && w < costs[r as usize][c as usize]
                    {
                        costs[r as usize][c as usize] = w;
                        pq.push(Reverse((w, r, c)));
                    }
                });

            println!();
            println!(" -> {:?}", pq);
        }

        println!(" :: {}", costs[rows - 1][cols - 1]);

        costs[rows - 1][cols - 1]
    }

    fn min_cost_bfs01(grid: Vec<Vec<i32>>) -> i32 {
        use std::collections::VecDeque;

        let (rows, cols) = (grid.len(), grid[0].len());
        let mut costs = vec![vec![i32::MAX; cols]; rows];

        let mut q = VecDeque::new();

        q.push_front((0i32, 0i32));
        costs[0][0] = 0;

        println!(" -> {:?}", q);

        while let Some((r, c)) = q.pop_front() {
            let curw = costs[r as usize][c as usize];

            [(0, 1), (0, -1), (1, 0), (-1, 0)]
                .into_iter()
                .zip(1..=4)
                .for_each(|((dx, dy), dir)| {
                    let w = (grid[r as usize][c as usize] != dir) as i32;
                    let (r, c) = (r + dx, c + dy);
                    if 0 <= r
                        && r < rows as i32
                        && 0 <= c
                        && c < cols as i32
                        && curw + w < costs[r as usize][c as usize]
                    {
                        costs[r as usize][c as usize] = curw + w;
                        match w {
                            0 => q.push_front((r, c)),
                            _ => q.push_back((r, c)),
                        }
                    }
                });

            println!(" -> {:?}", q);
        }

        println!(" :: {}", costs[rows - 1][cols - 1]);

        costs[rows - 1][cols - 1]
    }
}

/// 1462m Course Schedule IV
struct Sol1462;

impl Sol1462 {
    pub fn check_if_prerequisite(
        num_courses: i32,
        prerequisites: Vec<Vec<i32>>,
        queries: Vec<Vec<i32>>,
    ) -> Vec<bool> {
        let mut graph = vec![vec![]; num_courses as usize];

        prerequisites.iter().for_each(|v| {
            graph[v[0] as usize].push(v[1] as usize);
        });

        println!(" -> {:?}", graph);

        let mut floyd_warshall = vec![vec![false; num_courses as usize]; num_courses as usize];
        prerequisites.iter().for_each(|v| {
            floyd_warshall[v[0] as usize][v[1] as usize] = true;
        });
        (0..num_courses as usize).for_each(|v| floyd_warshall[v][v] = true);

        (0..num_courses as usize).for_each(|src| {
            (0..num_courses as usize).for_each(|dst| {
                (0..num_courses as usize).for_each(|via| {
                    floyd_warshall[src][dst] |=
                        floyd_warshall[src][via] && floyd_warshall[via][dst];
                })
            })
        });

        println!(" -> {:?}", floyd_warshall);

        let mut memory = vec![vec![false; num_courses as usize]; num_courses as usize];

        (0..num_courses as usize).for_each(|src| {
            let mut q = vec![];
            q.push(src);

            while let Some(v) = q.pop() {
                memory[src][v] = true;
                graph[v as usize].iter().for_each(|&u| {
                    if !memory[src][u] {
                        q.push(u)
                    }
                });
            }
        });

        assert_eq!(memory, floyd_warshall);

        queries.into_iter().fold(vec![], |mut rst, qry| {
            rst.push(memory[qry[0] as usize][qry[1] as usize]);
            rst
        })
    }
}

/// 1765m Map of Highest Peak
struct Sol1765;

impl Sol1765 {
    pub fn highest_peak(is_water: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        use std::collections::VecDeque;

        let (rows, cols) = (is_water.len(), is_water[0].len());
        let mut q = VecDeque::new();
        let mut visited = vec![vec![false; cols]; rows];

        (0..rows).for_each(|r| {
            (0..cols).for_each(|c| {
                if is_water[r][c] == 1 {
                    q.push_back((r as i32, c as i32, 0));
                    visited[r][c] = true;
                }
            })
        });

        let mut rst = vec![vec![0; cols]; rows];
        let dirs = [-1, 0, 1, 0, -1];

        println!(" -> {:?}", q);
        while let Some((r, c, h)) = q.pop_front() {
            println!(" -> {:?}", q);

            (0..4).for_each(|i| {
                let (x, y) = (r + dirs[i], c + dirs[i + 1]);
                if 0 <= x
                    && x < rows as i32
                    && 0 <= y
                    && y < cols as i32
                    && !visited[x as usize][y as usize]
                {
                    visited[x as usize][y as usize] = true;
                    rst[x as usize][y as usize] = h + 1;
                    q.push_back((x, y, h + 1));
                }
            });
        }

        rst
    }
}

/// 2127h Maximum Employees to Be Invited to a Meeting
struct Sol2127;

impl Sol2127 {
    pub fn maximum_invitations(favorite: Vec<i32>) -> i32 {
        use std::collections::VecDeque;

        let n = favorite.len();
        let mut ins = vec![0; n];
        favorite.iter().for_each(|&f| ins[f as usize] += 1);

        let mut q = VecDeque::from(
            ins.iter()
                .enumerate()
                .filter_map(|(v, &degree)| if degree == 0 { Some(v) } else { None })
                .collect::<Vec<_>>(),
        );

        let mut depth = vec![1; n];

        while let Some(v) = q.pop_front() {
            let u = favorite[v] as usize;

            depth[u] = depth[u].max(depth[v] + 1);
            ins[u] -= 1;
            if ins[u] == 0 {
                q.push_back(u);
            }
        }

        let (mut lcycle, mut tcycle) = (0, 0);
        (0..n).for_each(|v| {
            if ins[v] != 0 {
                let (mut l, mut cur) = (0, v);
                while ins[cur] > 0 {
                    ins[cur] = 0;
                    l += 1;
                    cur = favorite[cur] as usize;
                }

                match l {
                    2 => tcycle += depth[v] + depth[favorite[v] as usize],
                    _ => lcycle = lcycle.max(l),
                }
            }
        });

        println!(" :: {:?}", (lcycle, tcycle));

        lcycle.max(tcycle)
    }
}

/// 2658m Maximum Number of Fish in a Grid
struct Sol2658;

impl Sol2658 {
    pub fn find_max_fish(grid: Vec<Vec<i32>>) -> i32 {
        println!("* {:?}", grid);

        let (rows, cols) = (grid.len(), grid[0].len());
        let mut visited = vec![vec![false; cols]; rows];

        let dir = [-1, 0, 1, 0, -1];
        let mut xfish = 0;

        (0..rows).for_each(|r| {
            (0..cols).for_each(|c| {
                if grid[r][c] != 0 && !visited[r][c] {
                    println!("-> {:?}", (r, c));

                    let mut fish = 0;
                    let mut q = vec![];

                    q.push((r as i32, c as i32));
                    visited[r][c] = true;

                    while let Some((r, c)) = q.pop() {
                        println!(" -> {:?}", (r, c));

                        fish += grid[r as usize][c as usize];
                        xfish = xfish.max(fish);

                        (0..4).for_each(|i| {
                            let (r, c) = (r + dir[i], c + dir[i + 1]);
                            if 0 <= r
                                && r < rows as i32
                                && 0 <= c
                                && c < cols as i32
                                && grid[r as usize][c as usize] != 0
                                && !visited[r as usize][c as usize]
                            {
                                visited[r as usize][c as usize] = true;
                                q.push((r, c));
                            }
                        })
                    }
                }
            })
        });

        xfish
    }

    fn find_max_fish_recursion(grid: Vec<Vec<i32>>) -> i32 {
        fn dfs(grid: &mut Vec<Vec<i32>>, r: usize, c: usize) -> i32 {
            if grid[r][c] == 0 {
                return 0;
            }

            let mut fish = grid[r][c];
            grid[r][c] = 0;

            if r > 0 {
                fish += dfs(grid, r - 1, c);
            }
            if r + 1 < grid.len() {
                fish += dfs(grid, r + 1, c);
            }
            if c > 0 {
                fish += dfs(grid, r, c - 1);
            }
            if c + 1 < grid[0].len() {
                fish += dfs(grid, r, c + 1);
            }

            fish
        }

        let mut grid = grid;
        let mut xfish = 0;

        (0..grid.len())
            .for_each(|r| (0..grid[0].len()).for_each(|c| xfish = xfish.max(dfs(&mut grid, r, c))));

        xfish
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_2127() {
        assert_eq!(Sol2127::maximum_invitations(vec![2, 2, 1, 2]), 3);
        assert_eq!(Sol2127::maximum_invitations(vec![1, 2, 0]), 3);
        assert_eq!(Sol2127::maximum_invitations(vec![3, 0, 1, 4, 1]), 4);
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
}
