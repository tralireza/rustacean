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
            print!(" -> {:?} :: ", (w, r, c));
            if costs[r as usize][c as usize] < w {
                continue;
            }

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

        println!(" :: {}", costs[rows - 2][cols - 1]);

        costs[rows - 1][cols - 1]
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
        assert_eq!(
            Sol1368::min_cost(vec![
                vec![1, 1, 1, 1],
                vec![2, 2, 2, 2],
                vec![1, 1, 1, 1],
                vec![2, 2, 2, 2]
            ]),
            3
        );
        assert_eq!(
            Sol1368::min_cost(vec![vec![1, 1, 3], vec![3, 2, 2], vec![1, 1, 4]]),
            0
        );
        assert_eq!(Sol1368::min_cost(vec![vec![1, 2], vec![4, 3]]), 1);
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
}
