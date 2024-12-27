//! Graph DFS BFS ...

/// 3203h Find Minimum Diameter After Merging Two Trees
struct Solution3203;

impl Solution3203 {
    pub fn minimum_diameter_after_merge(edges1: Vec<Vec<i32>>, edges2: Vec<Vec<i32>>) -> i32 {
        fn ladj(edges: &Vec<Vec<i32>>) -> Vec<Vec<usize>> {
            let mut ladj = vec![vec![]; edges.len() + 1];
            for e in edges {
                ladj[e[0] as usize].push(e[1] as usize);
                ladj[e[1] as usize].push(e[0] as usize);
            }

            ladj
        }

        let (g1, g2) = (ladj(&edges1), ladj(&edges2));

        fn diameter(g: &Vec<Vec<usize>>) -> i32 {
            fn search(g: &Vec<Vec<usize>>, src: usize) -> (usize, i32) {
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(src);

                let mut visited = vec![false; g.len()];
                visited[src] = true;

                let (mut fnode, mut xdist) = (src, 0);

                while !queue.is_empty() {
                    for _ in 0..queue.len() {
                        if let Some(v) = queue.pop_front() {
                            fnode = v;
                            for &u in &g[v] {
                                if !visited[u] {
                                    visited[u] = true;
                                    queue.push_back(u);
                                }
                            }
                        }
                    }
                    xdist += 1;
                }

                (fnode, xdist - 1)
            }

            let (fnode, _) = search(g, 0);
            let (_, xdist) = search(g, fnode);

            xdist
        }

        let (d1, d2) = (diameter(&g1), diameter(&g2));

        d1.max(d2).max((d1 + 1) / 2 + (d2 + 1) / 2 + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution3203() {
        assert_eq!(
            Solution3203::minimum_diameter_after_merge(
                vec![vec![0, 1], vec![0, 2], vec![0, 3]],
                vec![vec![0, 1]]
            ),
            3
        );
        assert_eq!(
            Solution3203::minimum_diameter_after_merge(
                vec![
                    vec![0, 1],
                    vec![0, 2],
                    vec![0, 3],
                    vec![2, 4],
                    vec![2, 5],
                    vec![3, 6],
                    vec![2, 7]
                ],
                vec![
                    vec![0, 1],
                    vec![0, 2],
                    vec![0, 3],
                    vec![2, 4],
                    vec![2, 5],
                    vec![3, 6],
                    vec![2, 7]
                ]
            ),
            5
        );
    }
}
