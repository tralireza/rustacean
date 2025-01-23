//! # Graph (DFS, BFS)

/// 1267m Count Servers that Communicate
struct Sol1267;

impl Sol1267 {
    pub fn count_servers(grid: Vec<Vec<i32>>) -> i32 {
        let (rows, cols) = (grid.len(), grid[0].len());

        let (mut xcount, mut ycount) = (vec![0; cols], vec![0; rows]);
        (0..rows).for_each(|x| {
            (0..cols).for_each(|y| {
                if grid[x][y] == 1 {
                    xcount[y] += 1;
                    ycount[x] += 1;
                }
            });
        });

        println!(" -> {:?} {:?}", xcount, ycount);

        let mut servers = 0;
        (0..rows).for_each(|x| {
            (0..cols).for_each(|y| {
                if grid[x][y] == 1 && (xcount[y] > 1 || ycount[x] > 1) {
                    servers += 1;
                }
            })
        });

        servers
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
    fn test_1267() {
        assert_eq!(Sol1267::count_servers(vec![vec![1, 0], vec![0, 1]]), 0);
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
