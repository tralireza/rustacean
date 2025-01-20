//! # PQ (aka Heap) :: Rusty

/// 407h Trapping Rain Water II
struct Sol407;

impl Sol407 {
    pub fn trap_rain_water(height_map: Vec<Vec<i32>>) -> i32 {
        let (rows, cols) = (height_map.len(), height_map[0].len());

        #[derive(Debug, PartialEq, Eq, Ord, PartialOrd)]
        struct Cell {
            height: i32,
            r: i32,
            c: i32,
        }

        let mut visited = vec![vec![false; cols]; rows];

        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut pq = BinaryHeap::new();
        for r in 0..rows {
            for c in 0..cols {
                if r == 0 || c == 0 || r == rows - 1 || c == cols - 1 {
                    pq.push(Reverse(Cell {
                        height: height_map[r][c],
                        r: r as i32,
                        c: c as i32,
                    }));

                    visited[r][c] = true;
                }
            }
        }

        println!(" -> {:?}", pq);
        println!(" -> {:?}", visited);

        let mut trap_water = 0;
        let dirs = vec![-1, 0, 1, 0, -1];

        while let Some(Reverse(cell)) = pq.pop() {
            println!(" -> {:?}", cell);

            (0..4).for_each(|i| {
                let (r, c) = (cell.r + dirs[i], cell.c + dirs[i + 1]);
                if 0 <= r
                    && r < rows as i32
                    && 0 <= c
                    && c < cols as i32
                    && !visited[r as usize][c as usize]
                {
                    visited[r as usize][c as usize] = true;

                    let h = height_map[r as usize][c as usize];
                    if h < cell.height {
                        trap_water += cell.height - h;
                    }

                    pq.push(Reverse(Cell {
                        height: h.max(cell.height),
                        r: r,
                        c: c,
                    }));
                }
            });
        }

        println!(" -> {:?}", visited);

        trap_water
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_407() {
        assert_eq!(
            Sol407::trap_rain_water(vec![
                vec![1, 4, 3, 1, 3, 2],
                vec![3, 2, 1, 3, 2, 4],
                vec![2, 3, 3, 2, 3, 1]
            ]),
            4
        );
        assert_eq!(
            Sol407::trap_rain_water(vec![
                vec![3, 3, 3, 3, 3],
                vec![3, 2, 2, 2, 3],
                vec![3, 2, 1, 2, 3],
                vec![3, 2, 2, 2, 3],
                vec![3, 3, 3, 3, 3]
            ]),
            10
        );
    }
}
