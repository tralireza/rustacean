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
        let dirs = [-1, 0, 1, 0, -1];

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
                        r,
                        c,
                    }));
                }
            });
        }

        println!(" -> {:?}", visited);

        trap_water
    }
}

/// 3066m Minimum Operations to Exceed Threshold Value II
struct Sol3066;

impl Sol3066 {
    pub fn min_operations(nums: Vec<i32>, k: i32) -> i32 {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut pq = BinaryHeap::new();
        for n in nums {
            pq.push(Reverse(n as usize));
        }

        let mut ops = 0;
        while let Some(&Reverse(n)) = pq.peek() {
            if n < k as usize {
                if let Some(Reverse(x)) = pq.pop() {
                    if let Some(Reverse(y)) = pq.pop() {
                        pq.push(Reverse(x.min(y) * 2 + x.max(y)));
                    }
                }
            } else {
                break;
            }
            ops += 1;
        }

        ops
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

    #[test]
    fn test_3066() {
        assert_eq!(Sol3066::min_operations(vec![2, 11, 10, 1, 3], 10), 2);
        assert_eq!(Sol3066::min_operations(vec![1, 1, 2, 4, 9], 20), 4);
        assert_eq!(
            Sol3066::min_operations(vec![999999999, 999999999, 999999999], 1000000000),
            2
        );
    }
}
