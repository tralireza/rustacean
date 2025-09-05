//! # Enumeration

/// 2749m Minimum Operations to Make Integer Zero
struct Sol2749 {}

impl Sol2749 {
    pub fn make_the_integer_zero(num1: i32, num2: i32) -> i32 {
        let mut x: i64 = 1;
        loop {
            let target = num1 as i64 - x * num2 as i64;

            if target < x {
                return -1;
            }
            if x >= target.count_ones() as i64 {
                return x as _;
            }

            x += 1;
        }
    }
}

/// 3025m Find the Number of Ways to Place People I
struct Sol3025 {}

impl Sol3025 {
    pub fn number_of_pairs(points: Vec<Vec<i32>>) -> i32 {
        (0..points.len())
            .flat_map(|i| (0..points.len()).map(move |j| (i, j)))
            .filter(|(i, j)| i != j)
            .filter(|&(i, j)| points[i][0] <= points[j][0] && points[i][1] >= points[j][1])
            .fold(0, |pairs, (i, j)| {
                if points
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != i && k != j)
                    .map(|(_, point)| (point[0], point[1]))
                    .inspect(|p| println!("-> {i} {j} {p:?}"))
                    .any(|(x, y)| {
                        (x >= points[i][0] && x <= points[j][0])
                            && (y <= points[i][1] && y >= points[j][1])
                    })
                {
                    pairs
                } else {
                    pairs + 1
                }
            })
    }
}

/// 3027h Find the Number of Ways to Place People II
struct Sol3027 {}

impl Sol3027 {
    pub fn number_of_pairs(mut points: Vec<Vec<i32>>) -> i32 {
        points.sort_by(|p, q| p[0].cmp(&q[0]).then(q[1].cmp(&p[1])));
        println!("-> {points:?}");

        points
            .iter()
            .enumerate()
            .take(points.len() - 1)
            .fold(0, |pairs, (i, p)| {
                let mut x_min = p[0] - 1;
                let mut y_min = i32::MIN;

                pairs
                    + points.iter().skip(i + 1).fold(0, |pairs, q| {
                        if q[0] > x_min && q[1] > y_min && q[1] <= p[1] {
                            x_min = q[0];
                            y_min = q[1];

                            pairs + 1
                        } else {
                            pairs
                        }
                    })
            }) as _
    }
}

/// 3197h Find the Minimum Area to Cover All Ones II
struct Sol3197 {}

impl Sol3197 {
    /// 1 <= Rows, Cols <= 30
    /// N_rc: [0, 1]
    pub fn minimum_sum(grid: Vec<Vec<i32>>) -> i32 {
        fn square_all_ones(grid: &[Vec<i32>], t: usize, b: usize, l: usize, r: usize) -> usize {
            let (mut top, mut bottom) = (usize::MAX, 0);
            let (mut left, mut right) = (usize::MAX, 0);

            for (y, row) in grid.iter().enumerate().skip(t).take(b - t + 1) {
                for (x, &g) in row.iter().enumerate().skip(l).take(r - l + 1) {
                    if g == 1 {
                        (top, bottom) = (top.min(y), bottom.max(y));
                        (left, right) = (left.min(x), right.max(x));
                    }
                }
            }

            if top <= bottom && left <= right {
                (bottom - top + 1) * (right - left + 1)
            } else {
                usize::MAX / 3
            }
        }

        fn check(grid: &[Vec<i32>]) -> usize {
            let (rows, cols) = (grid.len(), grid[0].len());

            let mut v = rows * cols;
            for r in 0..rows - 1 {
                for c in 0..cols - 1 {
                    v = v.min(
                        square_all_ones(grid, 0, r, 0, cols - 1)
                            + square_all_ones(grid, r + 1, rows - 1, 0, c)
                            + square_all_ones(grid, r + 1, rows - 1, c + 1, cols - 1),
                    );
                    v = v.min(
                        square_all_ones(grid, 0, r, 0, c)
                            + square_all_ones(grid, 0, r, c + 1, cols - 1)
                            + square_all_ones(grid, r + 1, rows - 1, 0, cols - 1),
                    );
                }
            }

            for r in 0..rows - 2 {
                for r_m in r + 1..rows - 1 {
                    v = v.min(
                        square_all_ones(grid, 0, r, 0, cols - 1)
                            + square_all_ones(grid, r + 1, r_m, 0, cols - 1)
                            + square_all_ones(grid, r_m + 1, rows - 1, 0, cols - 1),
                    );
                }
            }

            v
        }

        fn rotate90ccw(grid: &[Vec<i32>]) -> Vec<Vec<i32>> {
            let (rows, cols) = (grid.len(), grid[0].len());
            let mut rotated = vec![vec![0; rows]; cols];

            for (r, row) in grid.iter().enumerate() {
                for (c, &g) in row.iter().enumerate() {
                    rotated[cols - c - 1][r] = g;
                }
            }

            for row in grid {
                println!(" {row:?}");
            }
            println!("--|CCW:90|-> ");
            for row in &rotated {
                println!(" {row:?}");
            }

            rotated
        }

        check(&grid).min(check(&rotate90ccw(&grid))) as _
    }
}

#[cfg(test)]
mod tests;
