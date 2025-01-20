//! # Hashing

/// 2661m First Completely Painted Row or Column
struct Sol2661;

impl Sol2661 {
    pub fn first_complete_index(arr: Vec<i32>, mat: Vec<Vec<i32>>) -> i32 {
        use std::collections::HashMap;

        let (rows, cols) = (mat.len(), mat[0].len());
        let mut hm = HashMap::new();

        for r in 0..rows {
            for c in 0..cols {
                hm.insert(mat[r][c], (r, c));
            }
        }

        let (mut rcount, mut ccount) = (vec![0; rows], vec![0; cols]);

        arr.iter()
            .take_while(|n| match hm.get(&n) {
                Some(&(r, c)) => {
                    rcount[r] += 1;
                    ccount[c] += 1;
                    rcount[r] != cols && ccount[c] != rows
                }
                _ => true,
            })
            .count() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2661() {
        assert_eq!(
            Sol2661::first_complete_index(vec![1, 3, 4, 2], vec![vec![1, 4], vec![2, 3]]),
            2
        );
        assert_eq!(
            Sol2661::first_complete_index(
                vec![2, 8, 7, 4, 1, 3, 5, 6, 9],
                vec![vec![3, 2, 5], vec![1, 4, 6], vec![8, 7, 9]]
            ),
            3
        );
    }
}
