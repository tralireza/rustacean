//! # Hashing

/// 1790 Check if One String Swap Can Make Strings Equal
struct Sol1790;

impl Sol1790 {
    pub fn are_almost_equal(s1: String, s2: String) -> bool {
        use std::collections::HashMap;

        let diffv = s1
            .chars()
            .zip(s2.chars())
            .filter(|(c1, c2)| c1 != c2)
            .take(3)
            .collect::<Vec<_>>();

        println!(
            ":: {} ~ {:?}",
            diffv.is_empty()
                || diffv.len() == 2 && diffv[0].0 == diffv[1].1 && diffv[0].1 == diffv[1].0,
            diffv
        );

        let (mut hm1, mut hm2) = (HashMap::new(), HashMap::new());
        let diffs =
            s1.chars()
                .zip(s2.chars())
                .filter(|(c1, c2)| c1 != c2)
                .fold(0, |r, (c1, c2)| {
                    hm1.entry(c1).and_modify(|f| *f += 1).or_insert(1);
                    hm2.entry(c2).and_modify(|f| *f += 1).or_insert(1);
                    r + 1
                });

        println!("-> {:?} | {:?}", hm1, hm2);

        if diffs > 2 {
            return false;
        }

        for (chr, f1) in hm1 {
            if let Some(&f2) = hm2.get(&chr) {
                if f2 != f1 {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

/// 2661m First Completely Painted Row or Column
struct Sol2661;

impl Sol2661 {
    pub fn first_complete_index(arr: Vec<i32>, mat: Vec<Vec<i32>>) -> i32 {
        use std::collections::HashMap;

        let (rows, cols) = (mat.len(), mat[0].len());
        let mut hm = HashMap::new();

        (0..rows).for_each(|r| {
            (0..cols).for_each(|c| {
                hm.insert(mat[r][c], (r, c));
            })
        });

        let (mut rcount, mut ccount) = (vec![0; rows], vec![0; cols]);

        arr.iter()
            .take_while(|n| match hm.get(n) {
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
    fn test_1790() {
        assert_eq!(
            Sol1790::are_almost_equal("bank".to_string(), "kanb".to_string()),
            true
        );
        assert_eq!(
            Sol1790::are_almost_equal("attack".to_string(), "defend".to_string()),
            false
        );
        assert_eq!(
            Sol1790::are_almost_equal("kelb".to_string(), "kelb".to_string()),
            true
        );

        assert_eq!(
            Sol1790::are_almost_equal("qgqeg".to_string(), "gqgeq".to_string()),
            false
        );
    }

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
