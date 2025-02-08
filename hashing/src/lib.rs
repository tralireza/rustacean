//! # Hashing

/// 1726m Tuple With Same Product
struct Sol1726;

impl Sol1726 {
    pub fn tuple_same_product(nums: Vec<i32>) -> i32 {
        use std::collections::HashMap;

        let mut fqs = HashMap::new();
        for i in 0..nums.len() {
            for j in i + 1..nums.len() {
                fqs.entry(nums[i] * nums[j])
                    .and_modify(|f| *f += 1)
                    .or_insert(1);
            }
        }

        println!("-> {:?}", fqs);

        fqs.into_values()
            .filter(|&f| f > 1)
            .map(|f| (f - 1) * f / 2) // nCk :: n!/k!(n-k)!
            .sum::<i32>()
            * 8
    }
}

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

/// 3160m Find the Number of Distinct Colors Among the Balls
struct Sol3160;

impl Sol3160 {
    pub fn query_results(limit: i32, queries: Vec<Vec<i32>>) -> Vec<i32> {
        use std::collections::HashMap;

        let (mut cm, mut balls) = (HashMap::new(), HashMap::new());
        let mut rst = vec![];

        queries.iter().for_each(|v| {
            let (b, c) = (v[0] as usize, v[1] as usize);

            if let Some(&prv) = balls.get(&b) {
                cm.entry(prv).and_modify(|f| *f -= 1);
                if let Some(&f) = cm.get(&prv) {
                    if f == 0 {
                        cm.remove(&prv);
                    }
                }
            }

            balls.entry(b).and_modify(|f| *f = c).or_insert(c);
            cm.entry(c).and_modify(|f| *f += 1).or_insert(1);

            println!("-> {:?} ~ {:?}", balls, cm);

            rst.push(cm.len() as i32);
        });

        println!(":: {:?}", rst);

        rst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1726() {
        assert_eq!(Sol1726::tuple_same_product(vec![2, 3, 4, 6]), 8);
        assert_eq!(Sol1726::tuple_same_product(vec![1, 2, 4, 5, 10]), 16);
    }

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

    #[test]
    fn test_3160() {
        assert_eq!(
            Sol3160::query_results(4, vec![vec![1, 4], vec![2, 5], vec![1, 3], vec![3, 4]]),
            vec![1, 2, 2, 3]
        );
        assert_eq!(
            Sol3160::query_results(
                4,
                vec![vec![0, 1], vec![1, 2], vec![2, 2], vec![3, 4], vec![4, 5]]
            ),
            vec![1, 2, 2, 3, 4]
        );
    }
}
