//! # Hashing

/// 599 Minimum Index Sum of Two Lists
struct Sol599;

impl Sol599 {
    pub fn find_restaurant(list1: Vec<String>, list2: Vec<String>) -> Vec<String> {
        use std::collections::{BTreeMap, HashMap};

        let hm1: HashMap<_, _> = list1.iter().enumerate().map(|(i, s)| (s, i)).collect();

        let mut m: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for (i, s) in list2.iter().enumerate() {
            if hm1.contains_key(s) {
                m.entry(i + hm1[s])
                    .and_modify(|v| v.push(s.to_string()))
                    .or_insert(vec![s.to_string()]);
            }
        }

        println!("-> {:?}", m);

        match m.first_key_value() {
            Some((_, v)) => v.to_vec(),
            _ => vec![],
        }
    }
}

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

/// 2342m Max Sum of a Pair With Equal Sum of Digits
struct Sol2342;

impl Sol2342 {
    pub fn maximum_sum(nums: Vec<i32>) -> i32 {
        let mut mem = [0; 9 * 9 + 1];

        let mut rst = -1;
        nums.iter().for_each(|&n| {
            let (mut dsum, mut x) = (0, n as usize);
            while x > 0 {
                dsum += x % 10;
                x /= 10;
            }

            if mem[dsum] > 0 {
                rst = rst.max(mem[dsum] + n);
            }
            mem[dsum] = mem[dsum].max(n);
        });

        println!("-> {:?}", mem);

        rst
    }
}

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

/// 2349m Design a Number Container System
struct NumberContainers {
    nmap: HashMap<i32, BinaryHeap<Reverse<i32>>>, // number -> PQ(index...)
    minds: HashMap<i32, i32>,                     // index -> number

    nset: HashMap<i32, BTreeSet<i32>>, // with BTreeSet :: number -> TreeSet(index...)
}

impl NumberContainers {
    fn new() -> Self {
        NumberContainers {
            nmap: HashMap::new(),
            minds: HashMap::new(),

            nset: HashMap::new(),
        }
    }

    fn change(&mut self, index: i32, number: i32) {
        if let Some(&prv) = self.minds.get(&index) {
            if let Some(nset) = self.nset.get_mut(&prv) {
                nset.remove(&index);
                if nset.is_empty() {
                    self.nset.remove(&prv);
                }
            }
        }
        self.nset.entry(number).or_default().insert(index);

        self.minds.insert(index, number);
        self.nmap
            .entry(number)
            .and_modify(|pq| pq.push(Reverse(index)))
            .or_insert(BinaryHeap::from([Reverse(index)]));

        println!("-> {:?} {:?} {:?}", self.minds, self.nmap, self.nset);
    }

    fn find(&mut self, number: i32) -> i32 {
        println!(
            " ? {:?}",
            if let Some(nset) = self.nset.get(&number) {
                nset.first()
            } else {
                Some(&-1)
            }
        );

        if let Some(pq) = self.nmap.get_mut(&number) {
            while let Some(&Reverse(i)) = pq.peek() {
                if let Some(&n) = self.minds.get(&i) {
                    if n == number {
                        return i;
                    }

                    pq.pop();
                }
            }
        }

        -1
    }
}

/// 2364m Count Number of Bad Pairs
struct Sol2364;

impl Sol2364 {
    pub fn count_bad_pairs(nums: Vec<i32>) -> i64 {
        use std::collections::HashMap;

        let mut hmap = HashMap::new();
        nums.iter().enumerate().fold(
            nums.len() as i64 * (nums.len() as i64 - 1) / 2, // total pairs: nCk n!/k!(n-k)!
            |r, (i, v)| {
                hmap.entry(v - i as i32)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                match hmap.get(&(v - i as i32)) {
                    Some(count) => r - count + 1,
                    _ => r,
                }
            },
        )
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

        println!("|| {:?}", limit);

        let (mut colors, mut balls) = (HashMap::new(), HashMap::new());
        let mut rst = vec![];

        queries.iter().for_each(|v| {
            let (ball, color) = (v[0] as usize, v[1] as usize);

            if let Some(&prv) = balls.get(&ball) {
                colors.entry(prv).and_modify(|f| *f -= 1);
                if let Some(&f) = colors.get(&prv) {
                    if f == 0 {
                        colors.remove(&prv);
                    }
                }
            }

            balls
                .entry(ball)
                .and_modify(|f| *f = color)
                .or_insert(color);
            colors.entry(color).and_modify(|f| *f += 1).or_insert(1);

            println!("-> {:?} ~ {:?}", balls, colors);

            rst.push(colors.len() as i32);
        });

        println!(":: {:?}", rst);

        rst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_599() {
        assert_eq!(
            Sol599::find_restaurant(
                vec![
                    "Shogun".to_string(),
                    "Tapioca Express".to_string(),
                    "Burger King".to_string(),
                    "KFC".to_string()
                ],
                vec![
                    "Piatti".to_string(),
                    "The Grill at Torrey Pines".to_string(),
                    "Hungry Hunter Steakhouse".to_string(),
                    "Shogun".to_string()
                ]
            ),
            vec!["Shogun".to_string()]
        );
        assert_eq!(
            Sol599::find_restaurant(
                vec!["happy".to_string(), "sad".to_string(), "good".to_string(),],
                vec!["sad".to_string(), "happy".to_string(), "good".to_string(),]
            ),
            vec!["sad".to_string(), "happy".to_string()]
        );
    }

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
    fn test_2342() {
        assert_eq!(Sol2342::maximum_sum(vec![18, 43, 36, 13, 7]), 54);
        assert_eq!(Sol2342::maximum_sum(vec![10, 12, 19, 14]), -1);

        assert_eq!(
            Sol2342::maximum_sum(vec![
                279, 169, 463, 252, 94, 455, 423, 315, 288, 64, 494, 337, 409, 283, 283, 477, 248,
                8, 89, 166, 188, 186, 128
            ]),
            872
        );
    }

    #[test]
    fn test_2349() {
        let mut nc = NumberContainers::new();
        assert_eq!(nc.find(10), -1);
        for i in [2, 1, 3, 5] {
            nc.change(i, 10);
        }
        assert_eq!(nc.find(10), 1);
        nc.change(1, 20);
        assert_eq!(nc.find(10), 2);
    }

    #[test]
    fn test_2364() {
        assert_eq!(Sol2364::count_bad_pairs(vec![4, 1, 3, 3]), 5);
        assert_eq!(Sol2364::count_bad_pairs(vec![1, 2, 3, 4, 5]), 0);
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
