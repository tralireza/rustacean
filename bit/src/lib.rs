//! # BIT Binary Index Tree, Segment Tree, Ordered Set

/// 218h The Skyline Problems
struct Sol218;

impl Sol218 {
    pub fn get_skyline(buildings: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        use std::collections::{BTreeSet, BinaryHeap};

        println!("|| {:?}", buildings);

        let mut sweep = BTreeSet::new();
        for building in buildings.iter() {
            sweep.insert(building[0]);
            sweep.insert(building[1]);
        }

        println!("-> {:?}", sweep);

        let mut pq = BinaryHeap::new();

        let (mut i, mut h) = (0, 0);
        let mut skyline = vec![];

        for &x in sweep.iter() {
            while i < buildings.len() && buildings[i][0] == x {
                pq.push((buildings[i][2], buildings[i][0], buildings[i][1]));
                i += 1;
            }

            while let Some(t) = pq.peek() {
                if t.2 <= x {
                    pq.pop();
                } else {
                    break;
                }
            }

            if let Some(t) = pq.peek() {
                if t.0 != h {
                    skyline.push(vec![x, t.0]);
                    h = t.0;
                }
            } else {
                skyline.push(vec![x, 0]);
                h = 0;
            }
        }

        skyline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_218() {
        assert_eq!(
            Sol218::get_skyline(vec![
                vec![2, 9, 10],
                vec![3, 7, 15],
                vec![5, 12, 12],
                vec![15, 20, 10],
                vec![19, 24, 8]
            ]),
            vec![
                vec![2, 10],
                vec![3, 15],
                vec![7, 12],
                vec![12, 0],
                vec![15, 10],
                vec![20, 8],
                vec![24, 0]
            ]
        );

        assert_eq!(
            Sol218::get_skyline(vec![vec![2, 9, 10], vec![12, 15, 10]]),
            vec![vec![2, 10], vec![9, 0], vec![12, 10], vec![15, 0]]
        );
    }
}
