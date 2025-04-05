//! # BIT Binary Indexed Tree, Segment Tree, Ordered Set

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

        println!(":: {:?}", skyline);

        skyline
    }
}

#[cfg(test)]
mod tests;
