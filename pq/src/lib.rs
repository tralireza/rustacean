//! # PQ (aka Heap) :: Rusty

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

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

        println!(" -> {pq:?}");
        println!(" -> {visited:?}");

        let mut trap_water = 0;
        let dirs = [-1, 0, 1, 0, -1];

        while let Some(Reverse(cell)) = pq.pop() {
            println!(" -> {cell:?}");

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

        println!(" -> {visited:?}");

        trap_water
    }
}

/// 778 Swim in Rising Water
struct Sol778 {}

impl Sol778 {
    pub fn swim_in_water(mut grid: Vec<Vec<i32>>) -> i32 {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut pq = BinaryHeap::new();
        pq.push((Reverse(grid[0][0]), 0, 0));

        let mut t = 0;
        while let Some((Reverse(height), r, c)) = pq.pop() {
            if r + 1 == grid.len() && c + 1 == grid[r].len() {
                return t.max(height);
            }

            t = t.max(height);

            for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (r, c) = (r.wrapping_add_signed(dr), c.wrapping_add_signed(dc));
                if r < grid.len() && c < grid[r].len() && grid[r][c] != -1 {
                    pq.push((Reverse(grid[r][c]), r, c));
                    grid[r][c] = -1;
                }
            }
        }

        println!("-> {grid:?}");

        t
    }
}

/// 1046 Last Stone Weight
struct Sol1046 {}

impl Sol1046 {
    pub fn last_stone_weight(stones: Vec<i32>) -> i32 {
        use std::collections::BinaryHeap;

        let mut hvs = BinaryHeap::new();
        for stone in stones {
            hvs.push(stone);
        }

        while hvs.len() > 1 {
            let w1 = hvs.pop().unwrap();
            let w2 = hvs.pop().unwrap();
            if w1 > w2 {
                hvs.push(w1 - w2);
            }
        }

        hvs.pop().unwrap_or(0)
    }
}

/// 1792m Maximum Average Pass Ratio
struct Sol1792 {}

impl Sol1792 {
    pub fn max_average_ratio(classes: Vec<Vec<i32>>, extra_students: i32) -> f64 {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Debug)]
        struct Class {
            gain: f64, // gain: (pass+1)/(total+1) - pass/total
            pass: i32,
            total: i32,
        }

        impl Ord for Class {
            fn cmp(&self, other: &Self) -> Ordering {
                self.gain.partial_cmp(&other.gain).unwrap()
            }
        }
        impl PartialOrd for Class {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Eq for Class {}
        impl PartialEq for Class {
            fn eq(&self, other: &Self) -> bool {
                self.pass == other.pass && self.total == other.total
            }
        }

        fn gain(pass: i32, total: i32) -> f64 {
            (pass + 1) as f64 / (total + 1) as f64 - pass as f64 / total as f64
        }

        let mut pq: BinaryHeap<Class> = classes
            .iter()
            .map(|v| (v[0], v[1]))
            .map(|(pass, total)| Class {
                gain: gain(pass, total),
                pass,
                total,
            })
            .collect();
        println!("-> {pq:?}");

        for _ in 0..extra_students {
            if let Some(Class {
                gain: _,
                pass,
                total,
            }) = pq.pop()
            {
                pq.push(Class {
                    gain: gain(pass + 1, total + 1),
                    pass: pass + 1,
                    total: total + 1,
                });
            }
        }
        println!("-> {pq:?}");

        pq.iter()
            .map(
                |&Class {
                     gain: _,
                     pass,
                     total,
                 }| pass as f64 / total as f64,
            )
            .sum::<f64>()
            / pq.len() as f64
    }
}

/// 1912h Design Movie Rental System
#[derive(Debug)]
struct MovieRentingSystem1912 {
    prices: std::collections::HashMap<(i32, i32), i32>, // (Shop, Movie) -> Price
    movies: std::collections::HashMap<i32, std::collections::BTreeSet<(i32, i32)>>, // Movie -> (Price, Shop)
    pq: std::collections::BTreeSet<(i32, i32, i32)>, // (Price, Shop, Movie)
}

impl MovieRentingSystem1912 {
    /// 0 <= Shop < 3*10^5
    /// 1 <= Movie, Price <= 10^4
    #[allow(unused_variables)]
    fn new(n: i32, entries: Vec<Vec<i32>>) -> Self {
        MovieRentingSystem1912 {
            prices: entries.iter().map(|v| ((v[0], v[1]), v[2])).collect(),
            movies: entries.iter().map(|v| (v[0], v[1], v[2])).fold(
                HashMap::new(),
                |mut movies, (shop, movie, price)| {
                    movies
                        .entry(movie)
                        .or_insert_with(BTreeSet::new)
                        .insert((price, shop));

                    movies
                },
            ),
            pq: BTreeSet::new(),
        }
    }

    fn search(&self, movie: i32) -> Vec<i32> {
        self.movies.get(&movie).map_or(vec![], |movies| {
            movies.iter().take(5).map(|&(_, shop)| shop).collect()
        })
    }

    fn rent(&mut self, shop: i32, movie: i32) {
        if let Some(&price) = self.prices.get(&(shop, movie)) {
            self.movies
                .get_mut(&movie)
                .map(|movies| movies.remove(&(price, shop)));

            self.pq.insert((price, shop, movie));
        }
    }

    fn drop(&mut self, shop: i32, movie: i32) {
        if let Some(&price) = self.prices.get(&(shop, movie)) {
            self.movies
                .get_mut(&movie)
                .map(|movies| movies.insert((price, shop)));

            self.pq.remove(&(price, shop, movie));
        }
    }

    fn report(&self) -> Vec<Vec<i32>> {
        self.pq
            .iter()
            .take(5)
            .map(|&(_, shop, movie)| vec![shop, movie])
            .collect()
    }
}

/// 2231 Largest Number After Digit Swaps by Parity
struct Sol2231 {}

impl Sol2231 {
    pub fn largest_integer(mut num: i32) -> i32 {
        use std::collections::BinaryHeap;

        let mut qs = [BinaryHeap::new(), BinaryHeap::new()];
        let mut pars = vec![];

        while num > 0 {
            let parity = (num % 10) as usize & 1;

            pars.push(parity);
            qs[parity].push(num % 10);

            num /= 10;
        }
        println!("-> {qs:?}");

        let mut x = 0;
        for &parity in pars.iter().rev() {
            if let Some(digit) = qs[parity].pop() {
                x = 10 * x + digit;
            }
        }

        x
    }
}

/// 2353m Design a Food Rating System
#[derive(Debug)]
struct FoodRatings2353 {
    data: HashMap<String, BinaryHeap<(i32, Reverse<String>)>>,
    f_cuisine: HashMap<String, String>,
    f_rating: HashMap<String, i32>,
}

impl FoodRatings2353 {
    fn new(foods: Vec<String>, cuisines: Vec<String>, ratings: Vec<i32>) -> Self {
        FoodRatings2353 {
            data: cuisines
                .iter()
                .enumerate()
                .fold(HashMap::new(), |mut data, (i, cuisine)| {
                    data.entry(cuisine.to_string())
                        .and_modify(|pq| pq.push((ratings[i], Reverse(foods[i].clone()))))
                        .or_insert(BinaryHeap::from([(ratings[i], Reverse(foods[i].clone()))]));

                    data
                }),
            f_cuisine: foods
                .iter()
                .enumerate()
                .map(|(i, food)| (food.clone(), cuisines[i].clone()))
                .collect(),
            f_rating: foods
                .iter()
                .zip(ratings.iter())
                .map(|(food, &rating)| (food.clone(), rating))
                .collect(),
        }
    }

    fn change_rating(&mut self, food: String, new_rating: i32) {
        if let Some(pq) = self.data.get_mut(&self.f_cuisine[&food]) {
            pq.push((new_rating, Reverse(food.to_string())));
            self.f_rating.insert(food.to_string(), new_rating);
        }
    }

    fn highest_rated(&mut self, cuisine: String) -> String {
        if let Some(pq) = self.data.get_mut(&cuisine) {
            while let Some((rating, Reverse(food))) = pq.peek() {
                if *rating == self.f_rating[food] {
                    return food.clone();
                } else {
                    pq.pop(); // old "Rating", throw away!
                }
            }
        }

        panic!()
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
        while let Some(&Reverse(x)) = pq.peek() {
            if x < k as usize {
                pq.pop();
                if let Some(Reverse(y)) = pq.pop() {
                    pq.push(Reverse(x.min(y) * 2 + x.max(y)));
                }
            } else {
                break;
            }
            ops += 1;
        }

        println!("-> {pq:?}");

        ops
    }
}

/// 3362m Zero Array Transformation III
struct Sol3362;

impl Sol3362 {
    /// 1 <= N, Q <= 10^5
    /// 0 <= N_ij <= 10^5
    pub fn max_removal(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> i32 {
        use std::collections::BinaryHeap;

        #[derive(Debug)]
        struct Fenwick {
            tree: Vec<i32>,
        }

        impl Fenwick {
            fn new(size: usize) -> Self {
                Fenwick {
                    tree: vec![0; size + 1],
                }
            }

            fn update(&mut self, mut i: usize, diff: i32) {
                while i < self.tree.len() {
                    self.tree[i] += diff;
                    i += i & (!i + 1);
                }
            }

            fn query(&self, mut i: usize) -> i32 {
                let mut r = 0;
                while i > 0 {
                    r += self.tree[i];
                    i -= i & (!i + 1);
                }
                r
            }
        }

        let mut fwt = Fenwick::new(nums.len() + 2);
        for query in &queries {
            fwt.update(query[0] as usize + 1, 1);
            fwt.update(query[1] as usize + 1 + 1, -1);
        }
        println!(
            "-> {:?} {fwt:?}",
            (1..=nums.len()).map(|i| fwt.query(i)).collect::<Vec<_>>()
        );

        let mut queries = queries;
        queries.sort_by(|q1, q2| {
            if q1[0] == q2[0] {
                q2[1].cmp(&q1[1])
            } else {
                q1[0].cmp(&q2[0])
            }
        });
        println!("-> {queries:?}");

        let mut pq = BinaryHeap::new();

        let (mut diff, mut diffs) = (0, vec![0; nums.len() + 1]);
        let mut q = 0;
        for (i, &n) in nums.iter().enumerate() {
            diff += diffs[i];

            while q < queries.len() && queries[q][0] as usize == i {
                pq.push(queries[q][1]);
                q += 1;
            }

            while diff < n {
                match pq.peek() {
                    Some(&right) if right >= i as i32 => {
                        diff += 1;
                        diffs[right as usize + 1] -= 1;

                        pq.pop();
                    }
                    _ => break,
                }
            }

            if diff < n {
                return -1;
            }
        }

        pq.len() as _
    }
}

/// 3408m Design Task Manager
#[derive(Debug)]
struct TaskManager3408 {
    oset: BTreeSet<(i32, i32, i32)>, // (Priority, Task, User)
    task: HashMap<i32, (i32, i32)>,  // Task: -> (User, Priority)
}

impl TaskManager3408 {
    fn new(tasks: Vec<Vec<i32>>) -> Self {
        TaskManager3408 {
            oset: tasks
                .iter()
                .map(|task| (task[2], task[1], task[0]))
                .collect(),
            task: tasks
                .iter()
                .map(|task| (task[1], (task[0], task[2])))
                .collect(),
        }
    }

    fn add(&mut self, user_id: i32, task_id: i32, priority: i32) {
        self.oset.insert((priority, task_id, user_id));
        self.task.insert(task_id, (user_id, priority));
    }

    fn edit(&mut self, task_id: i32, new_priority: i32) {
        if let Some((user, priority)) = self.task.get_mut(&task_id)
            && self.oset.take(&(*priority, task_id, *user)).is_some()
        {
            self.oset.insert((new_priority, task_id, *user));
            *priority = new_priority;
        }
    }

    fn rmv(&mut self, task_id: i32) {
        if let Some((user, priority)) = self.task.remove(&task_id) {
            self.oset.take(&(priority, task_id, user));
        }
    }

    fn exec_top(&mut self) -> i32 {
        if let Some((_, task, user)) = self.oset.pop_last() {
            self.task.remove(&task);
            return user;
        }
        -1
    }
}

#[cfg(test)]
mod tests;
