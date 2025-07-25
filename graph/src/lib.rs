//! # Graph (DFS, BFS, Topological Sort, Kahn's)

#![feature(let_chains)]

/// 684m Redundant Connection
struct Sol684;

impl Sol684 {
    pub fn find_redundant_connection(edges: Vec<Vec<i32>>) -> Vec<i32> {
        println!("* {:?}", edges);

        #[derive(Debug)]
        struct DJSet {
            parent: Vec<usize>,
            rank: Vec<usize>,
        }

        impl DJSet {
            fn new(count: usize) -> Self {
                DJSet {
                    parent: Vec::from_iter(0..=count),
                    rank: vec![0; count + 1],
                }
            }

            fn find(&mut self, v: usize) -> usize {
                if v != self.parent[v] {
                    self.parent[v] = self.find(self.parent[v]);
                }
                self.parent[v]
            }

            fn union(&mut self, u: usize, v: usize) -> bool {
                let (u, v) = (self.find(u), self.find(v));
                match u == v {
                    true => false,
                    false => {
                        match self.rank[u] > self.rank[v] {
                            true => self.parent[v] = u,
                            _ => {
                                self.parent[u] = v;
                                if self.rank[v] == self.rank[u] {
                                    self.rank[v] += 1;
                                }
                            }
                        }
                        true
                    }
                }
            }
        }

        let mut djset = DJSet::new(edges.len());
        for edge in edges {
            println!("-> {:?} ~ {:?}", edge, djset);

            if !djset.union(edge[0] as usize, edge[1] as usize) {
                return vec![edge[0], edge[1]];
            }
        }

        vec![]
    }

    fn find_redundant_connection_graph(edges: Vec<Vec<i32>>) -> Vec<i32> {
        println!("* {:?}", edges);

        let mut graph = vec![vec![]; edges.len() + 1];
        for e in &edges {
            graph[e[0] as usize].push(e[1] as usize);
            graph[e[1] as usize].push(e[0] as usize);
        }

        println!("-> {:?}", graph);

        let mut visited = vec![false; edges.len() + 1];
        let mut prv = vec![0; edges.len() + 1];
        let mut icycle = 0; // `None-Node` label for now

        let mut q = vec![1];

        while let Some(v) = q.pop() {
            visited[v] = true;

            graph[v].iter().for_each(|&u| {
                match visited[u] {
                    false => {
                        prv[u] = v;
                        q.push(u);
                    }
                    true => {
                        if prv[v] != u && icycle == 0 {
                            icycle = u; // `u` is starting a cycle
                            prv[u] = v;
                        }
                    }
                }
            });
        }

        println!("-> {:?}", prv);

        let mut cnodes = vec![false; edges.len() + 1];
        while !cnodes[icycle] {
            cnodes[icycle] = true;
            icycle = prv[icycle];
        }

        println!("-> {:?}", cnodes);

        edges
            .iter()
            .rev()
            .skip_while(|&v| !cnodes[v[0] as usize] || !cnodes[v[1] as usize])
            .take(1)
            .fold(vec![], |_, v| v.clone())
    }
}

/// 695m Max Area of Island
struct Sol695;

impl Sol695 {
    pub fn max_area_of_island(grid: Vec<Vec<i32>>) -> i32 {
        fn island(grid: &mut Vec<Vec<i32>>, r: usize, c: usize) -> i32 {
            println!("-> @ {:?} ~ {}", (r, c), grid[r][c]);

            match grid[r][c] {
                0 => 0,
                _ => {
                    grid[r][c] = 0;

                    let mut area = 1;
                    if r > 0 {
                        area += island(grid, r - 1, c);
                    }
                    if r + 1 < grid.len() {
                        area += island(grid, r + 1, c);
                    }
                    if c > 0 {
                        area += island(grid, r, c - 1);
                    }
                    if c + 1 < grid[r].len() {
                        area += island(grid, r, c + 1);
                    }
                    area
                }
            }
        }

        let mut grid = grid;

        let mut xarea = 0;
        for r in 0..grid.len() {
            for c in 0..grid[0].len() {
                xarea = xarea.max(island(&mut grid, r, c));
            }
        }

        println!(":: {}", xarea);

        xarea
    }
}

/// 753h Cracking the Safe
struct Sol753 {}
impl Sol753 {
    pub fn crack_safe(n: i32, k: i32) -> String {
        use std::collections::HashSet;

        fn dfs(v: String, k: usize, euler_path: &mut Vec<char>, visited: &mut HashSet<String>) {
            for chr in ('0'..='9').take(k) {
                let mut u: String = v.chars().collect();
                u.push(chr);

                if !visited.contains(&u) {
                    visited.insert(u.clone());
                    dfs(u.chars().skip(1).collect(), k, euler_path, visited);

                    euler_path.push(chr);
                }
            }
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut euler_path: Vec<char> = vec![];

        dfs(
            vec!['0'; n as usize - 1].iter().collect(),
            k as usize,
            &mut euler_path,
            &mut visited,
        );

        (0..n - 1).for_each(|_| euler_path.push('0'));

        println!("-> {visited:?}");

        euler_path.iter().collect()
    }
}

/// 802m Find Eventual Safe States
struct Sol802;

impl Sol802 {
    pub fn eventual_safe_nodes(graph: Vec<Vec<i32>>) -> Vec<i32> {
        let n = graph.len();

        let (mut rvs, mut ins) = (vec![vec![]; n], vec![0; n]);
        (0..n).for_each(|v| {
            graph[v].iter().for_each(|&u| {
                ins[v] += 1;
                rvs[u as usize].push(v);
            });
        });

        println!(" -> {:?} -> {:?}", graph, rvs);
        println!(" -> {:?}", ins);

        let mut q: Vec<_> = ins
            .iter()
            .enumerate()
            .filter_map(|(v, &degree)| if degree == 0 { Some(v) } else { None })
            .collect();

        let mut rst = vec![false; n];
        while let Some(v) = q.pop() {
            rst[v] = true;

            for u in rvs[v].iter().cloned() {
                ins[u] -= 1;
                if ins[u] == 0 {
                    q.push(u);
                }
            }
        }

        rst.iter()
            .enumerate()
            .filter_map(|(v, &flag)| if flag { Some(v as i32) } else { None })
            .collect()
    }
}

/// 827h Making A Large Island
struct Sol827;

impl Sol827 {
    pub fn largest_island(grid: Vec<Vec<i32>>) -> i32 {
        let mut grid = grid;
        let (rows, cols) = (grid.len(), grid[0].len());
        let dirs = [-1, 0, 1, 0, -1];

        let mut areas = vec![0, 0];
        let mut islands = 0;
        for r in 0..rows {
            for c in 0..cols {
                if grid[r][c] == 1 {
                    let mut area = 0;

                    let mut q = Vec::from([(r, c)]);
                    while let Some((r, c)) = q.pop() {
                        if grid[r][c] != 1 {
                            continue;
                        }

                        area += 1;
                        grid[r][c] = 2 + islands;

                        for i in 0..4 {
                            let (r, c) = (
                                r.wrapping_add_signed(dirs[i]),
                                c.wrapping_add_signed(dirs[i + 1]),
                            );
                            if r < rows && c < cols {
                                q.push((r, c));
                            }
                        }
                    }

                    areas.push(area);
                    islands += 1;
                }
            }
        }

        println!("-> Grid: {grid:?}");
        println!("-> Island Areas: {areas:?}");

        use std::collections::HashSet;

        let mut xarea = 0;
        for r in 0..rows {
            for c in 0..cols {
                if grid[r][c] == 0 {
                    let mut iset = HashSet::new();
                    for i in 0..4 {
                        let (r, c) = (
                            r.wrapping_add_signed(dirs[i]),
                            c.wrapping_add_signed(dirs[i + 1]),
                        );
                        if r < rows && c < cols {
                            iset.insert(grid[r][c]);
                        }
                    }
                    xarea = xarea.max(iset.iter().fold(0, |r, &v| r + areas[v as usize]) + 1);
                }
            }
        }

        match islands {
            0 => 1,
            1 if areas[2] as usize == rows * cols => areas[2],
            _ => xarea,
        }
    }
}

/// 909m Snakes and Ladders
struct Sol909 {}

impl Sol909 {
    /// 2 <= N <= 20
    pub fn snakes_and_ladders(board: Vec<Vec<i32>>) -> i32 {
        use std::collections::VecDeque;

        let n = board.len();
        let barray: Vec<i32> = board
            .into_iter()
            .rev()
            .enumerate()
            .flat_map(|(i, row)| {
                if i & 1 == 0 {
                    row
                } else {
                    row.into_iter().rev().collect()
                }
            })
            .collect();

        println!("-> Boustrophedon :: {barray:?}");

        let mut dists = vec![i32::MAX; n * n];

        let mut queue = VecDeque::from([0]);
        dists[0] = 0;
        while let Some(v) = queue.pop_front() {
            println!("-> {v:>2} {queue:?}");

            if v == n * n - 1 {
                continue;
            }

            for (mut u, &jump) in barray
                .iter()
                .enumerate()
                .take((v + 7).min(n * n))
                .skip(v + 1)
            {
                if jump != -1 {
                    u = jump as usize - 1;
                }

                if dists[u] > dists[v] + 1 {
                    dists[u] = dists[v] + 1;
                    queue.push_back(u);
                }
            }
        }

        println!("-> Distances :: {dists:?}");

        if dists[n * n - 1] == i32::MAX {
            return -1;
        }
        dists[n * n - 1]
    }
}

/// 1267m Count Servers that Communicate
struct Sol1267;

impl Sol1267 {
    pub fn count_servers(grid: Vec<Vec<i32>>) -> i32 {
        let (rows, cols) = (grid.len(), grid[0].len());

        let rn: Vec<_> = (0..rows)
            .map(|r| (0..cols).filter(|&c| grid[r][c] == 1).count())
            .collect();
        let cn: Vec<_> = (0..cols)
            .map(|c| (0..rows).filter(|&r| grid[r][c] == 1).count())
            .collect();

        println!(" -> R: {:?}   C: {:?}", rn, cn);

        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .filter(|&(r, c)| rn[r] > 1 || cn[c] > 1)
            .map(|(r, c)| grid[r][c])
            .sum()
    }
}

/// 1298h Maximum Candies You Can Get from Boxes
struct Sol1298 {}

impl Sol1298 {
    pub fn max_candies(
        status: Vec<i32>,
        candies: Vec<i32>,
        keys: Vec<Vec<i32>>,
        contained_boxes: Vec<Vec<i32>>,
        initial_boxes: Vec<i32>,
    ) -> i32 {
        let mut gots = vec![false; status.len()];
        let mut opened = vec![false; status.len()];
        let mut can_open: Vec<_> = status.iter().map(|&v| v == 1).collect();

        use std::collections::VecDeque;

        let mut queue = VecDeque::new();
        for ibox in initial_boxes.iter().map(|&v| v as usize) {
            gots[ibox] = true;
            if can_open[ibox] {
                queue.push_back(ibox);
                opened[ibox] = true;
            }
        }

        let mut total = 0;
        while let Some(qbox) = queue.pop_front() {
            total += candies[qbox];

            for kbox in keys[qbox].iter().map(|&v| v as usize) {
                can_open[kbox] = true;
                if gots[kbox] && !opened[kbox] {
                    queue.push_back(kbox);
                    opened[kbox] = true;
                }
            }

            for gbox in contained_boxes[qbox].iter().map(|&v| v as usize) {
                gots[gbox] = true;
                if can_open[gbox] && !opened[gbox] {
                    queue.push_back(gbox);
                    opened[gbox] = true;
                }
            }
        }

        total
    }
}

/// 1368h Minimum Cost to Make at Least One Valid Path in a Grid
struct Sol1368;

impl Sol1368 {
    pub fn min_cost(grid: Vec<Vec<i32>>) -> i32 {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let (rows, cols) = (grid.len(), grid[0].len());
        let mut costs = vec![vec![i32::MAX; cols]; rows];

        let mut pq = BinaryHeap::new();
        pq.push(Reverse((0, 0i32, 0i32)));
        costs[0][0] = 0;

        println!(" -> {:?}", pq);
        while let Some(Reverse((w, r, c))) = pq.pop() {
            if costs[r as usize][c as usize] < w {
                continue;
            }

            print!(" -> {:?} :: ", (w, r, c));

            [(0, 1), (0, -1), (1, 0), (-1, 0)]
                .into_iter()
                .zip(1..=4)
                .inspect(|o| print!(" {:?}", o))
                .for_each(|((dx, dy), dir)| {
                    let w = w + (dir != grid[r as usize][c as usize]) as i32;
                    let (r, c) = (r + dx, c + dy);
                    if 0 <= r
                        && r < rows as i32
                        && 0 <= c
                        && c < cols as i32
                        && w < costs[r as usize][c as usize]
                    {
                        costs[r as usize][c as usize] = w;
                        pq.push(Reverse((w, r, c)));
                    }
                });

            println!();
            println!(" -> {:?}", pq);
        }

        println!(" :: {}", costs[rows - 1][cols - 1]);

        costs[rows - 1][cols - 1]
    }

    fn min_cost_bfs01(grid: Vec<Vec<i32>>) -> i32 {
        use std::collections::VecDeque;

        let (rows, cols) = (grid.len(), grid[0].len());
        let mut costs = vec![vec![i32::MAX; cols]; rows];

        let mut q = VecDeque::new();

        q.push_front((0i32, 0i32));
        costs[0][0] = 0;

        println!(" -> {:?}", q);

        while let Some((r, c)) = q.pop_front() {
            let curw = costs[r as usize][c as usize];

            [(0, 1), (0, -1), (1, 0), (-1, 0)]
                .into_iter()
                .zip(1..=4)
                .for_each(|((dx, dy), dir)| {
                    let w = (grid[r as usize][c as usize] != dir) as i32;
                    let (r, c) = (r + dx, c + dy);
                    if 0 <= r
                        && r < rows as i32
                        && 0 <= c
                        && c < cols as i32
                        && curw + w < costs[r as usize][c as usize]
                    {
                        costs[r as usize][c as usize] = curw + w;
                        match w {
                            0 => q.push_front((r, c)),
                            _ => q.push_back((r, c)),
                        }
                    }
                });

            println!(" -> {:?}", q);
        }

        println!(" :: {}", costs[rows - 1][cols - 1]);

        costs[rows - 1][cols - 1]
    }
}

/// 1462m Course Schedule IV
struct Sol1462;

impl Sol1462 {
    pub fn check_if_prerequisite(
        num_courses: i32,
        prerequisites: Vec<Vec<i32>>,
        queries: Vec<Vec<i32>>,
    ) -> Vec<bool> {
        let mut graph = vec![vec![]; num_courses as usize];

        prerequisites.iter().for_each(|v| {
            graph[v[0] as usize].push(v[1] as usize);
        });

        println!(" -> {:?}", graph);

        let mut floyd_warshall = vec![vec![false; num_courses as usize]; num_courses as usize];
        prerequisites.iter().for_each(|v| {
            floyd_warshall[v[0] as usize][v[1] as usize] = true;
        });
        (0..num_courses as usize).for_each(|v| floyd_warshall[v][v] = true);

        (0..num_courses as usize).for_each(|src| {
            (0..num_courses as usize).for_each(|dst| {
                (0..num_courses as usize).for_each(|via| {
                    floyd_warshall[src][dst] |=
                        floyd_warshall[src][via] && floyd_warshall[via][dst];
                })
            })
        });

        println!(" -> {:?}", floyd_warshall);

        let mut memory = vec![vec![false; num_courses as usize]; num_courses as usize];

        (0..num_courses as usize).for_each(|src| {
            let mut q = vec![];
            q.push(src);

            while let Some(v) = q.pop() {
                memory[src][v] = true;
                graph[v].iter().for_each(|&u| {
                    if !memory[src][u] {
                        q.push(u)
                    }
                });
            }
        });

        assert_eq!(memory, floyd_warshall);

        queries.into_iter().fold(vec![], |mut rst, qry| {
            rst.push(memory[qry[0] as usize][qry[1] as usize]);
            rst
        })
    }
}

/// 1765m Map of Highest Peak
struct Sol1765;

impl Sol1765 {
    pub fn highest_peak(is_water: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        use std::collections::VecDeque;

        let (rows, cols) = (is_water.len(), is_water[0].len());
        let mut q = VecDeque::new();
        let mut visited = vec![vec![false; cols]; rows];

        (0..rows).for_each(|r| {
            (0..cols).for_each(|c| {
                if is_water[r][c] == 1 {
                    q.push_back((r as i32, c as i32, 0));
                    visited[r][c] = true;
                }
            })
        });

        let mut rst = vec![vec![0; cols]; rows];
        let dirs = [-1, 0, 1, 0, -1];

        println!(" -> {:?}", q);
        while let Some((r, c, h)) = q.pop_front() {
            println!(" -> {:?}", q);

            (0..4).for_each(|i| {
                let (x, y) = (r + dirs[i], c + dirs[i + 1]);
                if 0 <= x
                    && x < rows as i32
                    && 0 <= y
                    && y < cols as i32
                    && !visited[x as usize][y as usize]
                {
                    visited[x as usize][y as usize] = true;
                    rst[x as usize][y as usize] = h + 1;
                    q.push_back((x, y, h + 1));
                }
            });
        }

        rst
    }
}

/// 1976m Number of Ways to Arrive at Destination
struct Sol1976;

impl Sol1976 {
    /// 1 <= n <= 200
    /// 0 <= u, v <= n-1
    /// 1 <= t <= 10^9
    pub fn count_paths(n: i32, roads: Vec<Vec<i32>>) -> i32 {
        let n = n as usize;
        const M: u64 = 1e9 as u64 + 7;

        // [][](Time, #Path)
        let mut graph = vec![vec![(1e9 as u64 * 200 + 1, 0); n]; n];
        for road in roads {
            let (u, v, t) = (road[0] as usize, road[1] as usize, road[2] as u64);
            graph[u][v] = (t, 1);
            graph[v][u] = (t, 1);
        }

        for (i, vc) in graph.iter_mut().enumerate() {
            vc[i] = (0u64, 1);
        }

        use std::cmp::Ordering::*;

        // Floyd-Warshall O(N^3)
        for m in 0..n {
            for v in 0..n {
                for u in 0..n {
                    if v != m && u != m {
                        match (graph[u][m].0 + graph[m][v].0).cmp(&graph[u][v].0) {
                            Less => {
                                graph[u][v].0 = graph[u][m].0 + graph[m][v].0;
                                graph[u][v].1 = (graph[u][m].1 * graph[m][v].1) % M;
                            }
                            Equal => {
                                graph[u][v].1 += (graph[u][m].1 * graph[m][v].1) % M;
                                graph[u][v].1 %= M;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        println!("-> {:?}", graph);

        graph[0][n - 1].1 as i32
    }
}

/// 2127h Maximum Employees to Be Invited to a Meeting
struct Sol2127;

impl Sol2127 {
    pub fn maximum_invitations(favorite: Vec<i32>) -> i32 {
        use std::collections::VecDeque;

        let n = favorite.len();
        let mut ins = vec![0; n];
        favorite.iter().for_each(|&f| ins[f as usize] += 1);

        let mut q = VecDeque::from(
            ins.iter()
                .enumerate()
                .filter_map(|(v, &degree)| if degree == 0 { Some(v) } else { None })
                .collect::<Vec<_>>(),
        );

        let mut depth = vec![1; n];

        while let Some(v) = q.pop_front() {
            let u = favorite[v] as usize;

            depth[u] = depth[u].max(depth[v] + 1);
            ins[u] -= 1;
            if ins[u] == 0 {
                q.push_back(u);
            }
        }

        let (mut lcycle, mut tcycle) = (0, 0);
        (0..n).for_each(|v| {
            if ins[v] != 0 {
                let (mut l, mut cur) = (0, v);
                while ins[cur] > 0 {
                    ins[cur] = 0;
                    l += 1;
                    cur = favorite[cur] as usize;
                }

                match l {
                    2 => tcycle += depth[v] + depth[favorite[v] as usize],
                    _ => lcycle = lcycle.max(l),
                }
            }
        });

        println!(" :: {:?}", (lcycle, tcycle));

        lcycle.max(tcycle)
    }
}

/// 2359m Find Closed Node to Given Two Nodes
struct Sol2359 {}

impl Sol2359 {
    /// 2 <= N <= 10^5
    pub fn closest_meeting_node(edges: Vec<i32>, node1: i32, node2: i32) -> i32 {
        fn dfs(v: usize, dists: &mut [i32], edges: &[i32]) {
            let u = edges[v];
            if u != -1 && dists[u as usize] == i32::MAX {
                dists[u as usize] = dists[v] + 1;
                dfs(u as usize, dists, edges);
            }
        }

        fn bfs(src: usize, dists: &mut [i32], edges: &[i32]) {
            use std::collections::VecDeque;

            let mut queue = VecDeque::from([src]);
            dists[src] = 0;

            while let Some(v) = queue.pop_front() {
                let u = edges[v];
                if u != -1 && dists[u as usize] == i32::MAX {
                    dists[u as usize] = dists[v] + 1;
                    queue.push_back(u as usize);
                }
            }
        }

        let (mut dists1, mut dists2) = (vec![i32::MAX; edges.len()], vec![i32::MAX; edges.len()]);
        (dists1[node1 as usize], dists2[node2 as usize]) = (0, 0);

        dfs(node1 as usize, &mut dists1, &edges);
        println!("-> {node1} ~ {dists1:?}");

        bfs(node2 as usize, &mut dists2, &edges);
        println!("-> {node2} ~ {dists2:?}");

        (0..edges.len())
            .fold((-1, i32::MAX), |(m_node, m_dist), v| {
                if m_dist > dists1[v].max(dists2[v]) {
                    (v as i32, dists1[v].max(dists2[v]))
                } else {
                    (m_node, m_dist)
                }
            })
            .0
    }
}

/// 2467m Most Profitable Path in a Tree
struct Sol2467;

impl Sol2467 {
    pub fn most_profitable_path(edges: Vec<Vec<i32>>, bob: i32, amount: Vec<i32>) -> i32 {
        let n = edges.len() + 1;
        let mut graph = vec![vec![]; n];

        for e in edges {
            graph[e[0] as usize].push(e[1] as usize);
            graph[e[1] as usize].push(e[0] as usize);
        }

        println!("-> {:?}", graph);

        let mut bdist = vec![100_000; n];
        bdist[bob as usize] = 0;

        fn search(
            u: usize,
            p: usize,
            time: usize,
            graph: &Vec<Vec<usize>>,
            bdist: &mut Vec<usize>,
            amount: &Vec<i32>,
        ) -> i32 {
            let mut profit = 0;
            let mut neighbors = i32::MIN;

            for &v in &graph[u] {
                if v != p {
                    neighbors = neighbors.max(search(v, u, time + 1, graph, bdist, amount));
                    bdist[u] = bdist[u].min(bdist[v] + 1);
                }
            }

            use std::cmp::Ordering::*;
            profit += match bdist[u].cmp(&time) {
                Greater => amount[u],
                Equal => amount[u] / 2,
                _ => 0,
            };

            match neighbors {
                i32::MIN => profit,
                _ => profit + neighbors,
            }
        }

        search(0, n, 0, &graph, &mut bdist, &amount)
    }
}

/// 2493h Divide Nodes Into the Maximum Number of Groups
struct Sol2493;

impl Sol2493 {
    pub fn magnificent_sets(n: i32, edges: Vec<Vec<i32>>) -> i32 {
        use std::collections::VecDeque;

        println!("* {:?}", edges);

        let mut graph = vec![vec![]; n as usize + 1];
        for e in &edges {
            graph[e[0] as usize].push(e[1] as usize);
            graph[e[1] as usize].push(e[0] as usize);
        }
        println!("-> graph :: {:?}", graph);

        let mut bicolors = vec![-1; graph.len()];
        for v in 1..graph.len() {
            if bicolors[v] != -1 {
                continue;
            }

            bicolors[v] = 0;
            let mut q = vec![v];

            while let Some(v) = q.pop() {
                println!("-> {} {:?}", v, q);

                for &u in &graph[v] {
                    if bicolors[u] == bicolors[v] {
                        return -1; // graph is not `BiPartite`
                    }
                    if bicolors[u] != -1 {
                        continue;
                    }

                    bicolors[u] = bicolors[v] ^ 1;
                    q.push(u);
                }
            }
        }
        println!("-> colors :: {:?}", bicolors);

        let mut dist = vec![0; graph.len()];
        for n in 1..graph.len() {
            let mut dq = VecDeque::new();
            let mut visited = vec![false; graph.len()];

            dq.push_back(n);
            visited[n] = true;

            let mut xdist = 0;
            while !dq.is_empty() {
                for _ in 0..dq.len() {
                    if let Some(v) = dq.pop_front() {
                        for &u in &graph[v] {
                            if !visited[u] {
                                visited[u] = true;
                                dq.push_back(u);
                            }
                        }
                    }
                }
                xdist += 1;
            }

            dist[n] = xdist;
        }
        println!("-> distances :: {:?}", dist);

        let mut groups = 0;
        let mut visited = vec![false; graph.len()];
        for n in 1..graph.len() {
            if !visited[n] {
                let mut ngroup = 0;
                let mut q = vec![n];
                visited[n] = true;
                while let Some(v) = q.pop() {
                    ngroup = ngroup.max(dist[v]);
                    for &u in &graph[v] {
                        if !visited[u] {
                            visited[u] = true;
                            q.push(u);
                        }
                    }
                }

                groups += ngroup;
            }
        }

        groups
    }
}

/// 2503h Maximum Number of Points From Grid Queries
struct Sol2503;

impl Sol2503 {
    pub fn max_points(grid: Vec<Vec<i32>>, queries: Vec<i32>) -> Vec<i32> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut sqry = vec![];
        for (i, &qry) in queries.iter().enumerate() {
            sqry.push((qry, i));
        }
        sqry.sort_unstable();
        println!("-> {:?}", sqry);

        let (rows, cols) = (grid.len(), grid[0].len());
        let mut visited = vec![vec![false; cols]; rows];

        let mut pq = BinaryHeap::new();

        pq.push(Reverse((grid[0][0], 0usize, 0usize)));
        visited[0][0] = true;

        let mut rst = vec![0; queries.len()];
        let mut points = 0;

        const DIRS: [isize; 5] = [-1, 0, 1, 0, -1];
        for (qry, i) in sqry {
            while let Some(&Reverse((gval, r, c))) = pq.peek()
                && gval < qry
            {
                println!("-> {} {:?}", gval, (r, c));

                pq.pop();
                points += 1;

                for d in 0..4 {
                    let (r, c) = (
                        r.overflowing_add_signed(DIRS[d]).0,
                        c.overflowing_add_signed(DIRS[d + 1]).0,
                    );

                    if r < rows && c < cols && !visited[r][c] {
                        visited[r][c] = true;
                        pq.push(Reverse((grid[r][c], r, c)));

                        println!("-> * {:?} {:?}", (r, c), &pq);
                    }
                }
            }

            rst[i] = points;
        }

        rst
    }
}

/// 2608h Shortest Cycle in a Graph
struct Sol2608;

impl Sol2608 {
    pub fn find_shortest_cycle(n: i32, edges: Vec<Vec<i32>>) -> i32 {
        use std::collections::VecDeque;

        println!("* {:?}", edges);

        let n = n as usize;
        let mut graph = vec![vec![]; n];
        for e in &edges {
            let (v, u) = (e[0] as usize, e[1] as usize);
            graph[v].push(u);
            graph[u].push(v);
        }
        println!("-> graph :: {:?}", graph);

        let mut mcycle = usize::MAX;
        (0..n).for_each(|src| {
            let mut dist = vec![usize::MAX; n];

            let mut dq = VecDeque::new();
            dq.push_back(src);
            dist[src] = 0;

            while let Some(v) = dq.pop_front() {
                println!(" -> {} {:?}", v, dq);

                for &u in &graph[v] {
                    match dist[u] {
                        usize::MAX => {
                            dist[u] = dist[v] + 1;
                            dq.push_back(u);
                        }
                        _ => {
                            if dist[v] <= dist[u] {
                                mcycle = mcycle.min(dist[u] + dist[v] + 1);
                            }
                        }
                    }
                }
            }
        });

        match mcycle {
            usize::MAX => -1,
            _ => mcycle as i32,
        }
    }
}

/// 2658m Maximum Number of Fish in a Grid
struct Sol2658;

impl Sol2658 {
    pub fn find_max_fish(grid: Vec<Vec<i32>>) -> i32 {
        println!("* {:?}", grid);

        let (rows, cols) = (grid.len(), grid[0].len());
        let mut visited = vec![vec![false; cols]; rows];

        let dir = [-1, 0, 1, 0, -1];
        let mut xfish = 0;

        (0..rows).for_each(|r| {
            (0..cols).for_each(|c| {
                if grid[r][c] != 0 && !visited[r][c] {
                    println!("-> {:?}", (r, c));

                    let mut fish = 0;
                    let mut q = vec![];

                    q.push((r as i32, c as i32));
                    visited[r][c] = true;

                    while let Some((r, c)) = q.pop() {
                        println!(" -> {:?}", (r, c));

                        fish += grid[r as usize][c as usize];
                        xfish = xfish.max(fish);

                        (0..4).for_each(|i| {
                            let (r, c) = (r + dir[i], c + dir[i + 1]);
                            if 0 <= r
                                && r < rows as i32
                                && 0 <= c
                                && c < cols as i32
                                && grid[r as usize][c as usize] != 0
                                && !visited[r as usize][c as usize]
                            {
                                visited[r as usize][c as usize] = true;
                                q.push((r, c));
                            }
                        })
                    }
                }
            })
        });

        xfish
    }

    fn find_max_fish_recursion(grid: Vec<Vec<i32>>) -> i32 {
        fn dfs(grid: &mut Vec<Vec<i32>>, r: usize, c: usize) -> i32 {
            if grid[r][c] == 0 {
                return 0;
            }

            let mut fish = grid[r][c];
            grid[r][c] = 0;

            if r > 0 {
                fish += dfs(grid, r - 1, c);
            }
            if r + 1 < grid.len() {
                fish += dfs(grid, r + 1, c);
            }
            if c > 0 {
                fish += dfs(grid, r, c - 1);
            }
            if c + 1 < grid[0].len() {
                fish += dfs(grid, r, c + 1);
            }

            fish
        }

        let mut grid = grid;
        let mut xfish = 0;

        (0..grid.len())
            .for_each(|r| (0..grid[0].len()).for_each(|c| xfish = xfish.max(dfs(&mut grid, r, c))));

        xfish
    }
}

/// 2685m Count the Number of Complete Components
struct Sol2685;

impl Sol2685 {
    pub fn count_complete_components(n: i32, edges: Vec<Vec<i32>>) -> i32 {
        let mut cliques = 0;

        let n = n as usize;
        let mut graph = vec![vec![]; n];

        for edge in edges {
            let (v, u) = (edge[0] as usize, edge[1] as usize);
            graph[v].push(u);
            graph[u].push(v);
        }

        println!("-> {:?}", graph);

        fn dfs(graph: &[Vec<usize>], visited: &mut [bool], v: usize) -> (usize, usize) {
            let (mut vertices, mut edges) = (1, 0);

            for &u in &graph[v] {
                edges += 1;
                if !visited[u] {
                    visited[u] = true;

                    let rst = dfs(graph, visited, u);
                    vertices += rst.0;
                    edges += rst.1;
                }
            }

            (vertices, edges)
        }

        let mut visited = vec![false; n];
        for v in 0..n {
            if visited[v] {
                continue;
            }
            visited[v] = true;

            let (vertices, edges) = dfs(&graph, &mut visited, v);
            if vertices * (vertices - 1) == edges {
                cliques += 1;
            }
        }

        cliques
    }
}

/// 3108h Minimum Cost Walk in Weighted Graph
struct Sol3108;

impl Sol3108 {
    pub fn minimum_cost(n: i32, edges: Vec<Vec<i32>>, query: Vec<Vec<i32>>) -> Vec<i32> {
        use std::cmp::Ordering::*;

        let n = n as usize;
        let mut djset: Vec<usize> = (0..n).collect();
        let mut ranks = vec![0; n];
        let mut weights = vec![usize::MAX; n];

        fn find(djset: &mut [usize], x: usize) -> usize {
            if djset[x] != x {
                djset[x] = find(djset, djset[x]);
            }
            djset[x]
        }

        fn union(djset: &mut [usize], ranks: &mut [usize], x: usize, y: usize) -> usize {
            let x = find(djset, x);
            let y = find(djset, y);
            match x.cmp(&y) {
                Equal => x,
                _ => match ranks[x].cmp(&ranks[y]) {
                    Less => {
                        djset[x] = y;
                        y
                    }
                    _ => {
                        djset[y] = x;
                        if ranks[x] == ranks[y] {
                            ranks[x] += 1;
                        }
                        x
                    }
                },
            }
        }

        for v in &edges {
            union(&mut djset, &mut ranks, v[0] as usize, v[1] as usize);
        }

        for v in &edges {
            let r = find(&mut djset, v[0] as usize);
            weights[r] &= v[2] as usize;
        }

        println!("-> {:?}", (&djset, &ranks, &weights));

        let mut rst = vec![];
        for v in query {
            let (x, y) = (
                find(&mut djset, v[0] as usize),
                find(&mut djset, v[1] as usize),
            );
            match x.cmp(&y) {
                Equal => rst.push(weights[x] as i32),
                _ => rst.push(-1),
            }
        }

        rst
    }
}

/// 3341m Find Minimum Time to Reach Last Room I
struct Sol3341;

impl Sol3341 {
    pub fn min_time_to_reach(move_time: Vec<Vec<i32>>) -> i32 {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let (rows, cols) = (move_time.len(), move_time[0].len());
        let mut grid = vec![vec![i32::MAX; cols]; rows];
        let mut pq = BinaryHeap::new();

        grid[0][0] = 0;
        pq.push(Reverse((0, 0, 0)));

        while let Some(Reverse((time, r, c))) = pq.pop() {
            println!("-> {:?}", (time, r, c, &pq, &grid));

            if r == rows - 1 && c == cols - 1 {
                return time;
            }

            let dirs = [-1, 0, 1, 0, -1];
            for d in 0..4 {
                if let (Some(r), Some(c)) = (
                    r.checked_add_signed(dirs[d]),
                    c.checked_add_signed(dirs[d + 1]),
                ) {
                    if r < rows && c < cols && move_time[r][c].max(time) + 1 < grid[r][c] {
                        grid[r][c] = move_time[r][c].max(time) + 1;
                        pq.push(Reverse((grid[r][c], r, c)));
                    }
                }
            }
        }

        -1
    }
}

/// 3342m Find Minimum Time to Reach Last Room II
struct Sol3342;

impl Sol3342 {
    pub fn min_time_to_reach(move_time: Vec<Vec<i32>>) -> i32 {
        use std::cmp::Reverse;

        let (rows, cols) = (move_time.len(), move_time[0].len());
        let mut grid = vec![vec![i32::MAX; cols]; rows]; // Distances from {0,0}

        let dirs = [-1, 0, 1, 0, -1];
        let mut pq = std::collections::BinaryHeap::new();

        pq.push(Reverse((0, 0, 0, false)));
        grid[0][0] = 0;

        while let Some(Reverse((time, r, c, double))) = pq.pop() {
            println!("-> {:?}", (time, r, c, double, &pq, &grid));

            if r + 1 == rows && c + 1 == cols {
                return time;
            }

            let delta = if double { 2 } else { 1 };

            for d in 0..4 {
                if let (Some(r), Some(c)) = (
                    r.checked_add_signed(dirs[d]),
                    c.checked_add_signed(dirs[d + 1]),
                ) {
                    if r < rows && c < cols && move_time[r][c].max(time) + delta < grid[r][c] {
                        grid[r][c] = move_time[r][c].max(time) + delta;
                        pq.push(Reverse((grid[r][c], r, c, !double)));
                    }
                }
            }
        }

        -1
    }
}

/// 3372m Maximize the Number of Target Nodes After Connecting Trees I
struct Sol3372 {}

impl Sol3372 {
    /// 2 <= M, N <= 1000
    /// 0 <= K <= 1000
    pub fn max_target_nodes(edges1: Vec<Vec<i32>>, edges2: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
        let (mut tree1, mut tree2) = (
            vec![vec![]; edges1.len() + 1],
            vec![vec![]; edges2.len() + 1],
        );

        let mk_adjs = |tree: &mut [Vec<usize>], edges: &[Vec<i32>]| {
            for edge in edges {
                tree[edge[0] as usize].push(edge[1] as usize);
                tree[edge[1] as usize].push(edge[0] as usize);
            }
        };

        for (tree, edges) in [(&mut tree1, &edges1), (&mut tree2, &edges2)] {
            mk_adjs(tree, edges);
            println!("-> {edges:?} => {tree:?}");
        }

        fn dfs(v: usize, p: usize, k: i32, tree: &[Vec<usize>], dists: &mut [i32]) {
            for &u in &tree[v] {
                if u != p {
                    dists[u] = dists[v] + 1;
                    if dists[u] < k {
                        dfs(u, v, k, tree, dists);
                    }
                }
            }
        }

        let mut trgs = vec![];
        for src in 0..=edges2.len() {
            let mut dists = vec![i32::MAX; edges2.len() + 1];
            dists[src] = 0;
            dfs(src, usize::MAX, k - 1, &tree2, &mut dists);
            println!("-> {src} {dists:?}");

            trgs.push(dists.iter().filter(|&&d| d <= k - 1).count());
        }
        println!("-> {trgs:?}");

        let mut xtrgs = vec![];
        if let Some(&xtrg) = trgs.iter().max() {
            for src in 0..=edges1.len() {
                let mut dists = vec![i32::MAX; edges1.len() + 1];
                dists[src] = 0;
                dfs(src, usize::MAX, k, &tree1, &mut dists);
                println!("-> {src} {dists:?}");

                xtrgs.push((dists.iter().filter(|&&d| d <= k).count() + xtrg) as i32);
            }
        }

        xtrgs
    }
}

/// 3373m Maximize the Number of Target Nodes After Connecting Trees II
struct Sol3373 {}

impl Sol3373 {
    pub fn max_target_nodes(edges1: Vec<Vec<i32>>, edges2: Vec<Vec<i32>>) -> Vec<i32> {
        let (mut tree1, mut tree2) = (
            vec![vec![]; edges1.len() + 1],
            vec![vec![]; edges2.len() + 1],
        );

        let mk_adjs = |tree: &mut [Vec<usize>], edges: &[Vec<i32>]| {
            for edge in edges {
                tree[edge[0] as usize].push(edge[1] as usize);
                tree[edge[1] as usize].push(edge[0] as usize);
            }
        };

        for (tree, edges) in [(&mut tree1, &edges1), (&mut tree2, &edges2)] {
            mk_adjs(tree, edges);
            println!("-> {edges:?} => {tree:?}");
        }

        fn color_nodes(
            v: usize,
            p: usize,
            depth: usize,
            colors: &mut [usize],
            tree: &[Vec<usize>],
        ) {
            colors[v] = depth & 1;
            for &u in &tree[v] {
                if u != p {
                    color_nodes(u, v, depth + 1, colors, tree);
                }
            }
        }

        fn color_nodes_iterative(colors: &mut [usize], tree: &[Vec<usize>]) {
            use std::collections::VecDeque;

            let mut q_bfs = VecDeque::from([(0, usize::MAX, 0)]);
            while let Some((v, p, depth)) = q_bfs.pop_front() {
                colors[v] = depth & 1;
                for &u in &tree[v] {
                    if u != p {
                        q_bfs.push_back((u, v, depth + 1));
                    }
                }
            }
        }

        let mut colors1 = vec![0; tree1.len()];
        color_nodes(0, usize::MAX, 0, &mut colors1, &tree1);

        let mut counts1 = [0; 2];
        counts1[0] = colors1.iter().filter(|&&color| color == 0).count();
        counts1[1] = tree1.len() - counts1[0];

        println!("-> {counts1:?} {colors1:?}");

        let mut colors2 = vec![0; tree2.len()];
        color_nodes_iterative(&mut colors2, &tree2);

        let mut counts2 = [0; 2];
        counts2[0] = colors2.iter().filter(|&&color| color == 0).count();
        counts2[1] = tree2.len() - counts2[0];

        println!("-> {counts2:?} {colors2:?}");

        let mut xtrgs = vec![];
        for color in colors1 {
            xtrgs.push((counts1[color] + counts2[0].max(counts2[1])) as i32);
        }

        xtrgs
    }
}

#[cfg(test)]
mod tests;
