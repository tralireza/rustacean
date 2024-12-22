//! Binary Tree

/// 2872h Maximum Number of K-Divisible Components
struct Solution2872;

impl Solution2872 {
    pub fn max_k_divisible_components(
        n: i32,
        edges: Vec<Vec<i32>>,
        values: Vec<i32>,
        k: i32,
    ) -> i32 {
        if n == 1 {
            return 1;
        }

        let n = n as usize;
        let mut graph = vec![vec![]; n];
        let mut ins = vec![0; n]; // v: In Degrees

        for e in edges {
            let (v, u) = (e[0] as usize, e[1] as usize);

            graph[v].push(u);
            graph[u].push(v);

            ins[v] += 1;
            ins[u] += 1;
        }

        let mut queue = std::collections::VecDeque::new();
        for v in 0..n {
            if ins[v] == 1 {
                queue.push_back(v);
            }
        }

        let mut values = values;
        let mut cmps = 0;

        while let Some(v) = queue.pop_front() {
            println!(" -> Q :: {:?}", queue);

            ins[v] -= 1;
            let mut rval = 0;

            match values[v] % k {
                0 => cmps += 1,
                _ => rval += values[v],
            };

            for u in &graph[v] {
                let u = *u;
                if ins[u] > 0 {
                    values[u] += rval % k;

                    ins[u] -= 1;
                    if ins[u] == 1 {
                        queue.push_back(u);
                    }
                }
            }
        }

        cmps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution2872() {
        assert_eq!(
            Solution2872::max_k_divisible_components(
                5,
                vec![vec![0, 2], vec![1, 2], vec![1, 3], vec![2, 4]],
                vec![1, 8, 1, 4, 4],
                6
            ),
            2
        );
        println!();
        assert_eq!(
            Solution2872::max_k_divisible_components(
                7,
                vec![
                    vec![0, 1],
                    vec![0, 2],
                    vec![1, 3],
                    vec![1, 4],
                    vec![2, 5],
                    vec![2, 6]
                ],
                vec![3, 0, 6, 1, 5, 2, 1],
                3
            ),
            3
        );
    }
}
