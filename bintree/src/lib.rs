//! Binary Tree

use std::cell::RefCell;
/// TreeNode
use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
struct TreeNode {
    val: i32,
    left: Option<Rc<RefCell<TreeNode>>>,
    right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

/// 563 Binary Tree Tilt
struct Solution563;

impl Solution563 {
    pub fn find_tilt(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn postorder(node: &Option<Rc<RefCell<TreeNode>>>, tilt: &mut i32) -> i32 {
            if let Some(node) = node {
                let node = node.borrow();

                let (left, right) = (postorder(&node.left, tilt), postorder(&node.right, tilt));

                *tilt += (right - left).abs();
                return node.val + left + right;
            }

            0
        }

        let mut tilt = 0;
        postorder(&root, &mut tilt);

        tilt
    }
}

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
    fn test_solution563() {
        assert_eq!(
            Solution563::find_tilt(Some(Rc::new(RefCell::new(TreeNode {
                val: 1,
                left: Some(Rc::new(RefCell::new(TreeNode::new(2)))),
                right: Some(Rc::new(RefCell::new(TreeNode {
                    val: 3,
                    left: None,
                    right: None
                })))
            })))),
            1
        );
        //assert_eq!(Solution563::find_tilt(), 15);
        //assert_eq!(Solution563::find_tilt(), 9);
    }

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
