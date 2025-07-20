//! Rust :: Binary Tree

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

/// 1123m Lowest Common Ancestor of Deepest Leaves
struct Sol1123;

impl Sol1123 {
    pub fn lca_deepest_leaves(
        root: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        fn dfs(n: Option<Rc<RefCell<TreeNode>>>) -> (i32, Option<Rc<RefCell<TreeNode>>>) {
            println!("-> {:?}", n);

            match n {
                None => (0, None),
                Some(n) => {
                    let nref = n.borrow();
                    let (ld, l) = dfs(nref.left.clone());
                    let (rd, r) = dfs(nref.right.clone());

                    use std::cmp::Ordering::*;
                    match ld.cmp(&rd) {
                        Less => (1 + rd, r),
                        Greater => (1 + ld, l),
                        Equal => (1 + (ld | rd), Some(n.clone())),
                    }
                }
            }
        }

        dfs(root).1
    }
}

/// 2236 Root Equals Sum of Children
struct Sol2236 {}

impl Sol2236 {
    pub fn check_tree(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let root = root.unwrap();
        let n = root.borrow();
        let l = n.left.as_ref().unwrap().borrow();
        let r = n.right.as_ref().unwrap().borrow();

        n.val == l.val + r.val
    }
}

#[cfg(test)]
mod tests;
