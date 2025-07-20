use super::*;

#[test]
fn test_1123() {
    for (rst, root) in [
        (TreeNode::new(1), TreeNode::new(1)),
        (
            TreeNode::new(2),
            TreeNode {
                val: 0,
                left: Some(Rc::new(RefCell::new(TreeNode {
                    val: 1,
                    left: None,
                    right: Some(Rc::new(RefCell::new(TreeNode::new(2)))),
                }))),
                right: Some(Rc::new(RefCell::new(TreeNode::new(3)))),
            },
        ),
    ] {
        assert_eq!(
            Sol1123::lca_deepest_leaves(Some(Rc::new(RefCell::new(root)))),
            Some(Rc::new(RefCell::new(rst)))
        );
    }
}

#[test]
fn test_2236() {
    for (rst, root) in [(
        true,
        TreeNode {
            val: 10,
            left: Some(Rc::new(RefCell::new(TreeNode::new(4)))),
            right: Some(Rc::new(RefCell::new(TreeNode::new(6)))),
        },
    )] {
        println!("* {root:?}");
        assert_eq!(Sol2236::check_tree(Some(Rc::new(RefCell::new(root)))), rst);
        println!(":: {rst:?}");
    }
}
