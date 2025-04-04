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
