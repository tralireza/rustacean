use super::*;

#[test]
fn test_2161() {
    assert_eq!(
        Sol2161::pivot_array(vec![9, 12, 5, 10, 14, 3, 10], 10),
        vec![9, 5, 3, 10, 10, 12, 14]
    );
    assert_eq!(
        Sol2161::pivot_array(vec![-3, 4, 3, 2], 2),
        vec![-3, 2, 4, 3]
    );
}

#[test]
fn test_2460() {
    assert_eq!(
        Sol2460::apply_operations(vec![1, 2, 2, 1, 1, 0]),
        vec![1, 4, 2, 0, 0, 0]
    );
    assert_eq!(Sol2460::apply_operations(vec![0, 1]), vec![1, 0]);
    assert_eq!(
        Sol2460::apply_operations(vec![
            847, 847, 0, 0, 0, 399, 416, 416, 879, 879, 206, 206, 206, 272
        ]),
        vec![1694, 399, 832, 1758, 412, 206, 272, 0, 0, 0, 0, 0, 0, 0]
    );
}
