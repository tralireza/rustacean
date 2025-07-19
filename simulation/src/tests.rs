use super::*;

#[test]
fn test_2154() {
    for (rst, nums, original) in [(24, vec![5, 3, 6, 1, 12], 3), (4, vec![2, 7, 9], 4)] {
        println!("* {nums:?} {original}");
        assert_eq!(Sol2154::find_final_value(nums, original), rst);
        println!(":: {rst:?}");
    }
}

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
fn test_2402() {
    for (rst, n, meetings) in [
        (0, 2, vec![vec![0, 10], vec![1, 5], vec![2, 7], vec![3, 4]]),
        (
            1,
            3,
            vec![vec![1, 20], vec![2, 10], vec![3, 5], vec![4, 9], vec![6, 8]],
        ),
    ] {
        println!("* {n} {meetings:?}");
        assert_eq!(Sol2402::most_booked(n, meetings), rst);
        println!(":: {rst:?}");
    }
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
