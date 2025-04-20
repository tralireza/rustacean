use super::*;

#[test]
fn test_781() {
    for (rst, answers) in [(5, vec![1, 1, 2]), (11, vec![10, 10, 10])] {
        assert_eq!(Sol781::num_rabbits(answers), rst);
    }
}
