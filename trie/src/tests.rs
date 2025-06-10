use super::*;

#[test]
fn test_1163() {
    for (rst, s) in [
        ("bab".to_string(), "abab".to_string()),
        ("tcode".to_string(), "leetcode".to_string()),
    ] {
        println!("* {s}");
        assert_eq!(Sol1163::last_substring(s), rst);
        println!(":: {rst}");
    }
}
