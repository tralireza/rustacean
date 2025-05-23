use super::*;

#[test]
fn test_38() {
    for (rst, n) in [
        ("1211".to_string(), 4),
        ("1".to_string(), 1),
        ("111221".to_string(), 5),
    ] {
        assert_eq!(Sol38::count_and_say(n), rst);
    }
}

#[test]
fn test_65() {
    for f in [Sol65::is_number, Sol65::is_number_enum] {
        assert!(f("0".to_string()));
        assert!(!f("e".to_string()));
        assert!(!f(".".to_string()));

        assert!(f("2e0".to_string()));
    }
}

#[test]
fn test_917() {
    assert_eq!(
        Sol917::reverse_only_letters("a-bC-dEf-ghIj".to_string()),
        "j-Ih-gfE-dCba".to_string()
    );
}

#[test]
fn test_1154() {
    assert_eq!(Sol1154::day_of_year("2019-01-09".to_string()), 9);
    assert_eq!(Sol1154::day_of_year("2019-02-10".to_string()), 41);
}
