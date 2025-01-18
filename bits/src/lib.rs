//! # Bits (aka Bitwise)

/// 868 Binary Gap
struct Sol868;

impl Sol868 {
    pub fn binary_gap(n: i32) -> i32 {
        let (mut dist, mut cur) = (0, -32);

        let mut n = n;
        while n > 0 {
            cur += 1;
            dist = dist.max(cur);
            if n & 1 == 1 {
                cur = 0;
            }
            n >>= 1;
        }

        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_868() {
        assert_eq!(Sol868::binary_gap(22), 2);
        assert_eq!(Sol868::binary_gap(8), 0);
        assert_eq!(Sol868::binary_gap(5), 2);
    }
}
