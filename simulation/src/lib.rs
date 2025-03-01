//! # Rust :: Simulation

/// 2460 Apply Operations to an Array
struct Sol2460;

impl Sol2460 {
    pub fn apply_operations(nums: Vec<i32>) -> Vec<i32> {
        let mut nums = nums;
        for i in 0..nums.len() - 1 {
            match nums[i] == nums[i + 1] {
                true => {
                    nums[i] *= 2;
                    nums[i + 1] = 0;
                }
                _ => (),
            }
        }

        let mut vcopy = nums.to_vec();
        vcopy.sort_by_key(|&v| if v > 0 { 0 } else { 1 });

        println!("-> {:?}", vcopy);

        for z in 0..nums.len() {
            if nums[z] == 0 {
                let mut swap = z;
                while nums[swap] == 0 {
                    swap += 1;
                    if swap >= nums.len() {
                        return nums;
                    }
                }
                (nums[z], nums[swap]) = (nums[swap], nums[z]);
            }
        }

        nums
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
