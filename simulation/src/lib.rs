//! # Rust :: Simulation

/// 2161m Partition Array According to Given Pivot
struct Sol2161;

impl Sol2161 {
    pub fn pivot_array(nums: Vec<i32>, pivot: i32) -> Vec<i32> {
        let mut rst = vec![pivot; nums.len()];

        let (mut left, mut right) = (0, nums.len() - 1);
        for (l, r) in (0..nums.len()).zip((0..nums.len()).rev()) {
            if nums[l] < pivot {
                rst[left] = nums[l];
                left += 1;
            }
            if nums[r] > pivot {
                rst[right] = nums[r];
                right -= 1;
            }
        }

        println!(":: {:?}", rst);

        rst
    }
}

/// 2460 Apply Operations to an Array
struct Sol2460;

impl Sol2460 {
    pub fn apply_operations(nums: Vec<i32>) -> Vec<i32> {
        let mut nums = nums;
        for i in 0..nums.len() - 1 {
            if nums[i] == nums[i + 1] {
                nums[i] *= 2;
                nums[i + 1] = 0;
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
mod tests;
