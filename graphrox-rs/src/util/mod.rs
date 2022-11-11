use std::mem;
use std::slice;

pub mod constants {
    pub const MIN_THRESHOLD_DIVISOR_POWER_TEN: u32 = 17;
    pub const GRAPH_APPROXIMATION_MIN_THRESHOLD: f64 =
        1.0f64 / (10u64.pow(MIN_THRESHOLD_DIVISOR_POWER_TEN) as f64);

    // This is 8 for a good reason. It allows for each bit in a u64 to represent
    // an entry in an 8x8 block of an adjacency matrix. It should also be a power
    // of two so the compiler can optimize slow integer multiplications and
    // divisions into fast bitwise operations
    pub const COMPRESSION_BLOCK_DIMENSION: u64 = 8;
}

#[inline]
pub unsafe fn as_byte_slice<T: Sized>(item: &T) -> &[u8] {
    slice::from_raw_parts((item as *const T) as *const u8, mem::size_of::<T>())
}

#[inline]
pub fn clamp_threshold(threshold: f64) -> f64 {
    if threshold > 1.0 {
        1.0
    } else if threshold <= 0.0 {
        constants::GRAPH_APPROXIMATION_MIN_THRESHOLD
    } else {
        threshold
    }
}

pub trait Numeric: PartialOrd + Copy {
    fn max() -> Self;
    fn min() -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn has_decimal() -> bool;

    fn add_one(&self) -> Self;
    fn integral_digit_count(&self) -> usize;
}

impl Numeric for u8 {
    #[inline(always)]
    fn max() -> Self {
        u8::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        u8::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for u16 {
    #[inline(always)]
    fn max() -> Self {
        u16::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        u16::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for u32 {
    #[inline(always)]
    fn max() -> Self {
        u32::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        u32::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for u64 {
    #[inline(always)]
    fn max() -> Self {
        u64::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        u64::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for u128 {
    #[inline(always)]
    fn max() -> Self {
        u128::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        u128::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i8 {
    #[inline(always)]
    fn max() -> Self {
        i8::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        i8::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i16 {
    #[inline(always)]
    fn max() -> Self {
        i16::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        i16::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i32 {
    #[inline(always)]
    fn max() -> Self {
        i32::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        i32::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i64 {
    #[inline(always)]
    fn max() -> Self {
        i64::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        i64::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i128 {
    #[inline(always)]
    fn max() -> Self {
        i128::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        i128::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        true
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for f32 {
    #[inline(always)]
    fn max() -> Self {
        f32::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        f32::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn one() -> Self {
        1.0
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        true
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1.0
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        (*self as i128).to_string().len()
    }
}

impl Numeric for f64 {
    #[inline(always)]
    fn max() -> Self {
        f64::MAX
    }

    #[inline(always)]
    fn min() -> Self {
        f64::MIN
    }

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn one() -> Self {
        1.0
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        true
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        self + 1.0
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        (*self as i128).to_string().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, packed)]
    struct ByteSliceTestStruct {
        field1: u8,
        field2: u8,
        field3: u16,
        field4: u8,
    }

    #[test]
    fn test_as_byte_slice() {
        let structure = ByteSliceTestStruct {
            field1: 42,
            field2: 8,
            field3: 0x1234,
            field4: 2,
        };

        let structure_bytes = unsafe { as_byte_slice(&structure) };

        assert_eq!(structure_bytes.len(), 5);
        assert_eq!(structure_bytes[0], structure.field1);
        assert_eq!(structure_bytes[1], structure.field2);
        assert_eq!(structure_bytes[4], structure.field4);

        let field3 = structure.field3;
        let field3_bytes = unsafe { as_byte_slice(&field3) };

        assert_eq!(field3_bytes.len(), 2);
        assert_eq!(structure_bytes[2], field3_bytes[0]);
        assert_eq!(structure_bytes[3], field3_bytes[1]);
    }
}
