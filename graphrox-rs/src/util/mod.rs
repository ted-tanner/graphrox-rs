use std::mem;
use std::slice;

pub mod constants {
    pub const MIN_THRESHOLD_DIVISOR_POWER_TEN: u32 = 17;

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

pub trait Numeric: PartialOrd + Copy {
    fn min() -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn has_decimal() -> bool;

    fn add_one(&self) -> Self;
    fn integral_digit_count(&self) -> usize;
}

impl Numeric for u8 {
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

impl Numeric for bool {
    #[inline(always)]
    fn min() -> Self {
        false
    }

    #[inline(always)]
    fn zero() -> Self {
        false
    }

    #[inline(always)]
    fn one() -> Self {
        true
    }

    #[inline(always)]
    fn has_decimal() -> bool {
        false
    }

    #[inline(always)]
    fn add_one(&self) -> Self {
        true
    }

    #[inline(always)]
    fn integral_digit_count(&self) -> usize {
        0
    }
}
