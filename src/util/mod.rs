pub trait Numeric: PartialOrd + Copy {
    fn min() -> Self;
    fn zero() -> Self;
    fn has_decimal() -> bool;

    fn integral_digit_count(&self) -> usize;
}

impl Numeric for u8 {
    fn min() -> Self {
        u8::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for u16 {
    fn min() -> Self {
        u16::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for u32 {
    fn min() -> Self {
        u32::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for u64 {
    fn min() -> Self {
        u64::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for u128 {
    fn min() -> Self {
        u128::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i8 {
    fn min() -> Self {
        i8::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i16 {
    fn min() -> Self {
        i16::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i32 {
    fn min() -> Self {
        i32::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i64 {
    fn min() -> Self {
        i64::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for i128 {
    fn min() -> Self {
        i128::MIN
    }

    fn zero() -> Self {
        0
    }

    fn has_decimal() -> bool {
        true
    }

    fn integral_digit_count(&self) -> usize {
        self.to_string().len()
    }
}

impl Numeric for f32 {
    fn min() -> Self {
        f32::MIN
    }

    fn zero() -> Self {
        0.0
    }

    fn has_decimal() -> bool {
        true
    }

    fn integral_digit_count(&self) -> usize {
        (*self as i128).to_string().len()
    }
}

impl Numeric for f64 {
    fn min() -> Self {
        f64::MIN
    }

    fn zero() -> Self {
        0.0
    }

    fn has_decimal() -> bool {
        true
    }

    fn integral_digit_count(&self) -> usize {
        (*self as i128).to_string().len()
    }
}

impl Numeric for bool {
    fn min() -> Self {
        false
    }

    fn zero() -> Self {
        false
    }

    fn has_decimal() -> bool {
        false
    }

    fn integral_digit_count(&self) -> usize {
        0
    }
}
