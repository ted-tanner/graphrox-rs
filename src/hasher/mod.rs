use std::hash::Hasher;

#[derive(Clone, Default)]
pub struct GraphRoxHasher {
    current: u32,
}

impl Hasher for GraphRoxHasher {
    #[inline(always)]
    fn write_u64(&mut self, value: u64) {
        self.write_u32(value as u32);
    }

    #[inline(always)]
    fn write_u32(&mut self, value: u32) {
        let mut value = value ^ (value >> 16);
        value *= 0x21f0aaad;
        value ^= value >> 15;
        value *= 0xd35a2d97;
        value ^= value >> 15;

        self.current = value;
    }

    fn write(&mut self, bytes: &[u8]) {
        let value = if bytes.len() < 4 {
            let mut padded_bytes = vec![0; 4];
            padded_bytes[0..bytes.len()].clone_from_slice(bytes);
            unsafe { u32::from_ne_bytes(padded_bytes[0..4].try_into().unwrap_unchecked()) }
        } else {
            unsafe { u32::from_ne_bytes(bytes[0..4].try_into().unwrap_unchecked()) }
        };

        self.write_u32(value);
    }

    fn finish(&self) -> u64 {
        self.current as u64
    }
}
