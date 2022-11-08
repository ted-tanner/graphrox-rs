use std::hash::Hasher;
use std::num::Wrapping;

#[derive(Clone, Default)]
pub struct GraphRoxHasher {
    current: Wrapping<u32>,
}

impl Hasher for GraphRoxHasher {
    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.write_u32(value as u32);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.current = Wrapping(value);
        self.current ^= self.current >> 16;
        self.current *= 0x21f0aaad;
        self.current ^= self.current >> 15;
        self.current *= 0xd35a2d97;
        self.current ^= self.current >> 15;
    }

    #[inline]
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

    #[inline]
    fn finish(&self) -> u64 {
        self.current.0 as u64
    }
}
