use std::fmt::Display;

use ndarray::{Array, Ix1};
use rand::Rng;

pub trait BitString: Send + Sync + FromIterator<bool> {
    fn zeros(len: usize) -> Self
    where
        Self: Sized;

    fn ones(len: usize) -> Self
    where
        Self: Sized;

    fn random<R>(rng: &mut R, len: usize) -> Self
    where
        Self: Sized,
        R: Rng + ?Sized;

    fn get(&self, index: usize) -> bool;
    fn set(&mut self, index: usize, bit: bool);
    fn len(&self) -> usize;
    fn flip(&mut self, index: usize);

    fn iter(&self) -> BitStringIter<Self>
    where
        Self: Sized,
    {
        BitStringIter {
            bitstring: self,
            index: 0,
        }
    }

    fn clone(&self) -> Self
    where
        Self: Sized;
}

pub struct BitStringIter<'a, B>
where
    B: BitString,
{
    bitstring: &'a B,
    index: usize,
}

impl<'a, B> Iterator for BitStringIter<'a, B>
where
    B: BitString,
{
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.index >= self.bitstring.len() {
            None
        } else {
            Some(self.bitstring.get(self.index))
        };
        self.index += 1;

        return result;
    }
}

#[derive(Debug)]
pub struct U8BitString {
    bytes: Vec<u8>,
    len: usize,
}

impl BitString for U8BitString {
    fn zeros(len: usize) -> Self {
        let num_bytes = len / 8 + if len % 8 == 0 { 0 } else { 1 };
        let bytes = vec![0x00; num_bytes];

        Self { bytes, len }
    }

    fn ones(len: usize) -> Self {
        let num_bytes = len / 8 + if len % 8 == 0 { 0 } else { 1 };
        let bytes = vec![0xFF; num_bytes];

        Self { bytes, len }
    }

    fn random<R>(rng: &mut R, len: usize) -> Self
    where
        R: Rng + ?Sized,
    {
        let num_bytes = len / 8 + if len % 8 == 0 { 0 } else { 1 };
        let mut bytes = vec![0x00; num_bytes];
        rng.fill_bytes(&mut bytes);

        Self { bytes, len }
    }

    fn get(&self, index: usize) -> bool {
        assert!(index < self.len);
        let byte_index = index / 8;
        let bit_index = index % 8;
        self.bytes[byte_index] & (0x1 << (7 - bit_index)) == 0x1 << (7 - bit_index)
    }

    fn set(&mut self, index: usize, bit: bool) {
        assert!(index < self.len);
        let byte_index = index / 8;
        let bit_index = index % 8;

        let mut byte = self.bytes[byte_index];
        // reset the bit at bit_index to 0
        byte = byte & !(0x1 << (7 - bit_index));

        if bit {
            // set the bit at bit_index to 1
            byte = byte | 0x1 << (7 - bit_index)
        }

        self.bytes[byte_index] = byte;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn flip(&mut self, index: usize) {
        assert!(index < self.len);

        let byte_index = index / 8;
        let bit_index = index % 8;
        let mut byte = self.bytes[byte_index];

        // flip the bit at bit_index to 0
        byte = byte ^ (0x1 << (7 - bit_index));

        self.bytes[byte_index] = byte;
    }

    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
            len: self.len,
        }
    }
}

impl Display for U8BitString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = self
            .iter()
            .map(|bit| if bit { '1' } else { '0' })
            .collect::<String>();

        f.write_str(&str)
    }
}

impl From<Vec<bool>> for U8BitString {
    fn from(value: Vec<bool>) -> Self {
        let mut bitstring = Self::zeros(value.len());

        for (i, b) in value.iter().enumerate() {
            bitstring.set(i, b);
        }

        bitstring
    }
}

impl FromIterator<bool> for U8BitString {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let vec: Vec<_> = iter.into_iter().collect();
        vec.into()
    }
}

impl BitString for Array<bool, Ix1> {
    fn zeros(len: usize) -> Self {
        Array::from_elem(len, false)
    }

    fn ones(len: usize) -> Self {
        Array::from_elem(len, true)
    }

    fn random<R>(rng: &mut R, len: usize) -> Self
    where
        R: Rng + ?Sized,
    {
        (0..len).map(|_| rng.gen_bool(0.5)).collect()
    }

    fn get(&self, index: usize) -> bool {
        self[index]
    }

    fn set(&mut self, index: usize, bit: bool) {
        self[index] = bit
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn flip(&mut self, index: usize) {
        self[index] = !self[index]
    }

    fn clone(&self) -> Self {
        Clone::clone(&self)
    }
}

impl BitString for Vec<bool> {
    fn zeros(len: usize) -> Self {
        vec![false; len]
    }

    fn ones(len: usize) -> Self {
        vec![true; len]
    }

    fn random<R>(rng: &mut R, len: usize) -> Self
    where
        R: Rng + ?Sized,
    {
        (0..len).map(|_| rng.gen_bool(0.5)).collect()
    }

    fn get(&self, index: usize) -> bool {
        self[index]
    }

    fn set(&mut self, index: usize, bit: bool) {
        self[index] = bit
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn flip(&mut self, index: usize) {
        self[index] = !self[index]
    }

    fn clone(&self) -> Self {
        Clone::clone(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_bitstring_of_zeros() {
        let zeros = U8BitString::zeros(12);

        zeros.iter().for_each(|bit| assert_eq!(bit, false));
    }

    #[test]
    fn create_bitstring_of_ones() {
        let zeros = U8BitString::ones(12);

        zeros.iter().for_each(|bit| assert_eq!(bit, true));
    }

    #[test]
    fn set_bit() {
        let mut zeros = U8BitString::zeros(15);
        zeros.set(2, true);

        assert_eq!(zeros.get(2), true);

        zeros.set(2, false);

        assert_eq!(zeros.get(2), false);
    }

    #[test]
    fn flip_bit() {
        let mut zeros = U8BitString::zeros(8);
        zeros.flip(4);

        assert_eq!(zeros.get(4), true);

        zeros.flip(4);

        assert_eq!(zeros.get(4), false);
    }
}
