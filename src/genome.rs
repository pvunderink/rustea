use std::{
    fmt::Display,
    marker::PhantomData,
    ops::{Range, RangeInclusive},
    time::Duration,
};

use ndarray::{Array, Ix1};
use rand::Rng;

pub trait Discrete: Send + Sync {}
pub trait Real: Send + Sync {}
pub trait Permutation: Send + Sync {}
pub trait Cartesian<Gene>: Send + Sync {
    fn set(&mut self, index: usize, gene: Gene);
}

macro_rules! impl_trait {
    ($tr:ty => for $($t:ty),+) => {
        $(impl $tr for $t {})*
    }
  }

macro_rules! impl_cartesian {
    (for $($t:ty;$g:ty),+) => {
        $(impl Cartesian<$g> for $t {
            fn set(&mut self, index: usize, bit: $g) {
                self[index] = bit
            }
        })*
    }
}

macro_rules! impl_genotype {
    (for $($t:ty;$g:ty),+) => {
        $(impl Genotype<$g> for $t {
            fn get(&self, index: usize) -> $g {
                self[index]
            }

            fn len(&self) -> usize {
                self.len()
            }
        })*
    }
  }

macro_rules! impl_genome {
    (for $($t:ty;$g:ty),+) => {
        $(
            impl Genome<$g> for $t
            {
                fn get(&self, index: usize) -> &Gene<$g> {
                    &self[index]
                }

                fn len(&self) -> usize {
                    self.len()
                }
            }
        )*
    }
  }

#[derive(Clone, Debug)]
pub enum GeneRange<T> {
    Inclusive(RangeInclusive<T>),
    Exclusive(Range<T>),
}

#[macro_export]
macro_rules! range {
    ($l:literal..$u:literal) => {
        GeneRange::Exclusive($l..$u)
    };
    ($l:literal..=$u:literal) => {
        GeneRange::Inclusive($l..=$u)
    };
}

pub trait SampleUniformRange {
    fn sample_from_range<R>(rng: &mut R, range: GeneRange<Self>) -> Self
    where
        Self: Sized,
        R: Rng + ?Sized;
}

impl SampleUniformRange for bool {
    fn sample_from_range<R>(rng: &mut R, range: GeneRange<Self>) -> Self
    where
        Self: Sized,
        R: Rng + ?Sized,
    {
        match range {
            GeneRange::Inclusive(range) => {
                if !range.contains(&false) {
                    return true;
                } else if !range.contains(&true) {
                    return false;
                } else {
                    return rng.gen();
                }
            }
            GeneRange::Exclusive(range) => {
                if !range.contains(&false) {
                    return true;
                } else if !range.contains(&true) {
                    return false;
                } else {
                    return rng.gen();
                }
            }
        }
    }
}

macro_rules! impl_sample_uniform {
    (for $($t:ty),+) => {
        $(
            impl SampleUniformRange for $t {
                fn sample_from_range<R>(rng: &mut R, range: GeneRange<Self>) -> Self
                where
                    R: Rng + ?Sized,
                {
                    match range {
                        GeneRange::Inclusive(range) => rng.gen_range(range),
                        GeneRange::Exclusive(range) => rng.gen_range(range),
                    }
                }
            }
        )*
    }
}

impl_sample_uniform!(for u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize, f32, f64, char, Duration);

// The gene type represents the domain of values that a specific gene can take
#[derive(Clone)]
pub struct Gene<T>
where
    T: Copy + Send + Sync + SampleUniformRange,
{
    range: GeneRange<T>,
    _gene: PhantomData<T>,
}

impl<T> Gene<T>
where
    T: Copy + Send + Sync + SampleUniformRange,
{
    pub fn with_range(range: GeneRange<T>) -> Self {
        Gene {
            range,
            _gene: PhantomData::default(),
        }
    }

    pub fn range(&self) -> GeneRange<T> {
        self.range.clone()
    }
}

// A genome represents the domain of all possible genotypes
// Each gene in the genome has a range with possible values the genes could take
pub trait Genome<T>:
    FromIterator<Gene<T>> + IntoIterator<Item = Gene<T>> + Send + Sync + Clone
where
    T: Copy + Send + Sync + SampleUniformRange,
{
    fn uniform_with_range(len: usize, range: GeneRange<T>) -> Self {
        (0..len).map(|_| Gene::with_range(range.clone())).collect()
    }

    fn sample_uniform<Gnt, R>(&self, rng: &mut R) -> Gnt
    where
        R: Rng + ?Sized,
        Gnt: Genotype<T> + Sized,
    {
        self.iter()
            .map(|gene| T::sample_from_range(rng, gene.range.clone()))
            .collect()
    }

    fn get(&self, index: usize) -> &Gene<T>;

    fn len(&self) -> usize;

    fn iter(&self) -> GenomeIter<Self, T>
    where
        Self: Sized,
    {
        GenomeIter {
            genome: self,
            index: 0,
            _gene: PhantomData::default(),
        }
    }
}

pub struct GenomeIter<'a, G, T>
where
    G: Genome<T>,
    T: Copy + Send + Sync + SampleUniformRange,
{
    genome: &'a G,
    index: usize,
    _gene: PhantomData<T>, // TODO: this can probably be done more cleanly; without PhantomData
}

impl<'a, G, T> Iterator for GenomeIter<'a, G, T>
where
    G: Genome<T>,
    T: 'a + Copy + Send + Sync + SampleUniformRange,
{
    type Item = &'a Gene<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.index >= self.genome.len() {
            None
        } else {
            Some(self.genome.get(self.index))
        };
        self.index += 1;

        return result;
    }
}

pub trait Genotype<T>: FromIterator<T> + Send + Sync + Clone
where
    T: Copy + Send + Sync,
{
    fn get(&self, index: usize) -> T;

    fn len(&self) -> usize;

    fn iter(&self) -> GenotypeIter<Self, T>
    where
        Self: Sized,
    {
        GenotypeIter {
            genotype: self,
            index: 0,
            _gene: PhantomData::default(),
        }
    }
}

pub struct GenotypeIter<'a, G, T>
where
    G: Genotype<T>,
    T: Copy + Send + Sync,
{
    genotype: &'a G,
    index: usize,
    _gene: PhantomData<T>, // TODO: this can probably be done more cleanly; without PhantomData
}

impl<'a, G, T> Iterator for GenotypeIter<'a, G, T>
where
    G: Genotype<T>,
    T: Copy + Send + Sync,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.index >= self.genotype.len() {
            None
        } else {
            Some(self.genotype.get(self.index))
        };
        self.index += 1;

        return result;
    }
}

pub trait BitString: Genotype<bool> + Discrete + Cartesian<bool> {
    fn zeros(len: usize) -> Self
    where
        Self: Sized;

    fn ones(len: usize) -> Self
    where
        Self: Sized;

    fn flip(&mut self, index: usize);
}

#[derive(Debug)]
pub struct U8BitString {
    bytes: Vec<u8>,
    len: usize,
}

// impl RandomInit for U8BitString {
//     fn random<R>(rng: &mut R, len: usize) -> Self
//     where
//         R: Rng + ?Sized,
//     {
//         let num_bytes = len / 8 + if len % 8 == 0 { 0 } else { 1 };
//         let mut bytes = vec![0x00; num_bytes];
//         rng.fill_bytes(&mut bytes);

//         Self { bytes, len }
//     }
// }

impl Clone for U8BitString {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
            len: self.len,
        }
    }
}

impl Genotype<bool> for U8BitString {
    fn get(&self, index: usize) -> bool {
        assert!(index < self.len);
        let byte_index = index / 8;
        let bit_index = index % 8;
        self.bytes[byte_index] & (0x1 << (7 - bit_index)) == 0x1 << (7 - bit_index)
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl Cartesian<bool> for U8BitString {
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

    fn flip(&mut self, index: usize) {
        assert!(index < self.len);

        let byte_index = index / 8;
        let bit_index = index % 8;
        let mut byte = self.bytes[byte_index];

        // flip the bit at bit_index to 0
        byte = byte ^ (0x1 << (7 - bit_index));

        self.bytes[byte_index] = byte;
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

impl BitString for Vec<bool> {
    fn zeros(len: usize) -> Self {
        vec![false; len]
    }

    fn ones(len: usize) -> Self {
        vec![true; len]
    }

    fn flip(&mut self, index: usize) {
        self[index] = !self[index]
    }
}

impl BitString for Array<bool, Ix1> {
    fn zeros(len: usize) -> Self {
        Array::from_elem(len, false)
    }

    fn ones(len: usize) -> Self {
        Array::from_elem(len, true)
    }

    fn flip(&mut self, index: usize) {
        self[index] = !self[index]
    }
}

macro_rules! impl_cartesian_genotype_for_vec_types {
    ($tr:ty => for $($g:ty),+) => {
        $(
            impl_genome!(for Vec<Gene<$g>>; $g, Array<Gene<$g>, Ix1>; $g);
            impl_genotype!(for Vec<$g>; $g, Array<$g, Ix1>; $g);
            impl_cartesian!(for Vec<$g>; $g, Array<$g, Ix1>; $g);
            impl_trait!($tr => for Vec<$g>, Array<$g, Ix1>);
        )*
    };
}

impl_cartesian_genotype_for_vec_types!(Discrete => for bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize);
impl_cartesian_genotype_for_vec_types!(Real => for f32, f64);
impl_trait!(Discrete => for U8BitString);

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
