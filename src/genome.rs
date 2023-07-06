use crate::gene::{Allele, BoolDomain, Discrete, DiscreteDomain, DiscreteGene, Gene};

use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    slice::Iter,
};

use ndarray::{Array, Ix1};
use rand::Rng;

pub trait Permutation: Send + Sync {}
pub trait Cartesian<Gene>: Send + Sync {
    fn set(&mut self, index: usize, gene: Gene);
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

// A genome represents the domain of all possible genotypes
// Each gene in the genome has a range with possible values the genes could take
#[derive(Debug, Clone)]
pub struct Genome<A, G>
where
    A: Allele,
    G: Gene<A>,
{
    genes: Vec<G>,
    _allele: PhantomData<A>,
}

impl<A, G> Genome<A, G>
where
    A: Allele,
    G: Gene<A>,
{
    pub fn sample_uniform<Gnt, R>(&self, rng: &mut R) -> Gnt
    where
        R: Rng + ?Sized,
        Gnt: Genotype<A> + Sized,
    {
        self.iter().map(|gene| gene.sample_uniform(rng)).collect()
    }

    fn get(&self, index: usize) -> &G {
        &self.genes[index]
    }

    fn len(&self) -> usize {
        self.genes.len()
    }

    fn iter(&self) -> Iter<'_, G> {
        self.genes.iter()
    }
}

impl<A, G> FromIterator<G> for Genome<A, G>
where
    A: Allele,
    G: Gene<A>,
{
    fn from_iter<T: IntoIterator<Item = G>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
            _allele: PhantomData::default(),
        }
    }
}

impl<A, D> Genome<A, DiscreteGene<A, D>>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
{
    pub fn discrete_genome_with_domain(domain: &D, size: usize) -> Self {
        Self {
            genes: (0..size)
                .map(|_| DiscreteGene::with_domain(domain))
                .collect(),
            _allele: PhantomData::default(),
        }
    }
}

impl Genome<bool, DiscreteGene<bool, BoolDomain>> {
    pub fn bool_genome(size: usize) -> Self {
        Self {
            genes: (0..size)
                .map(|_| DiscreteGene::with_domain(&BoolDomain::default()))
                .collect(),
            _allele: PhantomData::default(),
        }
    }
}

pub trait Genotype<T>: FromIterator<T> + Send + Sync + Clone + Debug
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

pub trait BitString: Genotype<bool> + Cartesian<bool> {
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
    (for $($g:ty),+) => {
        $(
            impl_genotype!(for Vec<$g>; $g, Array<$g, Ix1>; $g);
            impl_cartesian!(for Vec<$g>; $g, Array<$g, Ix1>; $g);

        )*
    };
}

impl_cartesian_genotype_for_vec_types!(for bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize, f32, f64);

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
