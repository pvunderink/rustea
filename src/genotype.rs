use crate::gene::Allele;
use crate::genome::Cartesian;
use crate::types::FromIteratorUnsafe;
use std::fmt::Debug;
use std::marker::PhantomData;

macro_rules! impl_cartesian {
    (for $($t:ty;$g:ty),+) => {
        $(impl<const N: usize> Cartesian<$g> for $t {
            fn set(&mut self, index: usize, bit: $g) {
                self[index] = bit
            }
        })*
    }
}

macro_rules! impl_genotype {
    (for $($t:ty;$g:ty),+) => {
        $(impl<const N: usize> Genotype<$g> for $t {
            const LEN:usize = N;

            fn get(&self, index: usize) -> $g {
                self[index]
            }
        })*
    }
}

pub trait Genotype<A>: Sized + Send + Sync + Clone + Debug + FromIteratorUnsafe<A>
where
    A: Allele,
{
    const LEN: usize;

    fn get(&self, index: usize) -> A;

    fn len(&self) -> usize {
        Self::LEN
    }

    fn iter(&self) -> GenotypeIter<Self, A>
    where
        Self: Sized,
    {
        GenotypeIter {
            genotype: self,
            index: 0,
            _allele: PhantomData,
        }
    }
}

pub struct GenotypeIter<'a, G, A>
where
    G: Genotype<A>,
    A: Allele,
{
    genotype: &'a G,
    index: usize,
    _allele: PhantomData<A>, // TODO: this can probably be done more cleanly; without PhantomData
}

impl<'a, G, A> Iterator for GenotypeIter<'a, G, A>
where
    G: Genotype<A>,
    A: Allele,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.index >= self.genotype.len() {
            None
        } else {
            Some(self.genotype.get(self.index))
        };
        self.index += 1;

        result
    }
}

macro_rules! impl_cartesian_genotype_for_vec_types {
    (for $($g:ty),+) => {
        $(
            impl_genotype!(for [$g; N]; $g);
            impl_cartesian!(for [$g; N]; $g);

        )*
    };
}

impl_cartesian_genotype_for_vec_types!(for bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize, f32, f64);
