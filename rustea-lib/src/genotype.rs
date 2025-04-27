use crate::gene::Allele;
use crate::genome::Cartesian;
use crate::types::FromIteratorUnsafe;
use arrayvec::ArrayVec;
use std::fmt::Debug;
use std::marker::PhantomData;

#[macro_export]
macro_rules! impl_cartesian {
    (for $($t:ty;$g:ty),+) => {
        $(impl<const N: usize> Cartesian<$g> for $t {
            fn set(&mut self, index: usize, bit: $g) {
                self[index] = bit
            }
        })*
    }
}

#[macro_export]
macro_rules! impl_genotype {
    (for $($t:ty;$g:ty),+) => {
        $(impl<const N: usize> Genotype<$g> for $t {
            const LEN: usize = N;

            fn len(&self) -> usize {
                <$t>::len(&self)
            }

            fn get(&self, index: usize) -> $g {
                self[index]
            }
        })*
    }
}

#[macro_export]
macro_rules! impl_genotype2 {
    (for $($t:ty;$g:ty),+) => {
        $(impl<const N: usize> Genotype<$g> for $t {
            const LEN: usize = N;

            fn len(&self) -> usize {
                N
            }

            fn get(&self, index: usize) -> $g {
                self[index]
            }
        })*
    }
}

// impl<const N: usize> Genotype<bool> for ArrayVec<bool, N> {
//     const LEN: usize = N;

//     fn len(&self) -> usize {
//         ArrayVec::<bool, N>::len(&self)
//     }

//     fn get(&self, index: usize) -> bool {
//         self[index]
//     }
// }

pub trait Genotype<A>: Sized + Send + Sync + Clone + Debug + FromIteratorUnsafe<A>
where
    A: Allele,
{
    const LEN: usize;

    fn get(&self, index: usize) -> A;

    fn len(&self) -> usize;

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

#[derive(Debug, Clone)]
pub struct SizedVec<T, const N: usize> {
    vec: Vec<T>,
}

impl<T, const N: usize> SizedVec<T, N> {
    pub fn array_chunks<const K: usize>(&self) -> core::slice::ArrayChunks<'_, T, K> {
        self.vec.array_chunks()
    }
}

impl<T, const N: usize> IntoIterator for SizedVec<T, N> {
    type Item = T;

    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl<T, const N: usize> FromIteratorUnsafe<T> for SizedVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            vec: Vec::from_iter(iter),
        }
    }
}

impl<T, const N: usize> Genotype<T> for SizedVec<T, N>
where
    T: Allele,
{
    const LEN: usize = N;

    fn len(&self) -> usize {
        self.vec.len()
    }

    fn get(&self, index: usize) -> T {
        self.vec[index]
    }
}

impl<T, const N: usize> Cartesian<T> for SizedVec<T, N>
where
    T: Allele,
{
    fn set(&mut self, index: usize, gene: T) {
        self.vec[index] = gene;
    }
}

#[macro_export]
macro_rules! impl_cartesian_genotype_for_array {
    (for $($g:ty),+) => {
        $(
            impl_genotype2!(for [$g; N]; $g);
            impl_cartesian!(for [$g; N]; $g);

        )*
    };
}

#[macro_export]
macro_rules! impl_cartesian_genotype_for_arrayvec {
    (for $($g:ty),+) => {
        $(
            impl_genotype!(for ArrayVec<$g, N>; $g);
            impl_cartesian!(for ArrayVec<$g, N>; $g);

        )*
    };
}

impl_cartesian_genotype_for_arrayvec!(for bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize, f32, f64);
impl_cartesian_genotype_for_array!(for bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize, f32, f64);

// pub(crate) use impl_cartesian_genotype_for_vec_types;
