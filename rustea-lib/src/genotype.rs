use crate::gene::Allele;
use crate::types::FromIteratorUnsafe;
use std::fmt::Debug;
use std::marker::PhantomData;

#[macro_export]
macro_rules! impl_genotype_with_len {
    (for $($t:ty;$g:ty),+) => {
        $(impl Genotype<$g> for $t {
            fn len(&self) -> usize {
                <$t>::len(&self)
            }

            fn get(&self, index: usize) -> $g {
                self[index]
            }

            fn set(&mut self, index: usize, bit: $g) {
                self[index] = bit
            }
        })*
    }
}

#[macro_export]
macro_rules! impl_genotype_static_size_with_len {
    (for $($t:ty;$g:ty),+) => {
        $(impl<const N: usize> Genotype<$g> for $t {
            fn len(&self) -> usize {
                <$t>::len(&self)
            }

            fn get(&self, index: usize) -> $g {
                self[index]
            }

            fn set(&mut self, index: usize, bit: $g) {
                self[index] = bit
            }
        })*

        $(impl<const N: usize> FixedSizeGenotype<$g> for $t {
            const LEN: usize = N;
        })*
    }
}

#[macro_export]
macro_rules! impl_genotype_static_size {
    (for $($t:ty;$g:ty),+) => {
        $(impl<const N: usize> Genotype<$g> for $t {
            fn len(&self) -> usize {
                N
            }

            fn get(&self, index: usize) -> $g {
                self[index]
            }

            fn set(&mut self, index: usize, bit: $g) {
                self[index] = bit
            }
        })*

        $(impl<const N: usize> FixedSizeGenotype<$g> for $t {
            const LEN: usize = N;
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

pub trait Genotype<A>: Sized + Send + Sync + Clone + Debug
where
    A: Allele,
{
    fn get(&self, index: usize) -> A;

    fn set(&mut self, index: usize, gene: A);

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

pub trait FixedSizeGenotype<A>: Genotype<A> + FromIteratorUnsafe<A>
where
    A: Allele,
{
    const LEN: usize;
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

#[macro_export]
macro_rules! impl_genotype_for_vec {
    (for $($g:ty),+) => {
        $(
            impl_genotype_with_len!(for Vec<$g>; $g);
        )*
    };
}

#[macro_export]
macro_rules! impl_genotype_for_array {
    (for $($g:ty),+) => {
        $(
            impl_genotype_static_size!(for [$g; N]; $g);
        )*
    };
}

impl_genotype_for_array!(for bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize, f32, f64);
impl_genotype_for_vec!(for bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize, f32, f64);

#[cfg(feature = "arrayvec")]
mod arrayvec {
    use arrayvec::ArrayVec;

    #[macro_export]
    macro_rules! impl_genotype_for_arrayvec {
    (for $($g:ty),+) => {
        $(
            impl_genotype_static_size_with_len!(for ArrayVec<$g, N>; $g);
        )*
    };
}

    impl_genotype_for_arrayvec!(for bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize, f32, f64);
}

// pub(crate) use impl_cartesian_genotype_for_vec_types;
