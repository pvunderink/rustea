use std::{
    collections::HashSet,
    hash::Hash,
    marker::PhantomData,
    ops::{Add, Range, RangeInclusive},
};

use core::{fmt::Debug, panic};

use approx::AbsDiffEq;
use num_traits::{One, Zero};
use rand::{distributions::uniform::SampleUniform, Rng};
use rand_distr::{Distribution, WeightedIndex};

// TODO: Also implement a trait for uniform sampling
pub trait Gene<A>: Send + Sync + Clone {
    fn sample_uniform<R>(&self, rng: &mut R) -> A
    where
        R: Rng + ?Sized;
}

#[derive(Clone)]
pub struct DummyGene<A: Default> {
    _marker: PhantomData<A>,
}
impl<A: Allele> Gene<A> for DummyGene<A> {
    fn sample_uniform<R>(&self, rng: &mut R) -> A
    where
        R: Rng + ?Sized,
    {
        A::default()
    }
}

pub trait Allele: Sized + Send + Sync + Copy + Debug + Default {}

// Marker for discrete-valued genes and alleles
pub trait Discrete: Eq + PartialEq + Hash {}

pub trait Integer: Discrete + PartialOrd + Ord + SampleUniform {}

// Marker for real-valued genes and alleles
pub trait Real: PartialOrd + AbsDiffEq + SampleUniform {}

#[derive(Debug, Clone)]
pub struct DiscreteGene<A, D>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
{
    domain: D,
    _allele: PhantomData<A>,
}

impl<A, D> Gene<A> for DiscreteGene<A, D>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
{
    fn sample_uniform<R>(&self, rng: &mut R) -> A
    where
        R: Rng + ?Sized,
    {
        self.domain.sample_uniform(rng)
    }
}

impl<A, D> DiscreteGene<A, D>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
{
    pub fn with_domain(domain: &D) -> Self {
        Self {
            domain: domain.clone(),
            _allele: PhantomData,
        }
    }

    pub fn sample_with_weights<R, W>(&self, rng: &mut R, dist: &WeightedIndex<W>) -> A
    where
        R: Rng + ?Sized,
        W: SampleUniform + PartialOrd + for<'a> ::core::ops::AddAssign<&'a W> + Clone + Default,
    {
        let idx = dist.sample(rng);

        self.domain.get(idx)
    }

    pub fn domain(&self) -> &D {
        &self.domain
    }
}

#[derive(Clone)]
pub struct RealGene {}

pub struct DiscreteDomainIter<'a, A, D>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
{
    domain: &'a D,
    index: usize,
    _allele: PhantomData<A>,
}

impl<'a, A, D> Iterator for DiscreteDomainIter<'a, A, D>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.index >= self.domain.len() {
            None
        } else {
            Some(self.domain.get(self.index))
        };
        self.index += 1;
        result
    }
}

pub trait DiscreteDomain<A>: Clone + Send + Sync + FromIterator<A>
where
    A: Allele + Discrete,
{
    fn get(&self, idx: usize) -> A;
    fn index_of(&self, allele: A) -> usize;
    fn len(&self) -> usize;
    fn iter(&self) -> DiscreteDomainIter<A, Self>
    where
        Self: Sized,
    {
        DiscreteDomainIter {
            domain: self,
            index: 0,
            _allele: PhantomData,
        }
    }
    fn from_range(range: Range<A>) -> Self
    where
        Range<A>: Iterator<Item = A>,
    {
        range.collect()
    }
    fn from_inclusive_range(range: RangeInclusive<A>) -> Self
    where
        RangeInclusive<A>: Iterator<Item = A>,
    {
        range.collect()
    }
    fn union(self, other: Self) -> Self
    where
        Self: Sized,
    {
        let mut set = HashSet::<A>::default();

        set.extend(self.iter());
        set.extend(other.iter());

        set.into_iter().collect()
    }
    fn add(self, allele: A) -> Self;
    fn sample_uniform<R>(&self, rng: &mut R) -> A
    where
        R: Rng + ?Sized,
    {
        let r: f64 = rng.gen();
        let n: usize = (r * self.len() as f64) as usize;

        self.iter().nth(n).unwrap()
    }
}

// #[derive(Clone)]
// pub struct ContinuousIntegralDomain<A>
// where
//     A: Allele + Discrete + Ord + PartialOrd + Add + Zero,
// {
//     low: A,
//     high: A,
// }

// impl<A> FromIterator<A> for ContinuousIntegralDomain<A>
// where
//     A: Allele + Discrete + Ord + PartialOrd + Add + Zero + One,
// {
//     fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
//         let mut vec: Vec<_> = iter.into_iter().collect();
//         vec.sort();

//         let Some((Some(low), Some(high))) =
//             vec.into_iter()
//                 .fold(Some((None::<A>, None::<A>)), |bounds, elem| {
//                     let (l, h) = bounds?;

//                     let new_l = match l {
//                         Some(l) => l,
//                         None => elem,
//                     };

//                     let new_h = match h {
//                         Some(h) => {
//                             // TODO: Maybe not the best idea to use A::one() for this purpose
//                             if elem == h + A::one() {
//                                 h
//                             } else {
//                                 return None;
//                             }
//                         }
//                         None => elem,
//                     };

//                     Some((Some(new_l), Some(new_h)))
//                 })
//         else {
//             panic!("Provided iterator is not a continuous range of integers");
//         };

//         Self { low, high }
//     }
// }

// impl DiscreteDomain<usize> for ContinuousIntegralDomain<usize> {
//     fn get(&self, idx: usize) -> usize {
//         let allele = self.low + idx as usize;

//         assert!(allele <= self.high);

//         allele
//     }

//     fn index_of(&self, allele: usize) -> usize {
//         allele - self.low
//     }

//     fn len(&self) -> usize {
//         (self.high - self.low) + 1
//     }

//     fn from_inclusive_range(range: RangeInclusive<usize>) -> Self
//     where
//         RangeInclusive<usize>: Iterator<Item = usize>,
//     {
//         Self {
//             low: *range.start(),
//             high: *range.end(),
//         }
//     }

//     fn from_range(range: Range<usize>) -> Self
//     where
//         Range<usize>: Iterator<Item = usize>,
//     {
//         Self {
//             low: range.start,
//             high: range.end - 1,
//         }
//     }

//     fn union(self, other: Self) -> Self
//     where
//         Self: Sized,
//     {
//         let mut set = HashSet::<usize>::default();

//         set.extend(self.iter());
//         set.extend(other.iter());

//         let mut vec: Vec<_> = set.into_iter().collect();
//         vec.sort();
//         vec.into_iter().collect()
//     }

//     fn add(self, allele: usize) -> Self {
//         if allele >= self.low && allele <= self.high {
//             return self;
//         } else if allele == self.low - 1 {
//             return Self {
//                 low: allele,
//                 high: self.high,
//             };
//         } else if allele == self.high + 1 {
//             return Self {
//                 low: self.low,
//                 high: allele,
//             };
//         }
//         panic!("The new allele must be consecutive to the continuous domain")
//     }
// }

#[derive(Debug, Clone)]
pub struct DisjointIntegralDomain<A>
where
    A: Allele + Discrete + Ord + PartialOrd + Add + Zero + One,
{
    alleles: Vec<A>,
}

impl<A> FromIterator<A> for DisjointIntegralDomain<A>
where
    A: Allele + Discrete + Ord + PartialOrd + Add + Zero + One,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        Self {
            alleles: iter.into_iter().collect(),
        }
    }
}

impl<A> DiscreteDomain<A> for DisjointIntegralDomain<A>
where
    A: Allele + Discrete + Ord + PartialOrd + Add + Zero + One,
{
    fn get(&self, idx: usize) -> A {
        self.alleles[idx]
    }

    fn index_of(&self, allele: A) -> usize {
        self.alleles.binary_search(&allele).unwrap()
    }

    fn len(&self) -> usize {
        self.alleles.len()
    }

    fn union(self, other: Self) -> Self
    where
        Self: Sized,
    {
        let mut set = HashSet::<A>::default();

        set.extend(self.iter());
        set.extend(other.iter());

        let mut vec: Vec<_> = set.into_iter().collect();
        vec.sort();
        vec.into_iter().collect()
    }

    fn add(self, allele: A) -> Self {
        match self.alleles.binary_search(&allele) {
            Ok(_) => self, // already in domain
            Err(pos) => {
                let mut copy = self.alleles.clone();

                copy.insert(pos, allele);

                copy.into_iter().collect()
            }
        }
    }
}

impl<A> DisjointIntegralDomain<A>
where
    A: Allele + Discrete + Ord + PartialOrd + Add + Zero + One,
{
    pub fn empty() -> Self {
        Self {
            alleles: Vec::new(),
        }
    }
}

impl<A> From<DisjointIntegralDomain<A>> for Vec<A>
where
    A: Allele + Discrete + Ord + PartialOrd + Add + Zero + One,
{
    fn from(val: DisjointIntegralDomain<A>) -> Self {
        val.alleles
    }
}

#[macro_export]
macro_rules! idom {
    (@($dom:expr); $l:literal..$h:literal, $($rest:tt)*) => {
        idom!(@($dom.union(DisjointIntegralDomain::from_range($l..$h))); $($rest)*)
    };
    (@($dom:expr); $l:literal..=$h:literal, $($rest:tt)*) => {
        idom!(@($dom.union(DisjointIntegralDomain::from_inclusive_range($l..=$h))); $($rest)*)
    };
    (@($dom:expr); $l:literal..$h:literal) => {
        idom!(@($dom.union(DisjointIntegralDomain::from_range($l..$h))))
    };
    (@($dom:expr); $l:literal..=$h:literal) => {
        idom!(@($dom.union(DisjointIntegralDomain::from_inclusive_range($l..=$h))))
    };
    (@($dom:expr); $l:literal, $($rest:tt)*) => {
        idom!(@($dom.add($l)); $($rest)*)
    };
    (@($dom:expr); $l:literal) => {
        idom!(@($dom.add($l)))
    };
    (@($dom:expr);) => {
        $dom
    };
    (@($dom:expr)) => {
        $dom
    };
    ($($t:tt)*) => {
        idom!(@(DisjointIntegralDomain::empty()); $($t)*)
    };
}

// #[macro_export]
// macro_rules! cidom {
//     ($l:literal..$h:literal) => {
//         ContinuousIntegralDomain::from_range($l..$h)
//     };
//     ($l:literal..=$h:literal) => {
//         ContinuousIntegralDomain::from_inclusive_range($l..=$h)
//     };
// }

#[derive(Default, Clone)]
pub struct BoolDomain;

impl FromIterator<bool> for BoolDomain {
    fn from_iter<T: IntoIterator<Item = bool>>(_: T) -> Self {
        panic!("Cannot create a BoolDomain from an iterator, use bdom!() to construct a BoolDomain")
    }
}

impl DiscreteDomain<bool> for BoolDomain {
    fn get(&self, idx: usize) -> bool {
        match idx {
            0 => false,
            1 => true,
            _ => panic!("Invalid index into BoolDomain"),
        }
    }

    fn len(&self) -> usize {
        2
    }

    fn add(self, _: bool) -> Self {
        self
    }

    fn index_of(&self, allele: bool) -> usize {
        match allele {
            true => 1,
            false => 0,
        }
    }
}

#[macro_export]
macro_rules! bdom {
    () => {
        BoolDomain::default()
    };
}

pub trait RealDomain<A>: Clone + Send + Sync
where
    A: Allele + Real,
{
    fn sample_uniform<R>(&self, rng: &mut R) -> A
    where
        R: Rng + ?Sized;
}

#[derive(Clone)]
pub struct ExclusiveRangeRealDomain<A>
where
    A: Allele + Real,
{
    range: Range<A>,
}

impl<A> ExclusiveRangeRealDomain<A>
where
    A: Allele + Real,
{
    pub fn with_range(range: Range<A>) -> Self {
        Self { range }
    }

    pub fn range(&self) -> &Range<A> {
        &self.range
    }
}

impl<A> RealDomain<A> for ExclusiveRangeRealDomain<A>
where
    A: Allele + Real,
{
    fn sample_uniform<R>(&self, rng: &mut R) -> A
    where
        R: Rng + ?Sized,
    {
        rng.gen_range(self.range.clone())
    }
}

#[derive(Clone)]
pub struct InclusiveRangeRealDomain<A>
where
    A: Allele + Real,
{
    range: RangeInclusive<A>,
}

impl<A> InclusiveRangeRealDomain<A>
where
    A: Allele + Real,
{
    pub fn with_range(range: RangeInclusive<A>) -> Self {
        Self { range }
    }

    pub fn range(&self) -> &RangeInclusive<A> {
        &self.range
    }
}

impl<A> RealDomain<A> for InclusiveRangeRealDomain<A>
where
    A: Allele + Real,
{
    fn sample_uniform<R>(&self, rng: &mut R) -> A
    where
        R: Rng + ?Sized,
    {
        rng.gen_range(self.range.clone())
    }
}

#[macro_export]
macro_rules! rdom {
    ($l:literal..$h:literal) => {
        ExclusiveRangeRealDomain::with_range($l..$h)
    };
    ($l:literal..=$h:literal) => {
        InclusiveRangeRealDomain::with_range($l..=$h)
    };
}

macro_rules! impl_discrete_allele {
    (for $($ty:ty),+) => {
        $(
            impl Allele for $ty {}
            impl Discrete for $ty {}

        )*
    };
}

macro_rules! impl_real_allele {
    (for $($ty:ty),+) => {
        $(
            impl Allele for $ty {}
            impl Real for $ty {}

        )*
    };
}

impl_discrete_allele!(for bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize);
impl_real_allele!(for f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integral_domain_union() {
        let domain1 = DisjointIntegralDomain::from_inclusive_range(1..=3);
        let domain2 = DisjointIntegralDomain::from_range(4..6);

        let vec: Vec<_> = domain1.union(domain2).into();

        assert_eq!(vec, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_idom_macro_combined() {
        let domain = idom!(1..3, 3..=6, 7);

        assert_eq!(domain.iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_idom_macro_inclusive() {
        let domain = idom!(1..=5);

        assert_eq!(domain.iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_idom_macro_exclusive() {
        let domain = idom!(1..5);

        assert_eq!(domain.iter().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_idom_macro_single_element() {
        let domain = idom!(1);

        assert_eq!(domain.iter().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn test_idom_macro_disjoint_elements() {
        let domain = idom!(1, 3, 5, 7);

        assert_eq!(domain.iter().collect::<Vec<_>>(), vec![1, 3, 5, 7]);
    }

    #[test]
    fn test_bdom_contains_false_and_true() {
        let domain = bdom!();

        assert_eq!(domain.iter().collect::<Vec<_>>(), vec![false, true])
    }

    #[test]
    fn test_rdom_exclusive() {
        let range = -1.0..1.0;
        let domain = rdom!(-1.0..1.0);

        assert_eq!(*domain.range(), range)
    }

    #[test]
    fn test_rdom_inclusive() {
        let range = -1.0..=1.0;
        let domain = rdom!(-1.0..=1.0);

        assert_eq!(*domain.range(), range)
    }
}
