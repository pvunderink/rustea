use std::{
    collections::HashSet,
    hash::Hash,
    marker::PhantomData,
    ops::{Range, RangeInclusive},
};

use core::fmt::Debug;

use approx::AbsDiffEq;
use rand::{distributions::uniform::SampleUniform, Rng};

// TODO: Also implement a trait for uniform sampling
pub trait Gene<A>: Send + Sync + Clone {
    fn sample_uniform<R>(&self, rng: &mut R) -> A
    where
        R: Rng + ?Sized;
}

pub trait Allele: Send + Sync + Copy + Debug {}

// Marker for discrete-valued genes and alleles
pub trait Discrete: Eq + PartialEq + Hash {}

pub trait Integer: Discrete + PartialOrd + Ord + SampleUniform {}

// Marker for real-valued genes and alleles
pub trait Real: PartialOrd + AbsDiffEq + SampleUniform {}

#[derive(Clone)]
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
            _allele: PhantomData::default(),
        }
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
    fn len(&self) -> usize;
    fn iter(&self) -> DiscreteDomainIter<A, Self>
    where
        Self: Sized,
    {
        DiscreteDomainIter {
            domain: self,
            index: 0,
            _allele: PhantomData::default(),
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

#[derive(Clone)]
pub struct IntegralDomain<A>
where
    A: Allele + Discrete + Ord + PartialOrd,
{
    alleles: Vec<A>,
}

impl<A> FromIterator<A> for IntegralDomain<A>
where
    A: Allele + Discrete + Ord + PartialOrd,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        Self {
            alleles: iter.into_iter().collect(),
        }
    }
}

impl<A> DiscreteDomain<A> for IntegralDomain<A>
where
    A: Allele + Discrete + Ord + PartialOrd,
{
    fn get(&self, idx: usize) -> A {
        self.alleles[idx]
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

impl<A> IntegralDomain<A>
where
    A: Allele + Discrete + Ord + PartialOrd,
{
    pub fn empty() -> Self {
        Self {
            alleles: Vec::new(),
        }
    }
}

impl<A> Into<Vec<A>> for IntegralDomain<A>
where
    A: Allele + Discrete + Ord + PartialOrd,
{
    fn into(self) -> Vec<A> {
        self.alleles
    }
}

#[macro_export]
macro_rules! idom {
    (@($dom:expr); $l:literal..$h:literal, $($rest:tt)*) => {
        idom!(@($dom.union(IntegralDomain::from_range($l..$h))); $($rest)*)
    };
    (@($dom:expr); $l:literal..=$h:literal, $($rest:tt)*) => {
        idom!(@($dom.union(IntegralDomain::from_inclusive_range($l..=$h))); $($rest)*)
    };
    (@($dom:expr); $l:literal..$h:literal) => {
        idom!(@($dom.union(IntegralDomain::from_range($l..$h))))
    };
    (@($dom:expr); $l:literal..=$h:literal) => {
        idom!(@($dom.union(IntegralDomain::from_inclusive_range($l..=$h))))
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
        idom!(@(IntegralDomain::empty()); $($t)*)
    };
}

#[derive(Clone)]
pub struct BoolDomain {
    values: Vec<bool>,
}

impl FromIterator<bool> for BoolDomain {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let values: Vec<_> = iter
            .into_iter()
            .fold(HashSet::new(), |mut acc, b| {
                acc.insert(b);
                acc
            })
            .into_iter()
            .collect();

        Self { values }
    }
}

impl DiscreteDomain<bool> for BoolDomain {
    fn get(&self, idx: usize) -> bool {
        self.values[idx]
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn add(self, allele: bool) -> Self {
        if self.values.contains(&allele) {
            let mut values = self.values.clone();
            values.push(allele);

            Self { values }
        } else {
            self
        }
    }
}

impl Default for BoolDomain {
    fn default() -> Self {
        Self {
            values: vec![false, true],
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
        let domain1 = IntegralDomain::from_inclusive_range(1..=3);
        let domain2 = IntegralDomain::from_range(4..6);

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
    fn test_boolean_domain_false() {
        let domain: BoolDomain = [false].into_iter().collect();

        assert_eq!(domain.get(0), false)
    }

    #[test]
    fn test_boolean_domain_true() {
        let domain: BoolDomain = [true].into_iter().collect();

        assert_eq!(domain.get(0), true)
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
