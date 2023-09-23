use crate::{
    gene::{Allele, BoolDomain, Discrete, DiscreteDomain, DiscreteGene, Gene},
    genotype::Genotype,
    types::CollectUnsafe,
};

use std::{fmt::Debug, marker::PhantomData, slice::Iter};

use rand::Rng;

pub trait Permutation: Send + Sync {}
pub trait Cartesian<Gene>: Send + Sync {
    fn set(&mut self, index: usize, gene: Gene);
}

// A genome represents the domain of all possible genotypes
// Each gene in the genome has a range with possible values the genes could take
#[derive(Debug, Clone)]
pub struct Genome<Gnt, A, G>
where
    A: Allele,
    G: Gene<A>,
    Gnt: Genotype<A>,
{
    genes: Vec<G>,
    _allele: PhantomData<A>,
    _genotype: PhantomData<Gnt>,
}

impl<Gnt, A, G> Genome<Gnt, A, G>
where
    A: Allele,
    G: Gene<A>,
    Gnt: Genotype<A>,
{
    pub fn sample_uniform<R>(&self, rng: &mut R) -> Gnt
    where
        R: Rng + ?Sized,
        Gnt: Genotype<A> + Sized,
    {
        self.iter()
            .map(|gene| gene.sample_uniform(rng))
            .collect_unsafe()
    }

    pub fn get(&self, index: usize) -> &G {
        &self.genes[index]
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn iter(&self) -> Iter<'_, G> {
        self.genes.iter()
    }
}

impl<Gnt, A, D> Genome<Gnt, A, DiscreteGene<A, D>>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    Gnt: Genotype<A>,
{
    pub fn with_discrete_domain(domain: &D) -> Self {
        Self {
            genes: (0..Gnt::LEN)
                .map(|_| DiscreteGene::with_domain(domain))
                .collect(),
            _allele: PhantomData,
            _genotype: PhantomData,
        }
    }
}

impl<Gnt> Genome<Gnt, bool, DiscreteGene<bool, BoolDomain>>
where
    Gnt: Genotype<bool>,
{
    pub fn with_bool_domain() -> Self {
        Self {
            genes: (0..Gnt::LEN)
                .map(|_| DiscreteGene::with_domain(&BoolDomain))
                .collect(),
            _allele: PhantomData,
            _genotype: PhantomData,
        }
    }
}
