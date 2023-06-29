use std::marker::PhantomData;

use rand::Rng;

use crate::genome::{Genome, RandomInit};

#[derive(Debug)]
pub struct Individual<G, Gene, F>
where
    G: Genome<Gene>,
    F: Default + Copy,
{
    genotype: G,
    fitness: F,
    _gene: PhantomData<Gene>,
}

impl<G, Gene, F> Individual<G, Gene, F>
where
    G: Genome<Gene>,
    F: Default + Copy,
{
    pub fn from_genotype(genotype: G) -> Self {
        Individual {
            genotype,
            fitness: F::default(),
            _gene: PhantomData::default(),
        }
    }

    pub fn genotype(&self) -> &G {
        &self.genotype
    }

    pub fn fitness(&self) -> F {
        self.fitness
    }

    pub fn update_fitness(&mut self, fitness: F) {
        self.fitness = fitness
    }
}

impl<G, Gene, F> Individual<G, Gene, F>
where
    G: Genome<Gene> + RandomInit,
    F: Default + Copy,
{
    pub fn random<R>(rng: &mut R, len: usize) -> Self
    where
        R: Rng + ?Sized,
    {
        let genotype = G::random(rng, len);
        Individual {
            genotype,
            fitness: F::default(),
            _gene: PhantomData::default(),
        }
    }
}

impl<G, Gene, F> Clone for Individual<G, Gene, F>
where
    G: Genome<Gene>,
    F: Default + Copy,
{
    fn clone(&self) -> Self {
        Self {
            genotype: self.genotype.clone(),
            fitness: self.fitness.clone(),
            _gene: PhantomData::default(),
        }
    }
}
