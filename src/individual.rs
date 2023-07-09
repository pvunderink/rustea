use std::marker::PhantomData;

use rand::Rng;

use crate::{
    fitness::Fitness,
    gene::{Allele, Gene},
    genome::{Genome, Genotype},
};

#[derive(Debug)]
pub struct Individual<Gnt, A, F>
where
    Gnt: Genotype<A>,
    A: Allele,
    F: Fitness,
{
    genotype: Gnt,
    fitness: Option<F>,
    _gene: PhantomData<A>,
}

impl<Gnt, A, F> Individual<Gnt, A, F>
where
    Gnt: Genotype<A>,
    A: Allele,
    F: Fitness,
{
    pub fn from_genotype(genotype: Gnt) -> Self {
        Individual {
            genotype,
            fitness: None,
            _gene: PhantomData,
        }
    }

    pub fn sample_uniform<R, G>(rng: &mut R, genome: &Genome<A, G>) -> Self
    where
        R: Rng + ?Sized,
        G: Gene<A>,
    {
        let genotype = genome.sample_uniform(rng);

        Individual {
            genotype,
            fitness: None,
            _gene: PhantomData,
        }
    }

    pub fn genotype(&self) -> &Gnt {
        &self.genotype
    }

    pub fn fitness(&self) -> F {
        let Some(fitness) = self.fitness else {
            panic!("Cannot retreive fitness: the individual has not been evaluated yet");
        };
        fitness
    }

    pub fn set_fitness(&mut self, fitness: F) {
        self.fitness = Some(fitness)
    }
}

impl<Gnt, A, F> Clone for Individual<Gnt, A, F>
where
    Gnt: Genotype<A>,
    A: Allele,
    F: Fitness,
{
    fn clone(&self) -> Self {
        Self {
            genotype: self.genotype.clone(),
            fitness: self.fitness,
            _gene: PhantomData,
        }
    }
}
