use std::marker::PhantomData;

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
            _gene: PhantomData::default(),
        }
    }

    pub fn sample_uniform<G>(genome: &Genome<A, G>) -> Self
    where
        G: Gene<A>,
    {
        let mut rng = rand::thread_rng();

        let genotype = genome.sample_uniform(&mut rng);

        Individual {
            genotype,
            fitness: None,
            _gene: PhantomData::default(),
        }
    }

    pub fn genotype(&self) -> &Gnt {
        &self.genotype
    }

    pub fn fitness(&self) -> F {
        let Some(fitness) = self.fitness else {
            panic!("The individual has not been evaluated yet");
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
            fitness: self.fitness.clone(),
            _gene: PhantomData::default(),
        }
    }
}
