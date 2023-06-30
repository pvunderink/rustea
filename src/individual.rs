use std::marker::PhantomData;

use crate::genome::{Genome, Genotype, SampleUniformRange};

#[derive(Debug)]
pub struct Individual<Gnt, T, F>
where
    Gnt: Genotype<T>,
    T: Copy + Send + Sync + SampleUniformRange,
    F: Default + Copy,
{
    genotype: Gnt,
    fitness: F,
    _gene: PhantomData<T>,
}

impl<Gnt, T, F> Individual<Gnt, T, F>
where
    Gnt: Genotype<T>,
    T: Copy + Send + Sync + SampleUniformRange,
    F: Default + Copy,
{
    pub fn from_genotype(genotype: Gnt) -> Self {
        Individual {
            genotype,
            fitness: F::default(),
            _gene: PhantomData::default(),
        }
    }

    pub fn sample_uniform<Gnm>(genome: &Gnm) -> Self
    where
        Gnm: Genome<T>,
    {
        let mut rng = rand::thread_rng();

        let genotype = genome
            .iter()
            .map(|gene| T::sample_from_range(&mut rng, gene.range().clone()))
            .collect();

        Individual {
            genotype,
            fitness: F::default(),
            _gene: PhantomData::default(),
        }
    }

    pub fn genotype(&self) -> &Gnt {
        &self.genotype
    }

    pub fn fitness(&self) -> F {
        self.fitness
    }

    pub fn update_fitness(&mut self, fitness: F) {
        self.fitness = fitness
    }
}

impl<Gnt, T, F> Clone for Individual<Gnt, T, F>
where
    Gnt: Genotype<T>,
    T: Copy + Send + Sync + SampleUniformRange,
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
