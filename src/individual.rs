use rand::Rng;

use crate::bitstring::BitString;

#[derive(Debug)]
pub struct Individual<G, F>
where
    G: BitString,
    F: Default + Copy,
{
    genotype: G,
    fitness: F,
}

impl<G, F> Individual<G, F>
where
    G: BitString,
    F: Default + Copy,
{
    pub fn uniform_random<R>(rng: &mut R, len: usize) -> Self
    where
        R: Rng + ?Sized,
    {
        let genotype = G::random(rng, len);
        Individual {
            genotype,
            fitness: F::default(),
        }
    }

    pub fn from_genotype(genotype: G) -> Self {
        Individual {
            genotype,
            fitness: F::default(),
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

impl<G, F> Clone for Individual<G, F>
where
    G: BitString,
    F: Default + Copy,
{
    fn clone(&self) -> Self {
        Self {
            genotype: self.genotype.clone(),
            fitness: self.fitness.clone(),
        }
    }
}
