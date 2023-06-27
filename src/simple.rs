use std::{
    cmp::Ordering,
    sync::{Arc, Mutex},
};

use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;

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

    pub fn genotype(&self) -> &dyn BitString {
        &self.genotype
    }

    pub fn fitness(&self) -> F {
        self.fitness
    }
}

pub trait FitnessFunc<G, F>
where
    G: BitString,
    F: Default + Copy,
{
    fn evaluate(&self, individual: &mut Individual<G, F>) -> F;
    fn cmp(&self, individual_a: &Individual<G, F>, individual_b: &Individual<G, F>) -> Ordering;
    fn evaluations(&self) -> usize;
}

pub struct OneMaxFitnessFunc {
    counter: Arc<Mutex<usize>>,
}

impl OneMaxFitnessFunc {
    pub fn new() -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
        }
    }
}

impl<T: BitString> FitnessFunc<T, usize> for OneMaxFitnessFunc {
    fn evaluate(&self, individual: &mut Individual<T, usize>) -> usize {
        let fitness = individual.genotype.iter().filter(|bit| *bit).count();
        individual.fitness = fitness;

        let mut counter = self.counter.lock().unwrap();
        *counter += 1;

        fitness
    }

    fn evaluations(&self) -> usize {
        *self.counter.lock().unwrap()
    }

    fn cmp(
        &self,
        individual_a: &Individual<T, usize>,
        individual_b: &Individual<T, usize>,
    ) -> Ordering {
        individual_b.fitness.cmp(&individual_a.fitness)
    }
}

pub struct SimpleGA<'a, G, F>
where
    G: BitString,
    F: Default + Copy + Send + Sync,
{
    genotype_size: usize,
    population_size: usize,
    population: Vec<Individual<G, F>>,
    fitness_func: &'a (dyn FitnessFunc<G, F> + Send + Sync),
}

impl<'a, G, F> SimpleGA<'a, G, F>
where
    G: BitString,
    F: Default + Copy + Send + Sync,
{
    pub fn new(
        genotype_size: usize,
        population_size: usize,
        fitness_func: &'a (dyn FitnessFunc<G, F> + Send + Sync),
    ) -> Self {
        // Initialize population
        let population = (0..population_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut idv = Individual::uniform_random(&mut rng, genotype_size);
                fitness_func.evaluate(&mut idv);
                idv
            })
            .collect();

        Self {
            genotype_size,
            population_size,
            population,
            fitness_func,
        }
    }

    pub fn best_individual(&self) -> &Individual<G, F> {
        &self.population[0]
    }

    pub fn run(&mut self, evaluation_budget: usize) {
        let mut rng = rand::thread_rng();

        while self.fitness_func.evaluations() < evaluation_budget {
            // Shuffle the population
            self.population.shuffle(&mut rng);
            let mut population_pairs = Vec::<(_, _)>::new();

            // Organize the population into pairs for crossover
            for i in 0..self.population.len() / 2 {
                population_pairs.push((&self.population[2 * i], &self.population[2 * i + 1]));
            }

            // Perform crossover and evaluation in parallel
            let mut offspring: Vec<_> = population_pairs
                .par_iter()
                .flat_map(|(parent1, parent2)| {
                    let mut children = uniform_crossover(parent1, parent2, 0.5);
                    self.fitness_func.evaluate(&mut children[0]);
                    self.fitness_func.evaluate(&mut children[1]);

                    children
                })
                .collect();

            // Truncation selection
            self.population.append(&mut offspring);
            self.population
                .sort_by(|idv_a, idv_b| self.fitness_func.cmp(idv_a, idv_b));
            self.population.truncate(self.population_size);
        }
    }
}

pub fn uniform_crossover<G, F>(
    parent_a: &Individual<G, F>,
    parent_b: &Individual<G, F>,
    probability: f64,
) -> Vec<Individual<G, F>>
where
    G: BitString,
    F: Default + Copy,
{
    assert_eq!(
        parent_a.genotype.len(),
        parent_b.genotype.len(),
        "length of genotypes must be equal"
    );

    let mut rng = rand::thread_rng();

    // Generate an array of booleans
    // true indicates that the gene should be crossed over
    let choices: Vec<_> = (0..parent_a.genotype.len())
        .map(|_| rng.gen_bool(probability))
        .collect();

    // Create copies of parent a and b
    let mut offspring_a = parent_a.genotype.clone();
    let mut offspring_b = parent_b.genotype.clone();

    for (idx, b) in choices.iter().enumerate() {
        if *b {
            offspring_b.set(idx, parent_a.genotype.get(idx));
            offspring_a.set(idx, parent_b.genotype.get(idx));
        }
    }

    vec![
        Individual::from_genotype(offspring_a),
        Individual::from_genotype(offspring_b),
    ]
}

pub fn one_point_crossover<G, F>(
    parent_a: &Individual<G, F>,
    parent_b: &Individual<G, F>,
) -> Vec<Individual<G, F>>
where
    G: BitString,
    F: Default + Copy,
{
    assert_eq!(
        parent_a.genotype.len(),
        parent_b.genotype.len(),
        "length of genotypes must be equal"
    );

    let mut rng = rand::thread_rng();

    // Pick a crossover point (both endpoints are included)
    let crossover_point: usize = rng.gen_range(0..parent_a.genotype.len() + 1);

    // Create copies of parent a and b
    let mut offspring_a = parent_a.genotype.clone();
    let mut offspring_b = parent_b.genotype.clone();

    for idx in 0..parent_a.genotype.len() {
        if idx >= crossover_point {
            offspring_b.set(idx, parent_a.genotype.get(idx));
            offspring_a.set(idx, parent_b.genotype.get(idx));
        }
    }

    vec![
        Individual::from_genotype(offspring_a),
        Individual::from_genotype(offspring_b),
    ]
}

pub fn two_point_crossover<G, F>(
    parent_a: &Individual<G, F>,
    parent_b: &Individual<G, F>,
) -> Vec<Individual<G, F>>
where
    G: BitString,
    F: Default + Copy,
{
    let offspring = one_point_crossover(parent_a, parent_b);
    let offspring = one_point_crossover(&offspring[0], &offspring[1]);

    return offspring;
}
