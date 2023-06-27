use std::{
    cmp::Ordering,
    sync::{Arc, Mutex},
};

use ndarray::{Array, Ix1};
use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;

#[derive(Debug)]
pub struct Individual<T>
where
    T: Clone,
{
    genotype: Array<T, Ix1>,
    fitness: f64,
}

impl<T> Individual<T>
where
    T: Clone,
{
    pub fn from_genotype(genotype: Array<T, Ix1>) -> Self {
        Individual {
            genotype,
            fitness: 0.0,
        }
    }

    pub fn genotype(&self) -> &Array<T, Ix1> {
        &self.genotype
    }

    pub fn fitness(&self) -> f64 {
        self.fitness
    }
}

impl Individual<i32> {
    pub fn new_random_binary(genotype_length: usize) -> Self {
        let mut rng = rand::thread_rng();

        let bits = (0..genotype_length)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { 0 })
            .collect();
        let genotype = Array::from_vec(bits);

        Self {
            genotype,
            fitness: 0.0,
        }
    }
}

pub trait FitnessFunc<T: Clone> {
    fn evaluate(&self, individual: &mut Individual<T>) -> f64;
    fn cmp(&self, individual_a: &Individual<T>, individual_b: &Individual<T>) -> Ordering;
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

impl FitnessFunc<i32> for OneMaxFitnessFunc {
    fn evaluate(&self, individual: &mut Individual<i32>) -> f64 {
        let fitness = individual.genotype.iter().filter(|bit| **bit == 1).count() as f64;
        individual.fitness = fitness;

        let mut counter = self.counter.lock().unwrap();
        *counter += 1;

        fitness
    }

    fn evaluations(&self) -> usize {
        *self.counter.lock().unwrap()
    }

    fn cmp(&self, individual_a: &Individual<i32>, individual_b: &Individual<i32>) -> Ordering {
        individual_b.fitness.total_cmp(&individual_a.fitness)
    }
}

pub struct SimpleGA<'a> {
    genotype_size: usize,
    population_size: usize,
    population: Vec<Individual<i32>>,
    fitness_func: &'a (dyn FitnessFunc<i32> + Send + Sync),
}

impl<'a> SimpleGA<'a> {
    pub fn new(
        genotype_size: usize,
        population_size: usize,
        fitness_func: &'a (dyn FitnessFunc<i32> + Send + Sync),
    ) -> Self {
        // Initialize population
        let population = (0..population_size)
            .into_par_iter()
            .map(|_| {
                let mut idv = Individual::new_random_binary(genotype_size);
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

    pub fn best_individual(&self) -> &Individual<i32> {
        &self.population[0]
    }

    pub fn run(&mut self, evaluation_budget: usize) {
        let mut rng = rand::thread_rng();

        while self.fitness_func.evaluations() < evaluation_budget {
            // Shuffle the population
            self.population.shuffle(&mut rng);
            let mut population_pairs = Vec::<(&Individual<i32>, &Individual<i32>)>::new();

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

pub fn uniform_crossover<T>(
    parent_a: &Individual<T>,
    parent_b: &Individual<T>,
    probability: f64,
) -> Vec<Individual<T>>
where
    T: Clone,
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
    let mut offspring_a = parent_a.genotype.to_owned();
    let mut offspring_b = parent_b.genotype.to_owned();

    for (idx, b) in choices.iter().enumerate() {
        if *b {
            offspring_b[idx] = parent_a.genotype[idx].clone();
            offspring_a[idx] = parent_b.genotype[idx].clone();
        }
    }

    vec![
        Individual::from_genotype(offspring_a),
        Individual::from_genotype(offspring_b),
    ]
}

pub fn one_point_crossover<T>(
    parent_a: &Individual<T>,
    parent_b: &Individual<T>,
) -> Vec<Individual<T>>
where
    T: Clone,
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
    let mut offspring_a = parent_a.genotype.to_owned();
    let mut offspring_b = parent_b.genotype.to_owned();

    for idx in 0..parent_a.genotype.len() {
        if idx >= crossover_point {
            offspring_b[idx] = parent_a.genotype[idx].clone();
            offspring_a[idx] = parent_b.genotype[idx].clone();
        }
    }

    vec![
        Individual::from_genotype(offspring_a),
        Individual::from_genotype(offspring_b),
    ]
}

pub fn two_point_crossover<T>(
    parent_a: &Individual<T>,
    parent_b: &Individual<T>,
) -> Vec<Individual<T>>
where
    T: Clone,
{
    let offspring = one_point_crossover(parent_a, parent_b);
    let offspring = one_point_crossover(&offspring[0], &offspring[1]);

    return offspring;
}
