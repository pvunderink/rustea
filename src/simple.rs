use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;
use std::fmt::Debug;

use crate::{
    bitstring::BitString,
    fitness::{ApproxEq, FitnessFunc},
    individual::Individual,
    selection::SelectionOperator,
};

#[derive(Debug)]
pub enum Status {
    TargetReached,
    BudgetReached,
    Failed,
}

pub struct SimpleGA<'a, G, F, S>
where
    G: BitString,                                       // genome type
    F: Default + Copy + ApproxEq + Debug + Send + Sync, // fitness value type
    S: SelectionOperator,                               // selection operator type
{
    population: Vec<Individual<G, F>>,
    fitness_func: &'a FitnessFunc<'a, G, F>,
    selection_operator: S,
    target_fitness: Option<F>,
}

impl<'a, G, F, S> SimpleGA<'a, G, F, S>
where
    G: BitString,
    F: Default + Copy + ApproxEq + Debug + Send + Sync,
    S: SelectionOperator,
{
    pub fn new(
        genotype_size: usize,
        population_size: usize,
        fitness_func: &'a FitnessFunc<G, F>,
        selection_operator: S,
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
            population,
            fitness_func,
            selection_operator,
            target_fitness: Option::None,
        }
    }

    pub fn best_individual(&self) -> Option<&Individual<G, F>> {
        self.population
            .iter()
            .min_by(|idv_a, idv_b| self.fitness_func.cmp(idv_a, idv_b))
    }

    pub fn worst_individual(&self) -> Option<&Individual<G, F>> {
        self.population
            .iter()
            .max_by(|idv_a, idv_b| self.fitness_func.cmp(idv_a, idv_b))
    }

    pub fn set_target_fitness(&mut self, target: F) {
        self.target_fitness = Some(target)
    }

    pub fn run(&mut self, evaluation_budget: usize) -> Status {
        let mut rng = rand::thread_rng();

        while self.fitness_func.evaluations() < evaluation_budget {
            match self.target_fitness {
                Some(target) => match self.best_individual() {
                    Some(idv) => {
                        if idv.fitness().approx_eq(&target) {
                            return Status::TargetReached;
                        }
                    }
                    None => (),
                },
                None => (),
            }

            // Shuffle the population
            self.population.shuffle(&mut rng);
            let mut population_pairs = Vec::<(_, _)>::new();

            // Organize the population into pairs for crossover
            for i in 0..self.population.len() / 2 {
                population_pairs.push((&self.population[2 * i], &self.population[2 * i + 1]));
            }

            // Perform crossover and evaluation in parallel
            let offspring: Vec<_> = population_pairs
                .par_iter()
                .flat_map(|(parent1, parent2)| {
                    let mut children = uniform_crossover(parent1, parent2, 0.5);
                    self.fitness_func.evaluate(&mut children[0]);
                    self.fitness_func.evaluate(&mut children[1]);

                    children
                })
                .collect();

            // Truncation selection
            self.selection_operator
                .select(&mut self.population, offspring, self.fitness_func);
        }

        return Status::BudgetReached;
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
        parent_a.genotype().len(),
        parent_b.genotype().len(),
        "length of genotypes must be equal"
    );

    let mut rng = rand::thread_rng();

    // Generate an array of booleans
    // true indicates that the gene should be crossed over
    let choices: Vec<_> = (0..parent_a.genotype().len())
        .map(|_| rng.gen_bool(probability))
        .collect();

    // Create copies of parent a and b
    let mut offspring_a = parent_a.genotype().clone();
    let mut offspring_b = parent_b.genotype().clone();

    for (idx, b) in choices.iter().enumerate() {
        if *b {
            offspring_b.set(idx, parent_a.genotype().get(idx));
            offspring_a.set(idx, parent_b.genotype().get(idx));
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
        parent_a.genotype().len(),
        parent_b.genotype().len(),
        "length of genotypes must be equal"
    );

    let mut rng = rand::thread_rng();

    // Pick a crossover point (both endpoints are included)
    let crossover_point: usize = rng.gen_range(0..parent_a.genotype().len() + 1);

    // Create copies of parent a and b
    let mut offspring_a = parent_a.genotype().clone();
    let mut offspring_b = parent_b.genotype().clone();

    for idx in 0..parent_a.genotype().len() {
        if idx >= crossover_point {
            offspring_b.set(idx, parent_a.genotype().get(idx));
            offspring_a.set(idx, parent_b.genotype().get(idx));
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
