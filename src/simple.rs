use approx::{abs_diff_eq, AbsDiffEq};
use rayon::prelude::*;
use std::fmt::Debug;

use crate::{
    bitstring::BitString, fitness::FitnessFunc, individual::Individual,
    selection::SelectionOperator, variation::VariationOperator,
};

#[derive(Debug)]
pub enum Status {
    TargetReached,
    BudgetReached,
    Failed,
}

pub struct SimpleGA<'a, G, F, S, V>
where
    G: BitString,                                        // genome type
    F: Default + Copy + AbsDiffEq + Debug + Send + Sync, // fitness value type
    S: SelectionOperator,                                // selection operator type
{
    population: Vec<Individual<G, F>>,
    fitness_func: &'a FitnessFunc<'a, G, F>,
    selection_operator: S,
    variation_operator: V,
    target_fitness: Option<F>,
}

impl<'a, G, F, S, V> SimpleGA<'a, G, F, S, V>
where
    G: BitString,
    F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator,
{
    pub fn new(
        genotype_size: usize,
        population_size: usize,
        fitness_func: &'a FitnessFunc<G, F>,
        selection_operator: S,
        variation_operator: V,
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
            variation_operator,
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
        while self.fitness_func.evaluations() < evaluation_budget {
            match self.target_fitness {
                Some(target) => match self.best_individual() {
                    Some(idv) => {
                        if abs_diff_eq!(idv.fitness(), &target) {
                            return Status::TargetReached;
                        }
                    }
                    None => (),
                },
                None => (),
            }

            // Perform variation
            let offspring = self
                .variation_operator
                .create_offspring(&self.population, self.fitness_func);

            // Truncation selection
            self.selection_operator
                .select(&mut self.population, offspring, self.fitness_func);
        }

        return Status::BudgetReached;
    }
}
