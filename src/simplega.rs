use approx::{abs_diff_eq, AbsDiffEq};
use rayon::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

use crate::{
    fitness::FitnessFunc,
    genome::{Genome, RandomInit},
    individual::Individual,
    selection::SelectionOperator,
    variation::VariationOperator,
};

#[derive(Debug)]
pub enum Status {
    TargetReached(usize),
    BudgetReached(usize),
}

pub struct Uninitialized;
pub struct Initialized;

pub struct SimpleGA<'a, G, Gene, F, S, V, State = Uninitialized>
where
    G: Genome<Gene>, // genome type
    Gene: Clone + Send + Sync,
    F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator<G, Gene>,
{
    population: Vec<Individual<G, Gene, F>>,
    fitness_func: &'a FitnessFunc<'a, G, Gene, F>,
    selection_operator: S,
    variation_operator: V,
    target_fitness: Option<F>,
    _state: PhantomData<State>,
}

impl<'a, G, Gene, F, S, V, State> SimpleGA<'a, G, Gene, F, S, V, State>
where
    G: Genome<Gene>, // genome type
    Gene: Clone + Send + Sync,
    F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator<G, Gene>,
{
    pub fn set_target_fitness(&mut self, target: F) {
        self.target_fitness = Some(target)
    }
}

impl<'a, G, Gene, F, S, V> SimpleGA<'a, G, Gene, F, S, V, Uninitialized>
where
    G: Genome<Gene>, // genome type
    Gene: Clone + Send + Sync,
    F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator<G, Gene>,
{
    pub fn new(fitness_func: &'a FitnessFunc<G, Gene, F>, selection: S, variation: V) -> Self {
        Self {
            population: Vec::new(),
            fitness_func,
            selection_operator: selection,
            variation_operator: variation,
            target_fitness: Option::None,
            _state: PhantomData::default(),
        }
    }
}

impl<'a, G, Gene, F, S, V> SimpleGA<'a, G, Gene, F, S, V, Initialized>
where
    G: Genome<Gene>, // genome type
    Gene: Clone + Send + Sync,
    F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator<G, Gene>,
{
    pub fn best_individual(&self) -> Option<&Individual<G, Gene, F>> {
        self.population
            .iter()
            .min_by(|idv_a, idv_b| self.fitness_func.cmp(idv_a, idv_b))
    }

    pub fn worst_individual(&self) -> Option<&Individual<G, Gene, F>> {
        self.population
            .iter()
            .max_by(|idv_a, idv_b| self.fitness_func.cmp(idv_a, idv_b))
    }

    pub fn run(&mut self, evaluation_budget: usize) -> Status {
        while self.fitness_func.evaluations() < evaluation_budget {
            // Check if target fitness is reached
            match self.target_fitness {
                Some(target) => match self.best_individual() {
                    Some(idv) => {
                        if abs_diff_eq!(idv.fitness(), &target) {
                            return Status::TargetReached(self.fitness_func.evaluations());
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

            // Perform selection
            self.selection_operator
                .select(&mut self.population, offspring, self.fitness_func);
        }

        return Status::BudgetReached(self.fitness_func.evaluations());
    }
}

impl<'a, G, Gene, F, S, V> SimpleGA<'a, G, Gene, F, S, V, Uninitialized>
where
    G: Genome<Gene> + RandomInit, // genome type
    Gene: Clone + Send + Sync,
    F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator<G, Gene>,
{
    pub fn random_population(
        &self,
        population_size: usize,
        genome_size: usize,
    ) -> SimpleGA<'a, G, Gene, F, S, V, Initialized> {
        // Initialize population
        let population = (0..population_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut idv = Individual::random(&mut rng, genome_size);
                self.fitness_func.evaluate(&mut idv);
                idv
            })
            .collect();

        SimpleGA {
            population: population,
            fitness_func: self.fitness_func,
            selection_operator: self.selection_operator.clone(),
            variation_operator: self.variation_operator.clone(),
            target_fitness: self.target_fitness,
            _state: PhantomData::default(),
        }
    }
}
