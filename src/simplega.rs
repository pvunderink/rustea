use std::fmt::Debug;

use crate::{
    fitness::{FitnessFunc, OptimizationGoal},
    genome::{Genome, Genotype, SampleUniformRange},
    individual::Individual,
    selection::SelectionOperator,
    variation::VariationOperator,
};

#[derive(Debug)]
pub enum Status {
    TargetReached(usize),
    BudgetReached(usize),
}

pub struct SimpleGA<'a, Gnt, T, F, S, V>
where
    Gnt: Genotype<T>, // type of genotype
    T: Copy + Send + Sync + SampleUniformRange,
    F: Default + Copy + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator<Gnt, T>,
{
    // genome: Gnm,
    population: Vec<Individual<Gnt, T, F>>,
    fitness_func: FitnessFunc<'a, Gnt, T, F>,
    selection_operator: S,
    variation_operator: V,
    target_fitness: Option<F>,
}

impl<'a, Gnt, T, F, S, V> SimpleGA<'a, Gnt, T, F, S, V>
where
    Gnt: Genotype<T>, // type of genotype
    T: Copy + Send + Sync + SampleUniformRange,
    F: Default + Copy + PartialOrd + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator<Gnt, T>,
{
    pub fn best_individual(&self) -> Option<&Individual<Gnt, T, F>> {
        self.population
            .iter()
            .min_by(|idv_a, idv_b| self.fitness_func.cmp(&idv_a.fitness(), &idv_b.fitness()))
    }

    pub fn worst_individual(&self) -> Option<&Individual<Gnt, T, F>> {
        self.population
            .iter()
            .max_by(|idv_a, idv_b| self.fitness_func.cmp(&idv_a.fitness(), &idv_b.fitness()))
    }

    pub fn run(&mut self, evaluation_budget: usize) -> Status {
        while self.fitness_func.evaluations() < evaluation_budget {
            // Check if target fitness is reached
            match self.target_fitness {
                Some(target) => match self.best_individual() {
                    Some(idv) => {
                        // TODO: does not check for approximate equality; may not work for floating points
                        if self.fitness_func.cmp(&idv.fitness(), &target).is_le() {
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
                .create_offspring(&self.population, &self.fitness_func);

            // Perform selection
            self.selection_operator
                .select(&mut self.population, offspring, &self.fitness_func);
        }

        return Status::BudgetReached(self.fitness_func.evaluations());
    }
}

pub struct SimpleGABuilder<'a, Gnm, Gnt, T, F, S, V>
where
    Gnm: Genome<T>,   // type of genome
    Gnt: Genotype<T>, // type of genotype
    T: Copy + Send + Sync + SampleUniformRange,
    F: Default + Copy + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator<Gnt, T>,
{
    genome: Option<Gnm>,
    population: Option<Vec<Individual<Gnt, T, F>>>,
    evaluation_func: Option<&'a (dyn Fn(&Gnt) -> F + Send + Sync)>,
    goal: OptimizationGoal,
    selection_operator: Option<S>,
    variation_operator: Option<V>,
    target_fitness: Option<F>,
}

impl<'a, Gnm, Gnt, T, F, S, V> SimpleGABuilder<'a, Gnm, Gnt, T, F, S, V>
where
    Gnm: Genome<T>,   // type of genome
    Gnt: Genotype<T>, // type of genotype
    T: Copy + Send + Sync + SampleUniformRange,
    F: Default + Copy + PartialOrd + Debug + Send + Sync,
    S: SelectionOperator,
    V: VariationOperator<Gnt, T>,
{
    pub fn new() -> Self {
        Self {
            genome: None,
            population: None,
            evaluation_func: None,
            goal: OptimizationGoal::MINIMIZE,
            selection_operator: None,
            variation_operator: None,
            target_fitness: None,
        }
    }

    pub fn genome(mut self, genome: Gnm) -> Self {
        self.genome = Some(genome);
        self
    }

    pub fn random_population(mut self, size: usize) -> Self {
        let Some(genome) = self.genome.clone() else {
            panic!("Failed to initialize population: the genome must be defined before the population can be initialized");
        };

        let population = (0..size)
            .map(|_| Individual::sample_uniform(&genome))
            .collect();

        self.population = Some(population);

        self
    }

    pub fn goal(mut self, goal: OptimizationGoal) -> Self {
        self.goal = goal;
        self
    }

    pub fn evaluation_function(mut self, func: &'a (dyn Fn(&Gnt) -> F + Send + Sync)) -> Self {
        self.evaluation_func = Some(func);
        self
    }

    pub fn selection(mut self, operator: S) -> Self {
        self.selection_operator = Some(operator);
        self
    }

    pub fn variation(mut self, operator: V) -> Self {
        self.variation_operator = Some(operator);
        self
    }

    pub fn target(mut self, fitness: F) -> Self {
        self.target_fitness = Some(fitness);
        self
    }

    pub fn build(self) -> SimpleGA<'a, Gnt, T, F, S, V> {
        let Some(population) = self.population else {
            panic!("Failed to build: population not initialized");
        };

        let Some(evaluation_func) = self.evaluation_func else {
            panic!("Failed to build: evaluation function not specified");
        };

        let fitness_func = FitnessFunc::new(evaluation_func, self.goal);

        let Some(selection_operator) = self.selection_operator else {
            panic!("Failed to build: selection operator not specified");
        };

        let Some(variation_operator) = self.variation_operator else {
            panic!("Failed to build: variation operator not specified");
        };

        let target_fitness = self.target_fitness;

        SimpleGA {
            population,
            fitness_func,
            selection_operator,
            variation_operator,
            target_fitness,
        }
    }
}
