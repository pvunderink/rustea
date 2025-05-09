use std::fmt::Debug;

use crate::{
    fitness::{Fitness, FitnessEvaluator, FitnessFunc, OptimizationGoal},
    gene::{Allele, Gene},
    genome::Genome,
    genotype::{FixedSizeGenotype, Genotype},
    individual::Individual,
    selection::SelectionOperator,
    variation::VariationOperator,
};

#[derive(Debug)]
pub enum Status {
    TargetReached(usize),
    BudgetReached(usize),
}

pub struct SimpleGA<'a, Gnt, A, F, S, V>
where
    A: Allele,
    F: Fitness,
    S: SelectionOperator,
    V: VariationOperator<Gnt, A, F>,
    Gnt: Genotype<A>,
{
    // genome: Gnm,
    population: Vec<Individual<Gnt, A, F>>,
    fitness_func: FitnessEvaluator<'a, Gnt, A, F>,
    selection_operator: S,
    variation_operator: V,
    target_fitness: Option<F>,
    verbose: bool,
}

impl<'a, Gnt, A, F, S, V> SimpleGA<'a, Gnt, A, F, S, V>
where
    A: Allele,
    F: Fitness,
    S: SelectionOperator,
    V: VariationOperator<Gnt, A, F>,
    Gnt: Genotype<A>,
{
    pub fn best_individual(&self) -> Option<&Individual<Gnt, A, F>> {
        self.population
            .iter()
            .min_by(|idv_a, idv_b| self.fitness_func.cmp(&idv_a.fitness(), &idv_b.fitness()))
    }

    pub fn worst_individual(&self) -> Option<&Individual<Gnt, A, F>> {
        self.population
            .iter()
            .max_by(|idv_a, idv_b| self.fitness_func.cmp(&idv_a.fitness(), &idv_b.fitness()))
    }

    pub fn run(&mut self, evaluation_budget: usize) -> Status {
        // Perform initial evaluation
        self.population.iter_mut().for_each(|idv| {
            self.fitness_func.evaluate(idv);
        });

        let mut iteration = 0;

        // Main loop
        while self.fitness_func.evaluations() < evaluation_budget {
            if self.verbose {
                if iteration % 100 == 0 {
                    println!(
                        "Iteration: {:?}, Budget: {:?}/{:?}, Best: {:?}",
                        iteration,
                        self.fitness_func.evaluations(),
                        evaluation_budget,
                        self.best_individual().unwrap().fitness()
                    )
                }
            }
            // Check if target fitness is reached
            if let Some(target) = self.target_fitness {
                if let Some(idv) = self.best_individual() {
                    // TODO: does not check for approximate equality; may not work for floating points
                    if self.fitness_func.cmp(&idv.fitness(), &target).is_le() {
                        return Status::TargetReached(self.fitness_func.evaluations());
                    }
                }
            }

            // Perform variation
            let offspring = self
                .variation_operator
                .create_offspring(&self.population, &self.fitness_func);

            // Perform selection
            self.selection_operator
                .select(&mut self.population, offspring, &self.fitness_func);

            iteration += 1;
        }

        Status::BudgetReached(self.fitness_func.evaluations())
    }
}

#[derive(Clone)]
pub struct SimpleGABuilder<'a, Gnt, A, G, F, S, V>
where
    A: Allele,
    G: Gene<A>,
    F: Fitness,
    S: SelectionOperator,
    V: VariationOperator<Gnt, A, F>,
    Gnt: Genotype<A>,
{
    genome: Option<&'a Genome<Gnt, A, G>>,
    population: Option<Vec<Individual<Gnt, A, F>>>,
    evaluation_func: Option<&'a (dyn FitnessFunc<Gnt, A, F>)>,
    goal: OptimizationGoal,
    selection_operator: Option<S>,
    variation_operator: Option<V>,
    target_fitness: Option<F>,
    verbose: bool,
}

impl<'a, Gnt, A, G, F, S, V> SimpleGABuilder<'a, Gnt, A, G, F, S, V>
where
    A: Allele,
    G: Gene<A>,
    F: Fitness,
    S: SelectionOperator,
    V: VariationOperator<Gnt, A, F>,
    Gnt: Genotype<A>,
{
    pub fn new() -> Self {
        Self {
            genome: None,
            population: None,
            evaluation_func: None,
            goal: OptimizationGoal::Minimize,
            selection_operator: None,
            variation_operator: None,
            target_fitness: None,
            verbose: false,
        }
    }

    pub fn genome(mut self, genome: &'a Genome<Gnt, A, G>) -> Self {
        self.genome = Some(genome);
        self
    }

    pub fn with_population(mut self, population: Vec<Gnt>) -> Self {
        let population = population
            .into_iter()
            .map(|genotype| Individual::from_genotype(genotype))
            .collect();

        self.population = Some(population);

        self
    }

    pub fn goal(mut self, goal: OptimizationGoal) -> Self {
        self.goal = goal;
        self
    }

    pub fn evaluation_function(mut self, func: &'a (dyn FitnessFunc<Gnt, A, F>)) -> Self {
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

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn build(self) -> SimpleGA<'a, Gnt, A, F, S, V> {
        let Some(population) = self.population else {
            panic!("Failed to build: population not initialized");
        };

        let Some(evaluation_func) = self.evaluation_func else {
            panic!("Failed to build: evaluation function not specified");
        };

        let fitness_func = FitnessEvaluator::new(evaluation_func, self.goal);

        let Some(selection_operator) = self.selection_operator else {
            panic!("Failed to build: selection operator not specified");
        };

        let Some(variation_operator) = self.variation_operator else {
            panic!("Failed to build: variation operator not specified");
        };

        let target_fitness = self.target_fitness;

        let verbose = self.verbose;

        SimpleGA {
            population,
            fitness_func,
            selection_operator,
            variation_operator,
            target_fitness,
            verbose,
        }
    }
}

impl<'a, Gnt, A, G, F, S, V> SimpleGABuilder<'a, Gnt, A, G, F, S, V>
where
    A: Allele,
    G: Gene<A>,
    F: Fitness,
    S: SelectionOperator,
    V: VariationOperator<Gnt, A, F>,
    Gnt: FixedSizeGenotype<A>,
{
    pub fn random_population(mut self, size: usize) -> Self {
        let Some(genome) = self.genome else {
            panic!(
                "Failed to initialize population: the genome must be defined before the population can be initialized"
            );
        };

        let mut rng = rand::rng();

        let population = (0..size)
            .map(|_| Individual::sample_uniform(&mut rng, genome))
            .collect();

        self.population = Some(population);

        self
    }
}
