use crate::{
    fitness::FitnessFunc,
    genome::{BitString, Cartesian, Discrete, Genome},
    individual::Individual,
};
use approx::AbsDiffEq;
use derivative::Derivative;
use ndarray::{Array, Ix1};
use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

pub trait VariationOperator<G, Gene>: Clone
where
    Self: Sized,
    G: Genome<Gene>,
    Gene: Clone,
{
    fn create_offspring<F>(
        &self,
        population: &Vec<Individual<G, Gene, F>>,
        fitness_func: &FitnessFunc<'_, G, Gene, F>,
    ) -> Vec<Individual<G, Gene, F>>
    where
        F: Default + Copy + AbsDiffEq + Debug + Send + Sync;

    fn mutates(&self) -> bool;
}

#[derive(Clone)]
pub struct NoVariation;

impl<G, Gene> VariationOperator<G, Gene> for NoVariation
where
    G: Genome<Gene>,
    Gene: Clone + Send + Sync,
{
    fn create_offspring<F>(
        &self,
        population: &Vec<Individual<G, Gene, F>>,
        fitness_func: &FitnessFunc<'_, G, Gene, F>,
    ) -> Vec<Individual<G, Gene, F>>
    where
        F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
    {
        let offspring = population
            .par_iter()
            .map(|idv| {
                let mut child = idv.clone();

                fitness_func.evaluate(&mut child);

                child
            })
            .collect();

        offspring
    }

    fn mutates(&self) -> bool {
        false
    }
}

#[derive(Derivative, Clone)]
#[derivative(Default)]
pub struct UniformCrossover<G, Gene>
where
    G: Genome<Gene> + Discrete + Cartesian<Gene>,
    Gene: Clone,
{
    #[derivative(Default(value = "0.5"))]
    probability: f64,
    _genome: PhantomData<G>,
    _gene: PhantomData<Gene>,
}

impl<G, Gene> UniformCrossover<G, Gene>
where
    G: Genome<Gene> + Discrete + Cartesian<Gene>,
    Gene: Clone,
{
    pub fn with_probability(probability: f64) -> Self {
        Self {
            probability,
            _gene: PhantomData::default(),
            _genome: PhantomData::default(),
        }
    }

    fn crossover<F>(
        &self,
        parent_a: &Individual<G, Gene, F>,
        parent_b: &Individual<G, Gene, F>,
    ) -> Vec<Individual<G, Gene, F>>
    where
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
            .map(|_| rng.gen_bool(self.probability))
            .collect();

        // Create copies of parent a and b
        let mut offspring_a = parent_a.genotype().clone();
        let mut offspring_b = parent_b.genotype().clone();

        for (idx, b) in choices.iter().enumerate() {
            if b {
                offspring_b.set(idx, parent_a.genotype().get(idx));
                offspring_a.set(idx, parent_b.genotype().get(idx));
            }
        }

        vec![
            Individual::from_genotype(offspring_a),
            Individual::from_genotype(offspring_b),
        ]
    }
}

#[derive(Default, Clone)]
pub struct OnePointCrossover<G, Gene>
where
    G: Genome<Gene> + Discrete + Cartesian<Gene>,
    Gene: Clone,
{
    _genome: PhantomData<G>,
    _gene: PhantomData<Gene>,
}

impl<G, Gene> OnePointCrossover<G, Gene>
where
    G: Genome<Gene> + Discrete + Cartesian<Gene>,
    Gene: Clone,
{
    fn crossover<F>(
        &self,
        parent_a: &Individual<G, Gene, F>,
        parent_b: &Individual<G, Gene, F>,
    ) -> Vec<Individual<G, Gene, F>>
    where
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
}

#[derive(Default, Clone)]
pub struct TwoPointCrossover<G, Gene>
where
    G: Genome<Gene> + Discrete + Cartesian<Gene>,
    Gene: Clone,
{
    _genome: PhantomData<G>,
    _gene: PhantomData<Gene>,
}

impl<G, Gene> TwoPointCrossover<G, Gene>
where
    G: Genome<Gene> + Discrete + Cartesian<Gene>,
    Gene: Clone,
{
    fn crossover<F>(
        &self,
        parent_a: &Individual<G, Gene, F>,
        parent_b: &Individual<G, Gene, F>,
    ) -> Vec<Individual<G, Gene, F>>
    where
        F: Default + Copy,
    {
        assert_eq!(
            parent_a.genotype().len(),
            parent_b.genotype().len(),
            "length of genotypes must be equal"
        );

        let mut rng = rand::thread_rng();

        // Pick a crossover point (both endpoints are included)
        let crossover_point_1: usize = rng.gen_range(0..parent_a.genotype().len() + 1);
        let crossover_point_2: usize = rng.gen_range(0..parent_a.genotype().len() + 1);

        // Create copies of parent a and b
        let mut offspring_a = parent_a.genotype().clone();
        let mut offspring_b = parent_b.genotype().clone();

        for idx in 0..parent_a.genotype().len() {
            if idx >= crossover_point_1.min(crossover_point_2)
                && idx <= crossover_point_1.max(crossover_point_2)
            {
                offspring_b.set(idx, parent_a.genotype().get(idx));
                offspring_a.set(idx, parent_b.genotype().get(idx));
            }
        }

        vec![
            Individual::from_genotype(offspring_a),
            Individual::from_genotype(offspring_b),
        ]
    }
}

macro_rules! impl_two_parent_crossover {
    (for $($t:ty),+) => {
        $(
            impl<G, Gene> VariationOperator<G, Gene> for $t
            where
                G: Genome<Gene> + Discrete + Cartesian<Gene>,
                Gene: Clone + Send + Sync
            {
                fn create_offspring<F>(
                    &self,
                    population: &Vec<Individual<G, Gene, F>>,
                    fitness_func: &FitnessFunc<'_, G, Gene, F>,
                ) -> Vec<Individual<G, Gene, F>>
                where
                    Self: Sized,
                    F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
                {
                    let mut rng = rand::thread_rng();
                    // Shuffle the population
                    let mut population: Vec<_> = population.iter().collect();
                    population.shuffle(&mut rng);

                    let mut population_pairs = Vec::<(_, _)>::new();

                    // Organize the population into pairs for crossover
                    for i in 0..population.len() / 2 {
                        population_pairs.push((population[2 * i], population[2 * i + 1]));
                    }

                    // Perform crossover and evaluation in parallel
                    let offspring: Vec<_> = population_pairs
                        .par_iter()
                        .flat_map(|(parent1, parent2)| {
                            let mut children = self.crossover(parent1, parent2);

                            fitness_func.evaluate(&mut children[0]);
                            fitness_func.evaluate(&mut children[1]);

                            return children
                        })
                        .collect();

                    offspring
                }

                fn mutates(&self) -> bool {
                    false
                }
        })*
    }
  }

impl_two_parent_crossover!(
    for
        UniformCrossover<G, Gene>,
        OnePointCrossover<G, Gene>,
        TwoPointCrossover<G, Gene>
);

#[derive(Debug)]
struct UnivariateModel {
    probabilities: Array<f64, Ix1>,
}

impl UnivariateModel {
    fn estimate_from_population<G, F>(population: &Vec<Individual<G, bool, F>>) -> Self
    where
        G: BitString,
        F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
    {
        assert!(population.len() > 0);

        let len = population.first().unwrap().genotype().len();

        let counts = population
            .into_par_iter()
            .map(|idv| {
                let gen: Array<_, _> = idv.genotype().iter().collect();
                gen.map(|b| if *b { 1.0 } else { 0.0 })
            })
            .reduce(|| Array::zeros(len), |g1, g2| g1 + g2);

        Self {
            probabilities: counts / population.len() as f64,
        }
    }

    fn sample<G, F, R>(&self, rng: &mut R) -> Individual<G, bool, F>
    where
        G: BitString,
        F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
        R: Rng,
    {
        let genotype = self
            .probabilities
            .iter()
            .map(|p| rng.gen_bool(*p))
            .collect();

        Individual::from_genotype(genotype)
    }
}

#[derive(Default, Clone)]
pub struct UMDA<G>
where
    G: BitString,
{
    _genome: PhantomData<G>,
    // _gene: PhantomData<Gene>,
}

impl<G> VariationOperator<G, bool> for UMDA<G>
where
    G: BitString,
{
    fn create_offspring<F>(
        &self,
        population: &Vec<Individual<G, bool, F>>,
        fitness_func: &FitnessFunc<'_, G, bool, F>,
    ) -> Vec<Individual<G, bool, F>>
    where
        Self: Sized,
        F: Default + Copy + AbsDiffEq + Debug + Send + Sync,
    {
        let model = UnivariateModel::estimate_from_population(population);

        (0..population.len())
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();

                let mut child = model.sample(&mut rng);

                fitness_func.evaluate(&mut child);

                child
            })
            .collect()
    }

    fn mutates(&self) -> bool {
        false
    }
}
