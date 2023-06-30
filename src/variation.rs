use crate::{
    fitness::FitnessFunc,
    genome::{BitString, Cartesian, Discrete, Genotype, SampleUniformRange},
    individual::Individual,
};

use derivative::Derivative;
use ndarray::{Array, Ix1};
use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

pub trait VariationOperator<Gnt, T>: Clone
where
    Self: Sized,
    Gnt: Genotype<T>,
    T: Copy + Send + Sync + SampleUniformRange,
{
    fn create_offspring<F>(
        &self,
        population: &Vec<Individual<Gnt, T, F>>,
        fitness_func: &FitnessFunc<'_, Gnt, T, F>,
    ) -> Vec<Individual<Gnt, T, F>>
    where
        F: Default + Copy + PartialOrd + Debug + Send + Sync;

    fn mutates(&self) -> bool;
}

#[derive(Clone)]
pub struct NoVariation;

impl<Gnt, T> VariationOperator<Gnt, T> for NoVariation
where
    Gnt: Genotype<T>,
    T: Copy + Send + Sync + SampleUniformRange,
{
    fn create_offspring<F>(
        &self,
        population: &Vec<Individual<Gnt, T, F>>,
        fitness_func: &FitnessFunc<'_, Gnt, T, F>,
    ) -> Vec<Individual<Gnt, T, F>>
    where
        F: Default + Copy + PartialOrd + Debug + Send + Sync,
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
pub struct UniformCrossover<Gnt, T>
where
    Gnt: Genotype<T> + Discrete + Cartesian<T>,
    T: Copy + Send + Sync + SampleUniformRange,
{
    #[derivative(Default(value = "0.5"))]
    probability: f64,
    _genotype: PhantomData<Gnt>,
    _gene: PhantomData<T>,
}

impl<Gnt, T> UniformCrossover<Gnt, T>
where
    Gnt: Genotype<T> + Discrete + Cartesian<T>,
    T: Copy + Send + Sync + SampleUniformRange,
{
    pub fn with_probability(probability: f64) -> Self {
        Self {
            probability,
            _genotype: PhantomData::default(),
            _gene: PhantomData::default(),
        }
    }

    fn crossover<F>(
        &self,
        parent_a: &Individual<Gnt, T, F>,
        parent_b: &Individual<Gnt, T, F>,
    ) -> Vec<Individual<Gnt, T, F>>
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
pub struct OnePointCrossover<Gnt, T>
where
    Gnt: Genotype<T> + Discrete + Cartesian<T>,
    T: Copy + Send + Sync + SampleUniformRange,
{
    _genotype: PhantomData<Gnt>,
    _gene: PhantomData<T>,
}

impl<Gnt, T> OnePointCrossover<Gnt, T>
where
    Gnt: Genotype<T> + Discrete + Cartesian<T>,
    T: Copy + Send + Sync + SampleUniformRange,
{
    fn crossover<F>(
        &self,
        parent_a: &Individual<Gnt, T, F>,
        parent_b: &Individual<Gnt, T, F>,
    ) -> Vec<Individual<Gnt, T, F>>
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
pub struct TwoPointCrossover<Gnt, T>
where
    Gnt: Genotype<T> + Discrete + Cartesian<T>,
    T: Copy + Send + Sync + SampleUniformRange,
{
    _genotype: PhantomData<Gnt>,
    _gene: PhantomData<T>,
}

impl<Gnt, T> TwoPointCrossover<Gnt, T>
where
    Gnt: Genotype<T> + Discrete + Cartesian<T>,
    T: Copy + Send + Sync + SampleUniformRange,
{
    fn crossover<F>(
        &self,
        parent_a: &Individual<Gnt, T, F>,
        parent_b: &Individual<Gnt, T, F>,
    ) -> Vec<Individual<Gnt, T, F>>
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
            impl<Gnt, T> VariationOperator<Gnt, T> for $t
            where
                Gnt: Genotype<T> + Discrete + Cartesian<T>,
                T: Copy + Send + Sync + SampleUniformRange,
            {
                fn create_offspring<F>(
                    &self,
                    population: &Vec<Individual<Gnt, T, F>>,
                    fitness_func: &FitnessFunc<'_, Gnt, T, F>,
                ) -> Vec<Individual<Gnt, T, F>>
                where
                    Self: Sized,
                    F: Default + Copy + PartialOrd +  Debug + Send + Sync,
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
        UniformCrossover<Gnt, T>,
        OnePointCrossover<Gnt, T>,
        TwoPointCrossover<Gnt, T>
);

#[derive(Debug)]
struct UnivariateModel {
    probabilities: Array<f64, Ix1>,
}

impl UnivariateModel {
    fn estimate_from_population<Gnt, F>(population: &Vec<Individual<Gnt, bool, F>>) -> Self
    where
        Gnt: BitString,
        F: Default + Copy + Debug + Send + Sync,
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

    fn sample<Gnt, F, R>(&self, rng: &mut R) -> Individual<Gnt, bool, F>
    where
        Gnt: BitString,
        F: Default + Copy + Debug + Send + Sync,
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
pub struct UMDA<Gnt>
where
    Gnt: BitString,
{
    _genotype: PhantomData<Gnt>,
}

impl<Gnt> VariationOperator<Gnt, bool> for UMDA<Gnt>
where
    Gnt: BitString,
{
    fn create_offspring<F>(
        &self,
        population: &Vec<Individual<Gnt, bool, F>>,
        fitness_func: &FitnessFunc<'_, Gnt, bool, F>,
    ) -> Vec<Individual<Gnt, bool, F>>
    where
        Self: Sized,
        F: Default + Copy + PartialOrd + Debug + Send + Sync,
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
