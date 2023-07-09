use crate::{
    fitness::{Fitness, FitnessFunc},
    gene::{Allele, Discrete, DiscreteDomain, DiscreteGene},
    genome::{Cartesian, Genome, Genotype},
    individual::Individual,
    model::UnivariateModel,
};

use derivative::Derivative;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_distr::WeightedIndex;
use rayon::prelude::*;
use std::marker::PhantomData;

pub trait VariationOperator<Gnt, A>: Clone
where
    Self: Sized,
    Gnt: Genotype<A>,
    A: Allele,
{
    fn create_offspring<F>(
        &self,
        population: &[Individual<Gnt, A, F>],
        fitness_func: &FitnessFunc<'_, Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>>
    where
        F: Fitness;

    fn mutates(&self) -> bool;
}

#[derive(Clone)]
pub struct NoVariation;

impl<Gnt, A> VariationOperator<Gnt, A> for NoVariation
where
    Gnt: Genotype<A>,
    A: Allele,
{
    fn create_offspring<F>(
        &self,
        population: &[Individual<Gnt, A, F>],
        fitness_func: &FitnessFunc<'_, Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>>
    where
        F: Fitness,
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
pub struct UniformCrossover<Gnt, A>
where
    Gnt: Genotype<A> + Cartesian<A>,
    A: Allele + Discrete,
{
    #[derivative(Default(value = "0.5"))]
    probability: f64,
    _genotype: PhantomData<Gnt>,
    _allele: PhantomData<A>,
}

impl<Gnt, A> UniformCrossover<Gnt, A>
where
    Gnt: Genotype<A> + Cartesian<A>,
    A: Allele + Discrete,
{
    pub fn with_probability(probability: f64) -> Self {
        Self {
            probability,
            _genotype: PhantomData,
            _allele: PhantomData,
        }
    }

    fn crossover<F>(
        &self,
        parent_a: &Individual<Gnt, A, F>,
        parent_b: &Individual<Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>>
    where
        F: Fitness,
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
pub struct OnePointCrossover<Gnt, A>
where
    Gnt: Genotype<A> + Cartesian<A>,
    A: Allele + Discrete,
{
    _genotype: PhantomData<Gnt>,
    _allele: PhantomData<A>,
}

impl<Gnt, A> OnePointCrossover<Gnt, A>
where
    Gnt: Genotype<A> + Cartesian<A>,
    A: Allele + Discrete,
{
    fn crossover<F>(
        &self,
        parent_a: &Individual<Gnt, A, F>,
        parent_b: &Individual<Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>>
    where
        F: Fitness,
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
pub struct TwoPointCrossover<Gnt, A>
where
    Gnt: Genotype<A> + Cartesian<A>,
    A: Allele + Discrete,
{
    _genotype: PhantomData<Gnt>,
    _gene: PhantomData<A>,
}

impl<Gnt, A> TwoPointCrossover<Gnt, A>
where
    Gnt: Genotype<A> + Cartesian<A>,
    A: Allele + Discrete,
{
    fn crossover<F>(
        &self,
        parent_a: &Individual<Gnt, A, F>,
        parent_b: &Individual<Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>>
    where
        F: Fitness,
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
            impl<Gnt, A> VariationOperator<Gnt, A> for $t
            where
                Gnt: Genotype<A> + Cartesian<A>,
                A: Allele + Discrete,
            {
                fn create_offspring<F>(
                    &self,
                    population: &[Individual<Gnt, A, F>],
                    fitness_func: &FitnessFunc<'_, Gnt, A, F>,
                ) -> Vec<Individual<Gnt, A, F>>
                where
                    Self: Sized,
                    F: Fitness,
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
        UniformCrossover<Gnt, A>,
        OnePointCrossover<Gnt, A>,
        TwoPointCrossover<Gnt, A>
);

#[derive(Clone)]
pub struct Umda<'a, Gnt, A, D>
where
    Gnt: Genotype<A>,
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
{
    genome: &'a Genome<A, DiscreteGene<A, D>>,
    _genotype: PhantomData<Gnt>,
}

impl<'a, Gnt, A, D> Umda<'a, Gnt, A, D>
where
    Gnt: Genotype<A>,
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
{
    pub fn with_genome(genome: &'a Genome<A, DiscreteGene<A, D>>) -> Self {
        Self {
            genome,
            _genotype: PhantomData,
        }
    }
}

impl<'a, Gnt, A, D> VariationOperator<Gnt, A> for Umda<'a, Gnt, A, D>
where
    Gnt: Genotype<A>,
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
{
    fn create_offspring<F>(
        &self,
        population: &[Individual<Gnt, A, F>],
        fitness_func: &FitnessFunc<'_, Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>>
    where
        Self: Sized,
        F: Fitness,
    {
        let model = UnivariateModel::estimate_from_population(self.genome, population);

        (0..population.len())
            .into_par_iter()
            .map_init(
                || rand::thread_rng(), // each thread has its own rng
                |rng, _| {
                    let mut child = model.sample(rng);

                    fitness_func.evaluate(&mut child);

                    child
                },
            )
            .collect()
    }

    fn mutates(&self) -> bool {
        false
    }
}
