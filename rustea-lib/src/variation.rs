use crate::{
    fitness::{Fitness, FitnessFunc},
    gene::{Allele, Discrete, DiscreteDomain, DiscreteGene},
    genome::{Cartesian, Genome},
    genotype::Genotype,
    individual::Individual,
    model::UnivariateModel,
};

use derivative::Derivative;
use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;
use std::marker::PhantomData;

pub trait VariationOperator<Gnt, A, F>: Clone
where
    Self: Sized,
    A: Allele,
    F: Fitness,
    Gnt: Genotype<A>,
{
    fn create_offspring(
        &self,
        population: &[Individual<Gnt, A, F>],
        fitness_func: &FitnessFunc<'_, Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>>;

    fn mutates(&self) -> bool;
}

#[derive(Clone)]
pub struct NoVariation;

impl<Gnt, A, F> VariationOperator<Gnt, A, F> for NoVariation
where
    A: Allele,
    F: Fitness,
    Gnt: Genotype<A>,
{
    fn create_offspring(
        &self,
        population: &[Individual<Gnt, A, F>],
        fitness_func: &FitnessFunc<'_, Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>> {
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
    A: Allele + Discrete,
    Gnt: Genotype<A> + Cartesian<A>,
{
    #[derivative(Default(value = "0.5"))]
    probability: f64,
    _allele: PhantomData<A>,
    _genotype: PhantomData<Gnt>,
}

impl<Gnt, A> UniformCrossover<Gnt, A>
where
    A: Allele + Discrete,
    Gnt: Genotype<A> + Cartesian<A>,
{
    pub fn with_probability(probability: f64) -> Self {
        Self {
            probability,
            _allele: PhantomData,
            _genotype: PhantomData,
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
}

#[derive(Default, Clone)]
pub struct OnePointCrossover<Gnt, A>
where
    A: Allele + Discrete,
    Gnt: Genotype<A>,
{
    _allele: PhantomData<A>,
    _genotype: PhantomData<Gnt>,
}

impl<Gnt, A> OnePointCrossover<Gnt, A>
where
    A: Allele + Discrete,
    Gnt: Genotype<A> + Cartesian<A>,
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
    A: Allele + Discrete,
    Gnt: Genotype<A>,
{
    _gene: PhantomData<A>,
    _genotype: PhantomData<Gnt>,
}

impl<Gnt, A> TwoPointCrossover<Gnt, A>
where
    A: Allele + Discrete,
    Gnt: Genotype<A> + Cartesian<A>,
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
            impl<Gnt, A, F> VariationOperator<Gnt, A, F> for $t
            where
                A: Allele + Discrete,
                F: Fitness,
                Gnt: Genotype<A> + Cartesian<A>,
            {
                fn create_offspring(
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

#[derive(Debug, Clone)]
pub struct Umda<'a, Gnt, A, D>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    Gnt: Genotype<A>,
{
    genome: &'a Genome<Gnt, A, DiscreteGene<A, D>>,
}

impl<'a, Gnt, A, D> Umda<'a, Gnt, A, D>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    Gnt: Genotype<A>,
{
    pub fn with_genome(genome: &'a Genome<Gnt, A, DiscreteGene<A, D>>) -> Self {
        Self { genome }
    }
}

impl<'a, Gnt, A, D, F> VariationOperator<Gnt, A, F> for Umda<'a, Gnt, A, D>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    F: Fitness,
    Gnt: Genotype<A>,
{
    fn create_offspring(
        &self,
        population: &[Individual<Gnt, A, F>],
        fitness_func: &FitnessFunc<'_, Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>>
    where
        Self: Sized,
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
