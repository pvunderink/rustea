use std::marker::PhantomData;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{
    fitness::{Fitness, FitnessFunc},
    gene::{Allele, Discrete, DiscreteDomain, DiscreteGene},
    genome::Genome,
    genotype::FixedSizeGenotype,
    individual::Individual,
    model::{Factorization, MultivariateModel},
    variation::VariationOperator,
};

#[derive(Debug, Clone)]
pub struct Ecga<'a, Gnt, A, D, F>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    F: Fitness,
    Gnt: FixedSizeGenotype<A>,
{
    genome: &'a Genome<Gnt, A, DiscreteGene<A, D>>,
    p_best: f64,
    _genotype: PhantomData<Gnt>,
    _fitness: PhantomData<F>,
}

impl<'a, Gnt, A, D, F> Ecga<'a, Gnt, A, D, F>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    F: Fitness,
    Gnt: FixedSizeGenotype<A>,
{
    pub fn with_genome(genome: &'a Genome<Gnt, A, DiscreteGene<A, D>>, p_best: f64) -> Self {
        Self {
            genome,
            p_best,
            _genotype: PhantomData,
            _fitness: PhantomData,
        }
    }

    fn select_model(
        &self,
        initial_factorization: Factorization,
        population: &[&Individual<Gnt, A, F>],
    ) -> MultivariateModel<'_, Gnt, A, D, F>
    where
        Self: Sized,
    {
        let mut model = MultivariateModel::estimate_from_population(
            self.genome,
            population,
            initial_factorization,
        );

        loop {
            let candidates = model.factorization().par_join_all();

            let Some(best_model) = candidates
                .map(|fact| {
                    MultivariateModel::estimate_from_population(self.genome, population, fact)
                })
                .min_by(|model1, model2| {
                    model1
                        .combined_complexity()
                        .total_cmp(&model2.combined_complexity())
                })
            else {
                break;
            };

            if best_model.combined_complexity() <= model.combined_complexity() {
                model = best_model;
            } else {
                break;
            }
        }

        model
    }

    fn select_individuals<'b>(
        &self,
        population: &'b [Individual<Gnt, A, F>],
        fitness_func: &FitnessFunc<'_, Gnt, A, F>,
    ) -> Vec<&'b Individual<Gnt, A, F>> {
        let mut selection: Vec<_> = population.iter().collect();
        selection.sort_unstable_by(|a, b| fitness_func.cmp(&a.fitness(), &b.fitness()));
        selection
            .into_iter()
            .take((self.p_best * population.len() as f64) as usize)
            .collect()
    }
}

impl<'a, Gnt, A, D, F> VariationOperator<Gnt, A, F> for Ecga<'a, Gnt, A, D, F>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    F: Fitness,
    Gnt: FixedSizeGenotype<A>,
{
    fn create_offspring(
        &self,
        population: &[Individual<Gnt, A, F>],
        fitness_func: &FitnessFunc<'_, Gnt, A, F>,
    ) -> Vec<Individual<Gnt, A, F>>
    where
        Self: Sized,
    {
        let selection = self.select_individuals(population, fitness_func);
        let model = self.select_model(Factorization::univariate(self.genome.len()), &selection);

        println!("Factorization: {:?}", model.factorization());

        (0..population.len())
            .into_par_iter()
            .map_init(
                || rand::rng(), // each thread has its own rng
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
