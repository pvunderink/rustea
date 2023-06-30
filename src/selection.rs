use rand::seq::SliceRandom;
use std::fmt::Debug;

use crate::{
    fitness::FitnessFunc,
    genome::{Genotype, SampleUniformRange},
    individual::Individual,
};

pub trait SelectionOperator: Clone {
    fn select<Gnt, T, F>(
        &mut self,
        population: &mut Vec<Individual<Gnt, T, F>>,
        offspring: Vec<Individual<Gnt, T, F>>,
        fitness_func: &FitnessFunc<'_, Gnt, T, F>,
    ) where
        Self: Sized,
        Gnt: Genotype<T>,
        T: Copy + Send + Sync + SampleUniformRange,
        F: Default + Copy + PartialOrd + Debug;
}

#[derive(Clone)]
pub struct NoSelection;

impl SelectionOperator for NoSelection {
    fn select<Gnt, T, F>(
        &mut self,
        _: &mut Vec<Individual<Gnt, T, F>>,
        _: Vec<Individual<Gnt, T, F>>,
        _: &FitnessFunc<'_, Gnt, T, F>,
    ) where
        Gnt: Genotype<T>,
        T: Copy + Send + Sync + SampleUniformRange,
        F: Default + Copy + PartialOrd + Debug,
    {
    }
}

#[derive(Clone)]
pub struct TruncationSelection;

impl SelectionOperator for TruncationSelection {
    fn select<Gnt, T, F>(
        &mut self,
        population: &mut Vec<Individual<Gnt, T, F>>,
        offspring: Vec<Individual<Gnt, T, F>>,
        fitness_func: &FitnessFunc<'_, Gnt, T, F>,
    ) where
        Gnt: Genotype<T>,
        T: Copy + Send + Sync + SampleUniformRange,

        F: Default + Copy + PartialOrd + Debug,
    {
        let population_size = population.len();
        population.extend(offspring.into_iter());
        population.sort_by(|idv_a, idv_b| fitness_func.cmp(&idv_a.fitness(), &idv_b.fitness()));
        population.truncate(population_size);
    }
}

#[derive(Clone)]
pub struct TournamentSelection {
    tournament_size: usize,
    include_parents: bool,
}

impl SelectionOperator for TournamentSelection {
    fn select<Gnt, T, F>(
        &mut self,
        population: &mut Vec<Individual<Gnt, T, F>>,
        offspring: Vec<Individual<Gnt, T, F>>,
        fitness_func: &FitnessFunc<'_, Gnt, T, F>,
    ) where
        Gnt: Genotype<T>,
        T: Copy + Send + Sync + SampleUniformRange,
        F: Default + Copy + PartialOrd + Debug,
    {
        let population_size = population.len();
        let pool_size = offspring.len()
            + if self.include_parents {
                population_size
            } else {
                0
            };

        let mut pool: Vec<_> = Vec::with_capacity(pool_size);

        if self.include_parents {
            pool.append(population);
        }
        pool.extend(offspring);

        // N - pool size
        // p - pop size
        // o - offspring size
        // k - tournament size
        // t - number of times each individual is considered
        // t = k*p / N
        assert!(pool_size % self.tournament_size == 0);
        assert!(self.tournament_size * population_size % pool_size == 0);

        let num_iterations = self.tournament_size * population_size / pool_size;
        let num_tournaments = pool_size / self.tournament_size;

        population.clear();

        let mut rng = rand::thread_rng();

        for _ in 0..num_iterations {
            pool.shuffle(&mut rng);

            let mut winners: Vec<_> = (0..num_tournaments)
                .map(|i| {
                    let winner = pool
                        [self.tournament_size * i..self.tournament_size * i + self.tournament_size]
                        .iter()
                        .min_by(|idv_a, idv_b| fitness_func.cmp(&idv_a.fitness(), &idv_b.fitness()))
                        .unwrap();

                    winner.clone()
                })
                .collect();

            population.append(&mut winners);
        }

        assert!(population.len() == population_size)
    }
}
