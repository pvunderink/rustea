use rand::{seq::SliceRandom, Rng};
use std::fmt::Debug;

use crate::{
    bitstring::BitString,
    fitness::{ApproxEq, FitnessFunc},
    individual::Individual,
};

pub trait SelectionOperator {
    fn select<G, F>(
        &mut self,
        population: &mut Vec<Individual<G, F>>,
        offspring: Vec<Individual<G, F>>,
        fitness_func: &FitnessFunc<'_, G, F>,
    ) where
        Self: Sized,
        G: BitString,
        F: Default + Copy + ApproxEq + Debug;
}

pub struct TruncationSelection;

impl SelectionOperator for TruncationSelection {
    fn select<G, F>(
        &mut self,
        population: &mut Vec<Individual<G, F>>,
        offspring: Vec<Individual<G, F>>,
        fitness_func: &FitnessFunc<'_, G, F>,
    ) where
        G: BitString,
        F: Default + Copy + ApproxEq + Debug,
    {
        let population_size = population.len();
        population.extend(offspring.into_iter());
        population.sort_by(|idv_a, idv_b| fitness_func.cmp(idv_a, idv_b));
        population.truncate(population_size);
    }
}

pub struct TournamentSelection<'a, R>
where
    R: Rng + ?Sized,
{
    tournament_size: usize,
    include_parents: bool,
    rng: &'a mut R,
}

impl<'a, R> TournamentSelection<'a, R>
where
    R: Rng + ?Sized,
{
    pub fn new(tournament_size: usize, include_parents: bool, rng: &'a mut R) -> Self {
        Self {
            tournament_size,
            include_parents,
            rng,
        }
    }
}

impl<'a, R> SelectionOperator for TournamentSelection<'a, R>
where
    R: Rng + Sized,
{
    fn select<G, F>(
        &mut self,
        population: &mut Vec<Individual<G, F>>,
        offspring: Vec<Individual<G, F>>,
        fitness_func: &FitnessFunc<'_, G, F>,
    ) where
        G: BitString,
        F: Default + Copy + ApproxEq + Debug,
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

        for _ in 0..num_iterations {
            pool.shuffle(self.rng);

            let mut winners: Vec<_> = (0..num_tournaments)
                .map(|i| {
                    let winner = pool
                        [self.tournament_size * i..self.tournament_size * i + self.tournament_size]
                        .iter()
                        .min_by(|idv_a, idv_b| fitness_func.cmp(idv_a, idv_b))
                        .unwrap();

                    winner.clone()
                })
                .collect();

            population.append(&mut winners);
        }

        assert!(population.len() == population_size)
    }
}
