use crate::{
    bitstring::BitString,
    fitness::{ApproxEq, FitnessFunc},
    individual::Individual,
};

pub trait Selection {
    fn select<G, F>(
        &self,
        population: &mut Vec<Individual<G, F>>,
        offspring: &mut Vec<Individual<G, F>>,
        fitness_func: &FitnessFunc<'_, G, F>,
    ) where
        Self: Sized,
        G: BitString,
        F: Default + Copy + ApproxEq;
}

pub struct TruncationSelection;

impl Selection for TruncationSelection {
    fn select<G, F>(
        &self,
        population: &mut Vec<Individual<G, F>>,
        offspring: &mut Vec<Individual<G, F>>,
        fitness_func: &FitnessFunc<'_, G, F>,
    ) where
        G: BitString,
        F: Default + Copy + ApproxEq,
    {
        let population_size = population.len();
        population.append(offspring);
        population.sort_by(|idv_a, idv_b| fitness_func.cmp(idv_a, idv_b));
        population.truncate(population_size);
    }
}

pub struct TournamentSelection;
