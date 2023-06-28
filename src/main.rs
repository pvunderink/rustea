mod bitstring;
mod fitness;
mod individual;
mod selection;
mod simple;
mod statistics;
mod variation;

use std::{cmp::Ordering, time::Instant};

use crate::{
    bitstring::{BitString, U8BitString},
    fitness::FitnessFunc,
    individual::Individual,
    selection::TruncationSelection,
    simple::SimpleGA,
    variation::UniformCrossover,
};

fn main() {
    // Define evaluation function
    fn evaluate(idv: &mut Individual<U8BitString, usize>) -> usize {
        idv.genotype().iter().filter(|bit| *bit).count()
    }

    // Define how to compare fitness between individuals
    fn compare(
        idv_a: &Individual<U8BitString, usize>,
        idv_b: &Individual<U8BitString, usize>,
    ) -> Ordering {
        // Higher fitness is better
        idv_b.fitness().cmp(&idv_a.fitness())
    }

    // Setup fitness function
    let mut one_max = FitnessFunc::new(&evaluate, &compare);

    // Setup variation operator
    let variation = UniformCrossover::default();

    // Setup selection operator
    // let selection = TournamentSelection::new(8, true, &mut rng);
    let selection = TruncationSelection;

    // Setup genetic algorithm
    let size = 8192;
    let mut ga = SimpleGA::new(size, 800, &mut one_max, selection, variation);

    // Defining a target fitness allows the GA to stop early
    ga.set_target_fitness(size);

    let now = Instant::now();

    let status = ga.run(250000);

    let elapsed = now.elapsed();
    println!(
        "Best fitness: {}, Worst fitness: {}, Elapsed: {:.2?} ({:?})",
        ga.best_individual().unwrap().fitness(),
        ga.worst_individual().unwrap().fitness(),
        elapsed,
        status
    );
}
