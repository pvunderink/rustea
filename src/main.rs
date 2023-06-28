mod bitstring;
mod fitness;
mod individual;
mod selection;
mod simplega;
mod statistics;
mod variation;

use std::{cmp::Ordering, time::Instant};

use crate::{
    bitstring::BitString, fitness::FitnessFunc, individual::Individual,
    selection::TruncationSelection, simplega::SimpleGA, variation::UniformCrossover,
};

fn main() {
    type Genome = Vec<bool>;

    // Define evaluation function
    fn evaluate(idv: &Individual<Genome, usize>) -> usize {
        idv.genotype().iter().filter(|bit| *bit).count()
    }

    // Define how to compare fitness between individuals
    fn compare(idv_a: &Individual<Genome, usize>, idv_b: &Individual<Genome, usize>) -> Ordering {
        // Higher fitness is better
        idv_b.fitness().cmp(&idv_a.fitness())
    }

    // Setup fitness function
    let mut one_max = FitnessFunc::new(&evaluate, &compare);

    // Setup variation operator
    let variation = UniformCrossover::default();

    // Setup selection operator
    let selection = TruncationSelection;

    // Setup genetic algorithm
    const SIZE: usize = 8192;

    let mut ga = SimpleGA::new(SIZE, 800, &mut one_max, selection, variation);

    // Defining a target fitness allows the GA to stop early
    ga.set_target_fitness(SIZE);

    let now = Instant::now();

    let status = ga.run(250000);

    let elapsed = now.elapsed();
    println!(
        "Best fitness: {}, Worst fitness: {}, Elapsed: {:.2?}, Status: {:?}",
        ga.best_individual().unwrap().fitness(),
        ga.worst_individual().unwrap().fitness(),
        elapsed,
        status
    );
}
