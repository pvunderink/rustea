mod fitness;
mod genome;
mod individual;
mod selection;
mod simplega;
mod statistics;
mod variation;

use std::{cmp::Ordering, time::Instant};

use crate::{
    fitness::FitnessFunc, individual::Individual, selection::TruncationSelection,
    simplega::SimpleGA, variation::UniformCrossover,
};

fn main() {
    type Gene = bool;
    type Genome = Vec<Gene>;
    const GENOME_SIZE: usize = 8192;
    const POPULATION_SIZE: usize = 800;

    fn evaluate(idv: &Individual<Genome, Gene, usize>) -> usize {
        idv.genotype().iter().filter(|bit| **bit).count() // count the number of ones in the bitstring
    }

    fn compare(
        idv_a: &Individual<Genome, Gene, usize>,
        idv_b: &Individual<Genome, Gene, usize>,
    ) -> Ordering {
        idv_b.fitness().cmp(&idv_a.fitness()) // this means higher fitness is better
    }

    // Fitness function & variation operator & crossover operator
    let mut one_max = FitnessFunc::new(&evaluate, &compare);
    let variation = UniformCrossover::default();
    let selection = TruncationSelection;

    let ga = SimpleGA::new(&mut one_max, selection, variation);
    let mut ga = ga.random_population(POPULATION_SIZE, GENOME_SIZE);
    ga.set_target_fitness(GENOME_SIZE); // defining a target fitness allows the GA to stop early

    // Run EA
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
