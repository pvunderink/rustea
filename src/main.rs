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

    fn evaluate(idv: &Individual<Genome, usize>) -> usize {
        idv.genotype().iter().filter(|bit| *bit).count()
    }

    fn compare(idv_a: &Individual<Genome, usize>, idv_b: &Individual<Genome, usize>) -> Ordering {
        idv_b.fitness().cmp(&idv_a.fitness()) // this means higher fitness is better
    }

    let mut one_max = FitnessFunc::new(&evaluate, &compare);

    let variation = UniformCrossover::default();
    let selection = TruncationSelection;

    const SIZE: usize = 8192;

    let mut ga = SimpleGA::new(SIZE, 800, &mut one_max, selection, variation);

    ga.set_target_fitness(SIZE); // defining a target fitness allows the GA to stop early

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
