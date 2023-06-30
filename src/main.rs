mod fitness;
mod genome;
mod individual;
mod selection;
mod simplega;
mod statistics;
mod variation;

use std::time::Instant;

use crate::{
    fitness::OptimizationGoal,
    genome::{Gene, GeneRange, Genome as _},
    selection::TruncationSelection,
    simplega::SimpleGABuilder,
    variation::UniformCrossover,
};

type GeneType = bool;
type Genotype = Vec<GeneType>;
type Genome = Vec<Gene<GeneType>>;

const GENOME_SIZE: usize = 8192;
const POPULATION_SIZE: usize = 800;
const TARGET: usize = GENOME_SIZE;

fn one_max(genotype: &Genotype) -> usize {
    genotype.iter().filter(|bit| **bit).count() // count the number of ones in the bitstring
}

fn deceptive_trap(genotype: &Genotype) -> usize {
    let fitness = one_max(genotype);

    if fitness == genotype.len() {
        return 0;
    } else if fitness == 0 {
        return genotype.len();
    }
    return fitness;
}

fn main() {
    let builder = SimpleGABuilder::new();

    let mut ga = builder
        .genome(Genome::uniform_with_range(
            GENOME_SIZE,
            range!(false..=true),
        ))
        .random_population(POPULATION_SIZE)
        .evaluation_function(&one_max)
        .goal(OptimizationGoal::MAXIMIZE)
        .selection(TruncationSelection)
        .variation(UniformCrossover::default())
        .target(TARGET) // defining a target fitness allows the GA to stop early
        .build();

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
