#![feature(step_trait)]

mod fitness;
mod gene;
mod genome;
mod individual;
mod selection;
mod simplega;
mod statistics;
mod variation;

use std::time::Instant;

use crate::{
    fitness::OptimizationGoal, gene::BoolDomain, genome::Genome, selection::TruncationSelection,
    simplega::SimpleGABuilder, variation::UniformCrossover,
};

type GeneType = bool;
type Genotype = Vec<GeneType>;

const GENOME_SIZE: usize = 8192;
const POPULATION_SIZE: usize = 800;
const EVAL_BUDGET: usize = 250000;
const TARGET: usize = GENOME_SIZE;
const GOAL: OptimizationGoal = OptimizationGoal::MAXIMIZE;

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
        .genome(Genome::discrete_genome_with_domain(&bdom!(), GENOME_SIZE))
        .random_population(POPULATION_SIZE)
        .evaluation_function(&one_max)
        .goal(GOAL)
        .selection(TruncationSelection)
        .variation(UniformCrossover::default())
        .target(TARGET) // defining a target fitness allows the GA to stop early
        .build();

    // Run EA
    let now = Instant::now();
    let status = ga.run(EVAL_BUDGET);
    let elapsed = now.elapsed();

    println!(
        "Best fitness: {}, Worst fitness: {}, Elapsed: {:.2?}, Status: {:?}",
        ga.best_individual().unwrap().fitness(),
        ga.worst_individual().unwrap().fitness(),
        elapsed,
        status
    );
}
