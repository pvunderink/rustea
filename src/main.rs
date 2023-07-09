#![feature(step_trait)]

mod fitness;
mod gene;
mod genome;
mod individual;
mod model;
mod rng;
mod selection;
mod simplega;
mod statistics;
mod variation;

use std::time::Instant;

use fitness::Fitness;
use gene::Gene;
use selection::SelectionOperator;
use variation::VariationOperator;

use crate::{
    fitness::OptimizationGoal, gene::BoolDomain, genome::Genome, selection::TruncationSelection,
    simplega::SimpleGABuilder, variation::Umda,
};

type AlleleType = bool;
type Genotype = Vec<AlleleType>;

const GENOME_SIZE: usize = 4096;
const POPULATION_SIZE: usize = 450;
const EVAL_BUDGET: usize = 73000;
const TARGET: usize = GENOME_SIZE;
const GOAL: OptimizationGoal = OptimizationGoal::Maximize;
const RUNS: usize = 50;

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
    fitness
}

fn measure_success_rate<G, F, S, V>(
    builder: SimpleGABuilder<Genotype, AlleleType, G, F, S, V>,
    n: usize,
) -> f64
where
    G: Gene<AlleleType>,
    F: Fitness,
    S: SelectionOperator,
    V: VariationOperator<Genotype, AlleleType>,
{
    let mut success_count = 0usize;
    for i in 0..n {
        let mut ga = builder.clone().build();
        let status = ga.run(EVAL_BUDGET);

        match status {
            simplega::Status::TargetReached(_) => success_count += 1,
            simplega::Status::BudgetReached(_) => (),
        }

        println!("[{}/{}] Finished with status: {:?}", i + 1, n, status)
    }

    return (success_count as f64) / (n as f64);
}

fn main() {
    let genome = Genome::discrete_genome_with_domain(&bdom!(), GENOME_SIZE);

    let builder = SimpleGABuilder::new()
        .genome(&genome)
        .random_population(POPULATION_SIZE)
        .evaluation_function(&one_max)
        .goal(GOAL)
        .selection(TruncationSelection)
        .variation(Umda::with_genome(&genome))
        .target(TARGET);

    // Run EA
    let now = Instant::now();
    let success_rate = measure_success_rate(builder, RUNS);
    let elapsed = now.elapsed();

    println!(
        "Elapsed: {:.2?}, Success Rate: {:.2?}",
        elapsed, success_rate
    );
    // println!(
    //     "Best fitness: {}, Worst fitness: {}, Elapsed: {:.2?}, Status: {:?}",
    //     ga.best_individual().unwrap().fitness(),
    //     ga.worst_individual().unwrap().fitness(),
    //     elapsed,
    //     status
    // );
}
