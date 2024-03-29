pub extern crate rustea_lib;

use std::time::Instant;

use rustea_lib::bdom;
use rustea_lib::fitness::Fitness;
use rustea_lib::gene::Gene;
use rustea_lib::genotype::{Genotype, SizedVec};
use rustea_lib::selection::SelectionOperator;
use rustea_lib::variation::VariationOperator;

use rustea_lib::{
    ecga::Ecga,
    fitness::OptimizationGoal,
    gene::BoolDomain,
    genome::Genome,
    selection::TruncationSelection,
    simplega::SimpleGABuilder,
    simplega::Status,
    variation::{Umda, UniformCrossover},
};

const K: usize = 5;
const M: usize = 12;
const GENOME_SIZE: usize = K * M;
const POPULATION_SIZE: usize = 10000;
const EVAL_BUDGET: usize = 80000;

// const GENOME_SIZE: usize = 32;
// const POPULATION_SIZE: usize = 800;
// const EVAL_BUDGET: usize = 250000;

const TARGET: usize = GENOME_SIZE;
const GOAL: OptimizationGoal = OptimizationGoal::Maximize;
const RUNS: usize = 10;

type AlleleType = bool;
// type Gnt = ArrayVec<AlleleType, GENOME_SIZE>;
// type Gnt = [AlleleType; GENOME_SIZE];
type Gnt = SizedVec<AlleleType, GENOME_SIZE>;

fn one_max(genotype: &Gnt) -> usize {
    genotype.iter().filter(|bit| *bit).count() // count the number of ones in the bitstring
}

fn one_max_raw(chunk: &[bool]) -> usize {
    chunk.iter().filter(|bit| **bit).count() // count the number of ones in the bitstring
}

fn deceptive_trap(genotype: &Gnt) -> usize {
    genotype
        .array_chunks::<K>()
        .map(|chunk| {
            let fitness = one_max_raw(chunk);
            if fitness == K {
                return 0;
            } else if fitness == 0 {
                return K;
            }
            fitness
        })
        .sum()
}

fn measure_success_rate<G, F, S, V>(
    builder: SimpleGABuilder<Gnt, AlleleType, G, F, S, V>,
    n: usize,
) -> f64
where
    G: Gene<AlleleType>,
    F: Fitness,
    S: SelectionOperator,
    V: VariationOperator<Gnt, AlleleType, F>,
{
    let mut success_count = 0usize;
    for i in 0..n {
        let mut ga = builder.clone().build();
        let status = ga.run(EVAL_BUDGET);

        match status {
            Status::TargetReached(_) => success_count += 1,
            Status::BudgetReached(_) => (),
        }

        println!(
            "[{}/{}] Finished with status: {:?}. Best fitness: {:?}, Worst fitness: {:?}",
            i + 1,
            n,
            status,
            ga.best_individual().unwrap().fitness(),
            ga.worst_individual().unwrap().fitness()
        )
    }

    return (success_count as f64) / (n as f64);
}

fn main() {
    let genome = Genome::with_discrete_domain(&bdom!());

    let builder = SimpleGABuilder::new()
        .genome(&genome)
        .random_population(POPULATION_SIZE)
        .evaluation_function(&deceptive_trap)
        .goal(GOAL)
        .selection(TruncationSelection)
        // .variation(UniformCrossover::default())
        // .variation(Umda::with_genome(&genome))
        .variation(Ecga::with_genome(&genome, 0.02))
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
