mod bitstring;
mod simple;

use std::time::Instant;

use simple::OneMaxFitnessFunc;

use crate::{bitstring::U8BitString, simple::SimpleGA};

// fn count_ones(arr: Vec<i32>) -> usize {
//     arr.iter().filter(|x| **x == 1).count()
// }

// fn test() {
//     let parent_a = Individual::from_genotype(array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
//     let parent_b = Individual::from_genotype(array![1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

//     let mut ones_a = 0;
//     let mut ones_b = 0;

//     let n = 10000;

//     for _ in 0..n {
//         let offspring = uniform_crossover(&parent_a, &parent_b, 0.5);

//         ones_a += count_ones(offspring[0].genotype().to_vec());
//         ones_b += count_ones(offspring[1].genotype().to_vec());
//     }

//     println!("Average 1's offspring a: {}", (ones_a as f64) / (n as f64));
//     println!("Average 1's offspring b: {}", (ones_b as f64) / (n as f64));
// }

fn main() {
    let mut one_max = OneMaxFitnessFunc::new();
    let mut ga: SimpleGA<'_, U8BitString, _> = SimpleGA::new(8192, 1024, &mut one_max);

    let now = Instant::now();

    ga.run(500000);

    let elapsed = now.elapsed();
    println!(
        "Best fitness: {}, Elapsed: {:.2?}",
        ga.best_individual().fitness(),
        elapsed
    );
    println!("Best individual: {:?}", ga.best_individual())
}
