mod bitstring;
mod simple;

use std::{cmp::Ordering, time::Instant};

use crate::{
    bitstring::U8BitString,
    simple::{FitnessFunc, Individual, SimpleGA},
};

fn main() {
    fn evaluate(idv: &mut Individual<U8BitString, usize>) -> usize {
        idv.genotype().iter().filter(|bit| *bit).count()
    }

    fn compare(
        idv_a: &Individual<U8BitString, usize>,
        idv_b: &Individual<U8BitString, usize>,
    ) -> Ordering {
        idv_b.fitness().cmp(&idv_a.fitness())
    }

    let mut one_max = FitnessFunc::new(&evaluate, &compare);
    let mut ga: SimpleGA<'_, U8BitString, _> = SimpleGA::new(8192, 1024, &mut one_max);

    let now = Instant::now();

    ga.run(500000);

    let elapsed = now.elapsed();
    println!(
        "Best fitness: {}, Elapsed: {:.2?}",
        ga.best_individual().fitness(),
        elapsed
    );
}
