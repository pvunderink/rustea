use std::{
    cmp::Ordering,
    fmt::Debug,
    sync::{Arc, Mutex},
};

use approx::AbsDiffEq;

use crate::{genome::Genome, individual::Individual};

pub struct FitnessFunc<'a, G, Gene, F>
where
    G: Genome<Gene>,
    F: Default + Copy + AbsDiffEq + Debug,
{
    counter: Arc<Mutex<usize>>,
    evaluation_func: &'a (dyn Fn(&Individual<G, Gene, F>) -> F + Send + Sync),
    comparison_func:
        &'a (dyn Fn(&Individual<G, Gene, F>, &Individual<G, Gene, F>) -> Ordering + Send + Sync),
}

impl<'a, G, Gene, F> FitnessFunc<'a, G, Gene, F>
where
    G: Genome<Gene>,
    F: Default + Copy + AbsDiffEq + Debug,
{
    pub fn new(
        evaluation_func: &'a (dyn Fn(&Individual<G, Gene, F>) -> F + Send + Sync),
        comparison_func: &'a (dyn Fn(&Individual<G, Gene, F>, &Individual<G, Gene, F>) -> Ordering
                 + Send
                 + Sync),
    ) -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            evaluation_func,
            comparison_func,
        }
    }

    pub fn evaluate(&self, individual: &mut Individual<G, Gene, F>) -> F {
        let fitness = (self.evaluation_func)(individual);
        individual.update_fitness(fitness);

        let mut counter = self.counter.lock().unwrap();
        *counter += 1;

        fitness
    }

    pub fn evaluations(&self) -> usize {
        *self.counter.lock().unwrap()
    }

    pub fn cmp(&self, idv_a: &Individual<G, Gene, F>, idv_b: &Individual<G, Gene, F>) -> Ordering {
        (self.comparison_func)(idv_a, idv_b)
    }
}
