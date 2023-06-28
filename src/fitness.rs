use std::{
    cmp::Ordering,
    fmt::Debug,
    sync::{Arc, Mutex},
};

use approx::AbsDiffEq;

use crate::{bitstring::BitString, individual::Individual};

pub struct FitnessFunc<'a, G, F>
where
    G: BitString,
    F: Default + Copy + AbsDiffEq + Debug,
{
    counter: Arc<Mutex<usize>>,
    evaluation_func: &'a (dyn Fn(&mut Individual<G, F>) -> F + Send + Sync),
    comparison_func: &'a (dyn Fn(&Individual<G, F>, &Individual<G, F>) -> Ordering + Send + Sync),
}

impl<'a, G, F> FitnessFunc<'a, G, F>
where
    G: BitString,
    F: Default + Copy + AbsDiffEq + Debug,
{
    pub fn new(
        evaluation_func: &'a (dyn Fn(&mut Individual<G, F>) -> F + Send + Sync),
        comparison_func: &'a (dyn Fn(&Individual<G, F>, &Individual<G, F>) -> Ordering
                 + Send
                 + Sync),
    ) -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            evaluation_func,
            comparison_func,
        }
    }

    pub fn evaluate(&self, individual: &mut Individual<G, F>) -> F {
        let fitness = (self.evaluation_func)(individual);
        individual.update_fitness(fitness);

        let mut counter = self.counter.lock().unwrap();
        *counter += 1;

        fitness
    }

    pub fn evaluations(&self) -> usize {
        *self.counter.lock().unwrap()
    }

    pub fn cmp(&self, idv_a: &Individual<G, F>, idv_b: &Individual<G, F>) -> Ordering {
        (self.comparison_func)(idv_a, idv_b)
    }
}
