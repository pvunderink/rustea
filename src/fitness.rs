use std::{
    cmp::Ordering,
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::{bitstring::BitString, individual::Individual};

pub trait ApproxEq {
    fn approx_eq(&self, target: &Self) -> bool;
}

macro_rules! impl_int_ApproxEq {
  (for $($t:ty),+) => {
      $(impl ApproxEq for $t {
        fn approx_eq(&self, target: &Self) -> bool {
          *self == *target
        }
      })*
  }
}

// Implement ApproxEq on all integer types
impl_int_ApproxEq!(for isize, usize, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

pub struct FitnessFunc<'a, G, F>
where
    G: BitString,
    F: Default + Copy + ApproxEq + Debug,
{
    counter: Arc<Mutex<usize>>,
    evaluation_func: &'a (dyn Fn(&mut Individual<G, F>) -> F + Send + Sync),
    comparison_func: &'a (dyn Fn(&Individual<G, F>, &Individual<G, F>) -> Ordering + Send + Sync),
}

impl<'a, G, F> FitnessFunc<'a, G, F>
where
    G: BitString,
    F: Default + Copy + ApproxEq + Debug,
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
