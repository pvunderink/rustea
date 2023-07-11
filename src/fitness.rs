use std::{
    cmp::Ordering,
    fmt::Debug,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use crate::{gene::Allele, genotype::Genotype, individual::Individual};

#[derive(Debug, Clone)]
pub enum OptimizationGoal {
    Minimize,
    Maximize,
}

pub trait Fitness: Default + Copy + Debug + Send + Sync + PartialOrd {}

macro_rules! impl_fitness {
    (for $($ty:ty),+) => {
        $(
            impl Fitness for $ty {}

        )*
    };
}

impl_fitness!(for u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize, f32, f64);

pub struct FitnessFunc<'a, Gnt, A, F>
where
    A: Allele,
    F: Fitness,
    Gnt: Genotype<A>,
{
    counter: Arc<Mutex<usize>>,
    evaluation_func: &'a (dyn Fn(&Gnt) -> F + Send + Sync),
    goal: OptimizationGoal,
    _gene: PhantomData<A>,
}

impl<'a, Gnt, A, F> FitnessFunc<'a, Gnt, A, F>
where
    A: Allele,
    F: Fitness,
    Gnt: Genotype<A>,
{
    pub fn new(
        evaluation_func: &'a (dyn Fn(&Gnt) -> F + Send + Sync),
        goal: OptimizationGoal,
    ) -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            evaluation_func,
            goal,
            _gene: PhantomData,
        }
    }

    pub fn evaluate(&self, individual: &mut Individual<Gnt, A, F>) -> F {
        let fitness = (self.evaluation_func)(individual.genotype());
        individual.set_fitness(fitness);

        let mut counter = self.counter.lock().unwrap();
        *counter += 1;

        fitness
    }

    pub fn evaluations(&self) -> usize {
        *self.counter.lock().unwrap()
    }

    pub fn cmp(&self, a: &F, b: &F) -> Ordering {
        match self.goal {
            OptimizationGoal::Minimize => a.partial_cmp(b).unwrap(),
            OptimizationGoal::Maximize => b.partial_cmp(a).unwrap(),
        }
    }
}
