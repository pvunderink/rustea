use std::{
    cmp::Ordering,
    fmt::Debug,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use crate::{
    genome::{Genotype, SampleUniformRange},
    individual::Individual,
};

pub enum OptimizationGoal {
    MINIMIZE,
    MAXIMIZE,
}

pub struct FitnessFunc<'a, Gnt, T, F>
where
    Gnt: Genotype<T>,
    T: Copy + Send + Sync + SampleUniformRange,
    F: Default + Copy + Debug,
{
    counter: Arc<Mutex<usize>>,
    evaluation_func: &'a (dyn Fn(&Gnt) -> F + Send + Sync),
    goal: OptimizationGoal,
    _gene: PhantomData<T>,
}

impl<'a, Gnt, T, F> FitnessFunc<'a, Gnt, T, F>
where
    Gnt: Genotype<T>,
    T: Copy + Send + Sync + SampleUniformRange,
    F: Default + Copy + Debug + PartialOrd,
{
    pub fn new(
        evaluation_func: &'a (dyn Fn(&Gnt) -> F + Send + Sync),
        goal: OptimizationGoal,
    ) -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            evaluation_func,
            goal,
            _gene: PhantomData::default(),
        }
    }

    pub fn evaluate(&self, individual: &mut Individual<Gnt, T, F>) -> F {
        let fitness = (self.evaluation_func)(individual.genotype());
        individual.update_fitness(fitness);

        let mut counter = self.counter.lock().unwrap();
        *counter += 1;

        fitness
    }

    pub fn evaluations(&self) -> usize {
        *self.counter.lock().unwrap()
    }

    pub fn cmp(&self, a: &F, b: &F) -> Ordering {
        match self.goal {
            OptimizationGoal::MINIMIZE => a.partial_cmp(&b).unwrap(),
            OptimizationGoal::MAXIMIZE => b.partial_cmp(&a).unwrap(),
        }
    }
}
