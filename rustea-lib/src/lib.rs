#![feature(array_chunks)]

pub mod ecga;
pub mod fitness;
pub mod gene;
pub mod genome;
pub mod genotype;
pub mod individual;
pub mod model;
#[cfg(feature = "rngs")]
pub mod rng;
pub mod selection;
pub mod simplega;
#[cfg(feature = "statistics")]
pub mod statistics;
pub mod types;
pub mod variation;
