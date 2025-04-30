use rand::{Rng, SeedableRng};
// use rand_pcg::Pcg64;
use rand_xoshiro::{Xoshiro256PlusPlus, Xoshiro256StarStar};

pub trait RngGenerator<R>
where
    R: SeedableRng + ?Sized,
{
    fn from_seed(seed: u64) -> Self;
    fn next(&mut self) -> R;
}

pub struct XoshiroRngGenerator<R>
where
    R: Rng + ?Sized,
{
    rng: R,
}

impl RngGenerator<Xoshiro256PlusPlus> for XoshiroRngGenerator<Xoshiro256PlusPlus> {
    fn from_seed(seed: u64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    fn next(&mut self) -> Xoshiro256PlusPlus {
        let mut new_rng = self.rng.clone();
        new_rng.jump();
        let rng = self.rng.clone();
        self.rng = new_rng;
        rng
    }
}

impl RngGenerator<Xoshiro256StarStar> for XoshiroRngGenerator<Xoshiro256StarStar> {
    fn from_seed(seed: u64) -> Self {
        Self {
            rng: Xoshiro256StarStar::seed_from_u64(seed),
        }
    }
    fn next(&mut self) -> Xoshiro256StarStar {
        let mut new_rng = self.rng.clone();
        new_rng.jump();
        let rng = self.rng.clone();
        self.rng = new_rng;
        rng
    }
}

// impl RngGenerator<Pcg64> for XoshiroRngGenerator<Xoshiro256StarStar> {
//     fn from_seed(seed: u64) -> Self {
//         Self {
//             rng: Xoshiro256StarStar::seed_from_u64(seed),
//         }
//     }
//     fn next(&self) -> Xoshiro256StarStar {
//         let mut rng = self.rng.clone();
//         rng.jump();
//         rng
//     }
// }
