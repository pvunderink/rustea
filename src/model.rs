use crate::{
    fitness::Fitness,
    gene::{Allele, Discrete, DiscreteDomain, DiscreteGene},
    genome::Genome,
    genotype::Genotype,
    individual::Individual,
    types::CollectUnsafe,
};
use approx::abs_diff_ne;
use rand::Rng;
use rand_distr::WeightedIndex;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{marker::PhantomData, ops::Index};

#[derive(Debug)]
pub struct UnivariateModel<'a, Gnt, A, D, F, const LEN: usize>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    F: Fitness,
    Gnt: Genotype<A>,
{
    distributions: Vec<WeightedIndex<usize>>,
    genome: &'a Genome<A, DiscreteGene<A, D>, LEN>,
    _genotype: PhantomData<Gnt>,
    _fitness: PhantomData<F>,
}

impl<'a, Gnt, A, D, F, const LEN: usize> UnivariateModel<'a, Gnt, A, D, F, LEN>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    F: Fitness,
    Gnt: Genotype<A>,
{
    pub fn estimate_from_population(
        genome: &'a Genome<A, DiscreteGene<A, D>, LEN>,
        population: &[Individual<Gnt, A, F, LEN>],
    ) -> Self {
        assert!(!population.is_empty());

        let mut counts: Vec<Vec<usize>> = genome
            .iter()
            .map(|gene| gene.domain().iter().map(|_| 0).collect())
            .collect();

        for idv in population {
            for (idx, allele) in idv.genotype().iter().enumerate() {
                let vec = &mut counts[idx];
                let allele_idx = genome.get(idx).domain().index_of(allele);
                vec[allele_idx] += 1
            }
        }

        let distributions = counts
            .into_iter()
            .map(|counts| WeightedIndex::new(counts).unwrap())
            .collect();

        Self {
            distributions,
            genome,
            _genotype: PhantomData,
            _fitness: PhantomData,
        }
    }

    pub fn sample<R>(&self, rng: &mut R) -> Individual<Gnt, A, F, LEN>
    where
        R: Rng,
    {
        let genotype = self
            .genome
            .iter()
            .enumerate()
            .map(|(idx, gene)| gene.sample_with_weights(rng, &self.distributions[idx]))
            .collect_unsafe();

        Individual::from_genotype(genotype)
    }
}

#[derive(Debug)]
pub struct Factorization {
    factors: Vec<Vec<usize>>,
}

impl Index<usize> for Factorization {
    type Output = Vec<usize>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.factors[index]
    }
}

impl Factorization {
    pub fn univariate(len: usize) -> Self {
        Self {
            factors: (0..len).map(|i| vec![i]).collect(),
        }
    }

    pub fn join(&self, idx_a: usize, idx_b: usize) -> Self {
        let mut joined = self.factors[idx_a].clone();
        joined.extend(self.factors[idx_b].iter());

        let mut factors: Vec<_> = self
            .factors
            .iter()
            .enumerate()
            .filter_map(|(idx, vec)| {
                if idx == idx_a || idx == idx_b {
                    None
                } else {
                    Some(vec.clone())
                }
            })
            .collect();

        factors.push(joined);

        Self { factors }
    }

    pub fn join_all(&self) -> impl Iterator<Item = Self> + '_ {
        let n: usize = self.factors.len();
        (0..n - 1).flat_map(move |idx_a| (idx_a + 1..n).map(move |idx_b| self.join(idx_a, idx_b)))
    }

    // pub fn par_join_all(&self) -> impl ParallelIterator<Item = Self> + '_ {
    //     let n: usize = self.factors.len();
    //     (0..n - 1)
    //         .into_par_iter()
    //         .flat_map_iter(move |idx_a| (idx_a + 1..n).map(move |idx_b| self.join(idx_a, idx_b)))
    // }

    pub fn iter(&self) -> impl Iterator<Item = &Vec<usize>> + '_ {
        self.factors.iter().filter(|f| !f.is_empty())
    }

    pub fn iter_genotype<'a, Gnt, A>(
        &'a self,
        genotype: &'a Gnt,
    ) -> impl Iterator<Item = Vec<(usize, A)>> + '_
    where
        A: Allele + Discrete,
        Gnt: Genotype<A>,
    {
        self.iter()
            .map(|idxs| idxs.iter().map(|idx| (*idx, genotype.get(*idx))).collect())
    }
}

// struct JoinedFactorizationIterator<'a> {
//     factorization: Vec<Vec<usize>>,
//     current_idx_a: usize,
//     current_idx_b: usize,
//     old_a: Vec<usize>,
//     old_b: Vec<usize>,
//     _useless_ptr: &'a Factorization,
// }

// impl<'a> Iterator for JoinedFactorizationIterator<'a> {
//     type Item = &'a Factorization;

//     fn next(&'a mut self) -> Option<Self::Item> {
//         // repair from previous
//         if !(self.current_idx_a == 0 && self.current_idx_b == 0) {}

//         // evaluate next
//         self.current_idx_b += 1;

//         if self.current_idx_b == self.factorization.factors.len() {}

//         if self.current_idx_a == self.factorization.factors.len() {
//             return None;
//         }

//         {
//             // save copy of factor a
//             let factor_a = &self.factorization.factors[self.current_idx_a];
//             self.old_a.clear();
//             self.old_a.extend(factor_a.iter());
//         }

//         {
//             // save copy of factor b
//             let factor_b = &self.factorization.factors[self.current_idx_b];
//             self.old_b.clear();
//             self.old_b.extend(factor_b.iter());
//         }

//         // append contents of factor b to factor a
//         self.factorization.factors[self.current_idx_a]
//             .append(&mut self.factorization.factors[self.current_idx_b]);

//         // return current factorization
//         Some(&self.factorization)
//     }
// }

// impl ParallelIterator for JoinedFactorizationIterator {
//     type Item;

//     fn drive_unindexed<C>(self, consumer: C) -> C::Result
//     where
//         C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
//     {
//         todo!()
//     }
// }

impl PartialEq for Factorization {
    fn eq(&self, other: &Self) -> bool {
        self.factors == other.factors
    }
}

impl FromIterator<Vec<usize>> for Factorization {
    fn from_iter<T: IntoIterator<Item = Vec<usize>>>(iter: T) -> Self {
        Self {
            factors: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Factorization {
    type Item = Vec<usize>;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.factors.into_iter()
    }
}

#[derive(Debug)]
pub struct MultivariateModel<'a, Gnt, A, D, F, const LEN: usize>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    F: Fitness,
    Gnt: Genotype<A>,
{
    factorization: Factorization,
    probabilities: Vec<Vec<f64>>,
    genome: &'a Genome<A, DiscreteGene<A, D>, LEN>,
    sample_size: usize,
    _fitness: PhantomData<F>,
    _genotype: PhantomData<Gnt>,
}

impl<'a, Gnt, A, D, F, const LEN: usize> MultivariateModel<'a, Gnt, A, D, F, LEN>
where
    A: Allele + Discrete,
    D: DiscreteDomain<A>,
    F: Fitness,
    Gnt: Genotype<A>,
{
    pub fn estimate_from_population(
        genome: &'a Genome<A, DiscreteGene<A, D>, LEN>,
        population: &[&Individual<Gnt, A, F, LEN>],
        factorization: Factorization,
    ) -> Self {
        assert!(!population.is_empty());

        let mut counts: Vec<Vec<usize>> = factorization
            .iter()
            .map(|idxs| {
                let n = idxs
                    .iter()
                    .fold(1, |acc, idx| acc * genome.get(*idx).domain().len());
                vec![0; n]
            })
            .collect();

        for idv in population {
            for (factor_idx, alleles) in factorization.iter_genotype(idv.genotype()).enumerate() {
                let n = alleles.len();
                let idx: usize = alleles
                    .into_iter()
                    .enumerate()
                    .fold(vec![1usize; n], |acc, (i, (idx, allele))| {
                        let domain = genome.get(idx).domain();
                        let l = domain.len();
                        let mut new_acc = acc.clone();

                        (0..i).for_each(|j| new_acc[j] *= l);
                        new_acc[i] *= domain.index_of(allele);

                        new_acc
                    })
                    .iter()
                    .sum();

                let vec = &mut counts[factor_idx];
                vec[idx] += 1
            }
        }

        let probabilities = counts
            .into_iter()
            .map(|counts| {
                counts
                    .iter()
                    .map(|count| *count as f64 / population.len() as f64)
                    .collect()
            })
            .collect();

        Self {
            factorization,
            probabilities,
            genome,
            sample_size: population.len(),
            _fitness: PhantomData,
            _genotype: PhantomData,
        }
    }

    pub fn sample<R>(&self, rng: &mut R) -> Individual<Gnt, A, F, LEN>
    where
        R: Rng,
    {
        let mut alleles: Vec<A> = vec![A::default(); self.genome.len()];

        self.factorization
            .iter()
            .enumerate()
            .for_each(|(factor_idx, factor)| {
                let distr = WeightedIndex::new(&self.probabilities[factor_idx]).unwrap();
                let mut raw_idx = rng.sample(distr);

                // Extract allele indices from raw_idx
                let n = factor.len();
                let mut allele_idxs = vec![0usize; n];

                let divisors =
                    factor
                        .iter()
                        .enumerate()
                        .fold(vec![1usize; n], |acc, (i, gene_idx)| {
                            let domain = self.genome.get(*gene_idx).domain();
                            let l = domain.len();
                            let mut new_acc = acc.clone();

                            (0..i).for_each(|j| new_acc[j] *= l);
                            new_acc
                        });

                for i in 0..n {
                    let idx = raw_idx / divisors[i];
                    raw_idx = raw_idx % divisors[i];
                    allele_idxs[i] = idx;
                }

                factor
                    .iter()
                    .zip(allele_idxs.into_iter())
                    .for_each(|(gene_idx, allele_idx)| {
                        alleles[*gene_idx] = self.genome.get(*gene_idx).domain().get(allele_idx);
                    })
            });

        Individual::from_genotype(alleles.into_iter().collect_unsafe())
    }

    pub fn compressed_population_complexity(&self) -> f64 {
        let entropy_sum: f64 = self
            .probabilities
            .iter()
            .map(|probs| {
                probs
                    .iter()
                    .filter(|p| abs_diff_ne!(**p, 0.0, epsilon = 1e-5))
                    .map(|p| -p * p.log2())
                    .sum::<f64>()
            })
            .sum();
        self.sample_size as f64 * entropy_sum
    }

    pub fn model_complexity(&self) -> f64 {
        ((self.sample_size + 1) as f64).log2()
            * self
                .probabilities
                .iter()
                .map(|probs| probs.len() - 1)
                .sum::<usize>() as f64
    }

    pub fn combined_complexity(&self) -> f64 {
        self.compressed_population_complexity() + 0.2 * self.model_complexity()
    }

    pub fn factorization(&self) -> &Factorization {
        &self.factorization
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{bdom, BoolDomain};

    use super::*;

    #[test]
    fn join_univariate_factors_large() {
        const N: usize = 100;

        let factorization = Factorization::univariate(N);

        let iter = factorization.join_all();

        let factorizations: Vec<_> = iter.collect();

        assert_eq!(factorizations.len(), (N * (N - 1)) / 2)
    }

    #[test]
    fn join_univariate_factors_len_4() {
        const N: usize = 4;

        let factorization = Factorization::univariate(N);

        let iter = factorization.join_all();

        let factorizations: Vec<_> = iter.collect();

        assert_eq!(
            factorizations[0],
            vec![vec![2], vec![3], vec![0, 1]].into_iter().collect()
        );

        assert_eq!(
            factorizations[1],
            vec![vec![1], vec![3], vec![0, 2]].into_iter().collect()
        );

        assert_eq!(
            factorizations[2],
            vec![vec![1], vec![2], vec![0, 3]].into_iter().collect()
        );

        assert_eq!(
            factorizations[3],
            vec![vec![0], vec![3], vec![1, 2]].into_iter().collect()
        );

        assert_eq!(
            factorizations[4],
            vec![vec![0], vec![2], vec![1, 3]].into_iter().collect()
        );

        assert_eq!(
            factorizations[5],
            vec![vec![0], vec![1], vec![2, 3]].into_iter().collect()
        );

        assert_eq!(factorizations.len(), (N * (N - 1)) / 2)
    }

    #[test]
    fn multivariate_model_with_one_joined_factor() {
        const N: usize = 10;
        type Gnt = [bool; N];
        type Ftnss = f64;

        let genome = Genome::with_bool_domain();

        // Create factorization: [(0,1), 2, 3, 4, 5, 6, 7, 8, 9]
        let factorization = Factorization::univariate(N).join(0, 1);

        let mut rng = rand::thread_rng();

        // Create random population
        let population: Vec<_> = (0..100000)
            .map(|_| {
                let mut genotype: Gnt = genome.sample_uniform(&mut rng);
                if rng.gen::<f64>() < 0.25 {
                    genotype[0] = false;
                    genotype[1] = true;
                } else {
                    genotype[0] = true;
                    genotype[1] = false;
                }

                Individual::<_, _, Ftnss, N>::from_genotype(genotype)
            })
            .collect();

        let model = MultivariateModel::estimate_from_population(
            &genome,
            &population.iter().collect::<Vec<_>>(),
            factorization,
        );

        let mut counts = vec![0usize; N - 2];

        let mut count_00 = 0usize;
        let mut count_01 = 0usize;
        let mut count_10 = 0usize;
        let mut count_11 = 0usize;

        const SAMPLE_SIZE: usize = 100000;
        let samples: Vec<_> = (0..SAMPLE_SIZE).map(|_| model.sample(&mut rng)).collect();

        for sample in samples {
            let first_factor: Vec<_> = sample.genotype().iter().take(2).collect();

            if first_factor == vec![false, false] {
                count_00 += 1;
            } else if first_factor == vec![false, true] {
                count_01 += 1;
            } else if first_factor == vec![true, false] {
                count_10 += 1;
            } else if first_factor == vec![true, true] {
                count_11 += 1;
            }

            for (i, b) in sample.genotype().iter().skip(2).enumerate() {
                if b {
                    counts[i] += 1;
                }
            }
        }

        assert_eq!(count_00, 0);
        assert_eq!(count_11, 0);
        assert_abs_diff_eq!(count_01 as f64 / SAMPLE_SIZE as f64, 0.25, epsilon = 0.01);
        assert_abs_diff_eq!(count_10 as f64 / SAMPLE_SIZE as f64, 0.75, epsilon = 0.01);

        for count in counts {
            assert_abs_diff_eq!(count as f64 / SAMPLE_SIZE as f64, 0.5, epsilon = 0.01);
        }
    }

    #[test]
    fn compressed_population_complexity() {
        type Ftnss = f64;
        const N: usize = 4;

        let genome = Genome::with_bool_domain();

        // Create random population
        let population: Vec<_> = vec![
            Individual::<_, _, Ftnss, N>::from_genotype([true, false, false, false]),
            Individual::from_genotype([true, true, false, true]),
            Individual::from_genotype([false, true, true, true]),
            Individual::from_genotype([true, true, false, false]),
            Individual::from_genotype([false, false, true, false]),
            Individual::from_genotype([false, true, true, true]),
            Individual::from_genotype([true, false, false, false]),
            Individual::from_genotype([true, false, false, true]),
        ];

        let univariate_model = MultivariateModel::estimate_from_population(
            &genome,
            &population.iter().collect::<Vec<_>>(),
            Factorization::univariate(N),
        );

        let joined_model = MultivariateModel::estimate_from_population(
            &genome,
            &population.iter().collect::<Vec<_>>(),
            Factorization::univariate(N).join(0, 2),
        );

        assert_abs_diff_eq!(
            univariate_model.compressed_population_complexity(),
            31.3,
            epsilon = 0.1
        );

        assert_abs_diff_eq!(
            joined_model.compressed_population_complexity(),
            23.6,
            epsilon = 0.1
        );
    }

    #[test]
    fn model_complexity() {
        type Ftnss = f64;
        const N: usize = 4;

        let genome = Genome::with_bool_domain();

        // Create random population
        let population: Vec<_> = vec![
            Individual::<_, _, Ftnss, N>::from_genotype([true, false, false, false]),
            Individual::from_genotype([true, true, false, true]),
            Individual::from_genotype([false, true, true, true]),
            Individual::from_genotype([true, true, false, false]),
            Individual::from_genotype([false, false, true, false]),
            Individual::from_genotype([false, true, true, true]),
            Individual::from_genotype([true, false, false, false]),
            Individual::from_genotype([true, false, false, true]),
        ];

        let univariate_model = MultivariateModel::estimate_from_population(
            &genome,
            &population.iter().collect::<Vec<_>>(),
            Factorization::univariate(N),
        );

        let joined_model = MultivariateModel::estimate_from_population(
            &genome,
            &population.iter().collect::<Vec<_>>(),
            Factorization::univariate(N).join(0, 2),
        );

        assert_abs_diff_eq!(univariate_model.model_complexity(), 12.7, epsilon = 0.1);

        assert_abs_diff_eq!(joined_model.model_complexity(), 15.8, epsilon = 0.1);
    }
}
