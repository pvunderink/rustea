use ndarray::{Array, Ix1, Ix2};
use ndarray_linalg::{Cholesky, UPLO};
use rand::Rng;

fn sample_multivariate_normal(
    mean: &Array<f64, Ix1>,
    covariance: &Array<f64, Ix2>,
) -> Array<f64, Ix1> {
    let n = mean.len();
    // Cholesky decomposition
    let lower = covariance.cholesky(UPLO::Lower).unwrap();

    let mut rng = rand::thread_rng();

    let random_vec: Array<f64, Ix1> = (0..n)
        .map(|_| rng.sample(rand_distr::StandardNormal))
        .collect();

    lower.dot(&random_vec) + mean
}

#[cfg(test)]
mod tests {
    use std::ops::Add;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn mean_of_samples_approx_equals_mean_of_dist() {
        let mut rng = rand::thread_rng();

        const N: usize = 4;

        // Create random variance and mean vectors
        let mean: Array<f64, Ix1> = (0..N)
            .map(|_| rng.sample(rand_distr::StandardNormal))
            .collect();

        let variance: Array<f64, Ix1> = Array::ones(N) / 10.0;
        let covariance = Array::from_diag(&variance);

        println!("{:?}", covariance);

        const NUM_SAMPLES: usize = 100000;

        let samples: Vec<_> = (0..NUM_SAMPLES)
            .map(|_| sample_multivariate_normal(&mean, &covariance))
            .collect();

        let mut sum_vec: Array<f64, Ix1> = Array::zeros(N);

        for sample in samples {
            sum_vec = sum_vec.add(&sample);
        }
        sum_vec /= NUM_SAMPLES as f64;

        println!("{:?}", mean);
        println!("{:?}", sum_vec);

        for i in 0..N {
            assert_relative_eq!(sum_vec[i], mean[i], epsilon = 0.01);
        }
    }
}
