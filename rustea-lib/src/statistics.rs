use ndarray::{Array, Ix1, Ix2};
use ndarray_linalg::{Cholesky, UPLO};
use rand::Rng;

#[allow(dead_code)]
fn sample_multivariate_normal(
    mean: &Array<f64, Ix1>,
    covariance: &Array<f64, Ix2>,
) -> Array<f64, Ix1> {
    let n = mean.len();

    // Cholesky decomposition
    let lower = covariance.cholesky(UPLO::Lower).unwrap();

    // Sample 'n' standard normal variables
    let mut rng = rand::rng();
    let random_vec: Array<f64, Ix1> = (0..n)
        .map(|_| rng.sample(rand_distr::StandardNormal))
        .collect();

    // Scale and translate the random sample (L*v + mean)
    lower.dot(&random_vec) + mean
}

#[cfg(test)]
mod tests {
    use std::ops::Add;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn mean_of_samples_approx_equals_mean_of_dist() {
        const NUM_SAMPLES: usize = 100000; // sample size
        const N: usize = 4; // dimension

        let mut rng = rand::rng();

        // Create random covariance matrix and mean vector
        let mean: Array<f64, Ix1> = (0..N)
            .map(|_| rng.sample(rand_distr::StandardNormal))
            .collect();
        let variance: Array<f64, Ix1> = Array::ones(N) / 10.0;
        let covariance = Array::from_diag(&variance);

        // Draw samples from multivariate normal distribution
        let samples: Vec<_> = (0..NUM_SAMPLES)
            .map(|_| sample_multivariate_normal(&mean, &covariance))
            .collect();

        // Calculate mean
        let mut sum_vec: Array<f64, Ix1> = Array::zeros(N);
        for sample in samples {
            sum_vec = sum_vec.add(&sample);
        }
        sum_vec /= NUM_SAMPLES as f64;

        // Assert that the sample mean approximately equals the true mean
        for i in 0..N {
            assert_relative_eq!(sum_vec[i], mean[i], epsilon = 0.01);
        }
    }
}
