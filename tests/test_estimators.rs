//! Test the numerical operations of estimator implementations.
//!
//! [`covariance`], [`information`], [`information_root`], [`ud`] and [`unscented`] linearised estimator implementations are tested.
//! [`sir`] sampled estimator is tested using the same model as for the linearised estimators.
//!
//! Tests are performed with Dynamic matrices and matrices with fixed dimensions.

use na::{Dim, Dynamic};
use na::DVector;
use na::U2;
use na::Vector2;
use nalgebra as na;

use bayes_estimate::estimators::information_root::InformationRootState;
use bayes_estimate::estimators::sir::SampleState;
use bayes_estimate::estimators::ud::UDState;
use bayes_estimate::models::{InformationState, KalmanState};
use fat_estimators::*;

mod fat_estimators;

#[test]
fn test_covariance() {
    simple::test_estimator(&mut KalmanState::new_zero(U2));
    simple::test_estimator(&mut KalmanState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_information() {
    simple::test_estimator(&mut InformationState::new_zero(U2));
    simple::test_estimator(&mut InformationState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_information_root() {
    simple::test_estimator(&mut InformationRootState::new_zero(U2));
    simple::test_estimator(&mut InformationRootState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_ud() {
    simple::test_estimator(&mut FatUDState {
        ud: UDState::new_zero(U2),
        obs_uncorrelated: true,
    });
    simple::test_estimator(&mut FatUDState {
        ud: UDState::new_zero(U2),
        obs_uncorrelated: false,
    });
    simple::test_estimator(&mut FatUDState {
        ud: UDState::new_zero(Dynamic::new(2)),
        obs_uncorrelated: true,
    });
    simple::test_estimator(&mut FatUDState {
        ud: UDState::new_zero(Dynamic::new(2)),
        obs_uncorrelated: false,
    });
}

#[test]
fn test_unscented() {
    simple::test_estimator(&mut FatUnscentedState::new_zero(U2));
    simple::test_estimator(&mut FatUnscentedState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_sir() {
    let rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(1u64);
    {   // Vector2
        let samples = vec![Vector2::<f64>::zeros(); 10000];
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples.clone(), Box::new(rng.clone())),
            systematic_resampler: false,
            kalman_roughening: false
        });
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples.clone(), Box::new(rng.clone())),
            systematic_resampler: true,
            kalman_roughening: false
        });
    }
    {   // DVector
        let samples = vec![DVector::zeros(2); 10000];
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples.clone(), Box::new(rng.clone())),
            systematic_resampler: false,
            kalman_roughening: false
        });
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples.clone(), Box::new(rng.clone())),
            systematic_resampler: true,
            kalman_roughening: false
        });
    }
    {   // DVector with kalman_roughening
        let samples = vec![DVector::zeros(2); 10000];
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples.clone(), Box::new(rng.clone())),
            systematic_resampler: false,
            kalman_roughening: true
        });
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples.clone(), Box::new(rng.clone())),
            systematic_resampler: true,
            kalman_roughening: true
        });
    }
}


pub mod simple {
    use approx;
    use na::{allocator::Allocator, DefaultAllocator, U1, U2};
    use na::{Dim, RealField};
    use na::{Matrix, Matrix1, Matrix1x2, Matrix2, Matrix2x1, MatrixMN, Vector1, Vector2, VectorN};
    use na::base::storage::Storage;
    use nalgebra as na;
    use num_traits::pow;

    use bayes_estimate::models::{KalmanEstimator, KalmanState};
    use bayes_estimate::noise::{CorrelatedNoise, CoupledNoise};

    use super::fat_estimators::*;

    const DT: f64 = 0.01;
    // Velocity noise, giving mean squared error bound
    const V_GAMMA: f64 = 1.;
    // Velocity correlation, giving velocity change time constant
    const V_NOISE: f64 = 0.2;

    // Filter's Initial state uncertainty
    const I_P_NOISE: f64 = 2.;
    const I_V_NOISE: f64 = 0.1;
    // Noise on observing system state
    const OBS_NOISE: f64 = 0.1;

    /// Simple prediction model.
    fn fx<D: super::Dim>(x: &VectorN<f64, D>) -> VectorN<f64, D>
        where
            DefaultAllocator: Allocator<f64, D>,
    {
        let f_vv: f64 = (-DT * V_GAMMA).exp();

        let mut xp = (*x).clone();
        xp[0] += DT * x[1];
        xp[1] *= f_vv;
        xp
    }

    /// Simple observation model.
    fn hx<D: Dim>(x: &VectorN<f64, D>) -> Vector1<f64>
        where
            DefaultAllocator: Allocator<f64, D>,
    {
        Vector1::new(x[0])
    }


    /// Numerically test the estimation operations of a TestEstimator.
    ///
    /// Prediction und observation operations are performed and the expected KalmanState is checked.
    pub fn test_estimator<D: Dim>(est: &mut dyn FatEstimator<D>)
        where
            DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>
            + Allocator<f64, U1, D> + Allocator<f64, D, U1> + Allocator<f64, U1>
            + Allocator<f64, U2, U2>
            + Allocator<usize, D, D> + Allocator<usize, D>,
    {
        let d = est.dim();

        let f_vv: f64 = (-DT * V_GAMMA).exp();

        let linear_pred_model = new_copy(d, d, Matrix2::new(1., DT, 0., f_vv));
        let additive_noise = CoupledNoise {
            q: Vector1::new(DT * pow((1. - f_vv) * V_NOISE, 2)),
            G: new_copy(d, U1, Matrix2x1::new(0.0, 1.0)),
        };

        let linear_obs_model = new_copy(U1, d, Matrix1x2::new(1.0, 0.0));
        let co_obs_noise = CorrelatedNoise {
            Q: Matrix1::new(pow(OBS_NOISE, 2)),
        };
        let z = &Vector1::new(1000.);

        let init_state: KalmanState<f64, D> = KalmanState {
            x: new_copy(d, U1, Vector2::new(1000., 1.5)),
            X: new_copy(d, d, Matrix2::new(pow(I_P_NOISE, 2), 0.0, 0.0, pow(I_V_NOISE, 2))),
        };

        est.init(&init_state).unwrap();

        let xx = est.kalman_state().unwrap();
        println!("init={:.6}{:.6}", xx.x, xx.X);
        est.trace_state();

        for _c in 0..2 {
            let predict_x = est.state().unwrap();
            let predict_xp = fx(&predict_x);
            est.predict_fn(&predict_xp, fx, &linear_pred_model, &additive_noise);
            let pp = KalmanEstimator::kalman_state(est).unwrap();
            println!("pred={:.6}{:.6}", pp.x, pp.X);
            est.trace_state();

            est.observe(&z, hx, &linear_obs_model, &co_obs_noise).unwrap();

            let oo = est.kalman_state().unwrap();
            println!("obs={:.6}{:.6}", oo.x, oo.X);
            est.trace_state();
        }

        est.observe(&z, hx, &linear_obs_model, &co_obs_noise).unwrap();

        let xx = est.kalman_state().unwrap();
        println!("final={:.6}{:.6}", xx.x, xx.X);

        expect_state(&KalmanState::<f64, D> { x: xx.x, X: xx.X }, 0.5 * est.allow_error_by());
    }

    /// Test the KalmanState is as expected.
    fn expect_state<D: Dim>(state: &KalmanState<f64, D>, allow_by: f64)
        where
            DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
    {
        let expect_x = Vector2::new(1000.004971, 1.470200);
        approx::assert_relative_eq!(state.x[0], expect_x[0], max_relative = 0.00000001 * allow_by);
        approx::assert_relative_eq!(state.x[1], expect_x[1], max_relative = 0.01 * allow_by);

        approx::assert_abs_diff_eq!(state.X[(0,0)], 0.003331, epsilon = 0.000001 * allow_by);
        approx::assert_abs_diff_eq!(state.X[(0,1)], 0.000032, epsilon = 0.000001 * allow_by);
        approx::assert_abs_diff_eq!(state.X[(1,1)], 0.009607, epsilon = 0.000003 * allow_by);
    }

    /// Create a Dynamic or Static copy.
    fn new_copy<N: RealField, R: Dim, C: Dim, R1: Dim, C1: Dim, S1: Storage<N, R1, C1>>(
        r: R,
        c: C,
        m: Matrix<N, R1, C1, S1>,
    ) -> MatrixMN<N, R, C>
        where
            DefaultAllocator: Allocator<N, R, C>
    {
        MatrixMN::<N, R, C>::from_iterator_generic(r, c, m.iter().cloned())
    }
}