//! Test the numerical operations of estimator implementations.
//!
//! [`covariance`], [`information`], [`information_root`], [`ud`] and [`unscented`] linearised estimator implementations are tested.
//! [`sir`] sampled estimator is tested using the same model as for the linearised estimators.
//!
//! Tests are performed with Dynamic matrices and matrices with fixed dimensions.

use na::DVector;
use na::Vector2;
use na::U2;
use na::{Dim, Dynamic};
use nalgebra as na;
use nalgebra::allocator::Allocator;
use nalgebra::storage::Storage;
use nalgebra::{DefaultAllocator, Matrix, MatrixMN, RealField, U3};

use bayes_estimate::estimators::information_root::InformationRootState;
use bayes_estimate::estimators::sir::SampleState;
use bayes_estimate::estimators::ud::UdState;
use bayes_estimate::models::{InformationState, KalmanState};
use fat_estimators::*;

mod fat_estimators;

#[test]
fn test_covariance() {
    simple::test_estimator(&mut KalmanState::new_zero(U2));
    simple::test_estimator(&mut KalmanState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_covariance_rtheta() {
    rtheta::test_estimator(&mut KalmanState::new_zero(U3));
    rtheta::test_estimator(&mut KalmanState::new_zero(Dynamic::new(3)));
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
    simple::test_estimator(&mut FatUdState {
        ud: UdState::new_zero(U2),
        obs_uncorrelated: true,
    });
    simple::test_estimator(&mut FatUdState {
        ud: UdState::new_zero(U2),
        obs_uncorrelated: false,
    });
    simple::test_estimator(&mut FatUdState {
        ud: UdState::new_zero(Dynamic::new(2)),
        obs_uncorrelated: true,
    });
    simple::test_estimator(&mut FatUdState {
        ud: UdState::new_zero(Dynamic::new(2)),
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
    {
        // Vector2
        let samples = vec![Vector2::<f64>::zeros(); 10000];
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples.clone(), Box::new(rng.clone())),
            systematic_resampler: false,
            kalman_roughening: false,
        });
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples, Box::new(rng.clone())),
            systematic_resampler: true,
            kalman_roughening: false,
        });
    }
    {
        // DVector
        let samples = vec![DVector::zeros(2); 10000];
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples.clone(), Box::new(rng.clone())),
            systematic_resampler: false,
            kalman_roughening: false,
        });
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples, Box::new(rng.clone())),
            systematic_resampler: true,
            kalman_roughening: false,
        });
    }
    {
        // DVector with kalman_roughening
        let samples = vec![DVector::zeros(2); 10000];
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples.clone(), Box::new(rng.clone())),
            systematic_resampler: false,
            kalman_roughening: true,
        });
        simple::test_estimator(&mut FatSampleState {
            sample: SampleState::new_equal_likelihood(samples, Box::new(rng)),
            systematic_resampler: true,
            kalman_roughening: true,
        });
    }
}

/// Create a Dynamic or Static copy.
fn new_copy<N: RealField, R: Dim, C: Dim, R1: Dim, C1: Dim, S1: Storage<N, R1, C1>>(
    r: R,
    c: C,
    m: Matrix<N, R1, C1, S1>,
) -> MatrixMN<N, R, C>
where
    DefaultAllocator: Allocator<N, R, C>,
{
    MatrixMN::<N, R, C>::from_iterator_generic(r, c, m.iter().cloned())
}

pub mod simple {
    use na::Dim;
    use na::{allocator::Allocator, DefaultAllocator, U1, U2};
    use na::{Matrix1, Matrix1x2, Matrix2, Matrix2x1, Vector1, Vector2, VectorN};
    use nalgebra as na;

    use bayes_estimate::models::{KalmanEstimator, KalmanState};
    use bayes_estimate::noise::{CorrelatedNoise, CoupledNoise};

    use super::fat_estimators::*;
    use super::new_copy;

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
    fn f<D: super::Dim>(x: &VectorN<f64, D>) -> VectorN<f64, D>
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
    fn h<D: Dim>(x: &VectorN<f64, D>) -> Vector1<f64>
    where
        DefaultAllocator: Allocator<f64, D>,
    {
        Vector1::new(x[0])
    }

    fn h_normalize(_z: &mut VectorN<f64, U1>, _z0: &VectorN<f64, U1>) {}

    /// Numerically test the estimation operations of a TestEstimator.
    ///
    /// Prediction und observation operations are performed and the expected KalmanState is checked.
    pub fn test_estimator<D: Dim>(est: &mut impl FatEstimator<D, U1, U1>)
    where
        DefaultAllocator: Allocator<f64, D, D>
            + Allocator<f64, D>
            + Allocator<f64, U1, D>
            + Allocator<f64, D, U1>
            + Allocator<f64, U1>
            + Allocator<f64, U2, U2>
            + Allocator<usize, D, D>
            + Allocator<usize, D>,
    {
        let d = est.dim();

        let f_vv: f64 = (-DT * V_GAMMA).exp();

        let linear_pred_model = new_copy(d, d, Matrix2::new(1., DT, 0., f_vv));
        let additive_noise = CoupledNoise {
            q: Vector1::new(DT * ((1. - f_vv) * V_NOISE).powi(2)),
            G: new_copy(d, U1, Matrix2x1::new(0.0, 1.0)),
        };

        let linear_obs_model = new_copy(U1, d, Matrix1x2::new(1.0, 0.0));
        let co_obs_noise = CorrelatedNoise {
            Q: Matrix1::new(OBS_NOISE.powi(2)),
        };
        let z = &Vector1::new(1000.);

        let init_state: KalmanState<f64, D> = KalmanState {
            x: new_copy(d, U1, Vector2::new(1000., 1.5)),
            X: new_copy(
                d,
                d,
                Matrix2::new(I_P_NOISE.powi(2), 0.0, 0.0, I_V_NOISE.powi(2)),
            ),
        };

        est.init(&init_state).unwrap();

        let xx = est.kalman_state().unwrap();
        println!("init={:.6}{:.6}", xx.x, xx.X);
        est.trace_state();

        for _c in 0..2 {
            let predict_x = est.state().unwrap();
            let predict_xp = f(&predict_x);
            est.predict_fn(&predict_xp, f, &linear_pred_model, &additive_noise);
            let pp = KalmanEstimator::kalman_state(est).unwrap();
            println!("pred={:.6}{:.6}", pp.x, pp.X);
            est.trace_state();

            est.observe(&z, h, h_normalize, &linear_obs_model, &co_obs_noise)
                .unwrap();

            let oo = est.kalman_state().unwrap();
            println!("obs={:.6}{:.6}", oo.x, oo.X);
            est.trace_state();
        }

        est.observe(&z, h, h_normalize, &linear_obs_model, &co_obs_noise)
            .unwrap();

        let xx = est.kalman_state().unwrap();
        println!("final={:.6}{:.6}", xx.x, xx.X);

        expect_state(
            &KalmanState::<f64, D> { x: xx.x, X: xx.X },
            0.5 * est.allow_error_by(),
        );
    }

    /// Test the KalmanState is as expected.
    fn expect_state<D: Dim>(state: &KalmanState<f64, D>, allow_by: f64)
    where
        DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
    {
        let expect_x = Vector2::new(1000.004971, 1.470200);
        approx::assert_relative_eq!(
            state.x[0],
            expect_x[0],
            max_relative = 0.00000001 * allow_by
        );
        approx::assert_relative_eq!(state.x[1], expect_x[1], max_relative = 0.01 * allow_by);

        approx::assert_abs_diff_eq!(state.X[(0, 0)], 0.003331, epsilon = 0.000001 * allow_by);
        approx::assert_abs_diff_eq!(state.X[(0, 1)], 0.000032, epsilon = 0.000001 * allow_by);
        approx::assert_abs_diff_eq!(state.X[(1, 1)], 0.009607, epsilon = 0.000003 * allow_by);
    }
}

pub mod rtheta {
    use std::f64::consts::PI;

    use na::base::storage::Storage;
    use na::Dim;
    use na::{allocator::Allocator, DefaultAllocator, U1, U2};
    use na::{Matrix2, Matrix3, Matrix3x2, Vector2, Vector3, VectorN};
    use nalgebra as na;

    use bayes_estimate::models::{Estimator, KalmanEstimator, KalmanState};
    use bayes_estimate::noise::{CorrelatedNoise, CoupledNoise};

    use super::fat_estimators::*;
    use super::new_copy;
    use nalgebra::MatrixMN;

    const NOISE_MODEL: bool = true;

    const RANGE_NOISE: f64 = if NOISE_MODEL { 0.1 } else { 1e-6 };
    const ANGLE_NOISE: f64 = if NOISE_MODEL {
        0.1f64 * PI / 180.
    } else {
        1e-6
    };
    const Z_CORRELATION: f64 = 0e-1; // (Un)Correlated observation model

    // predict model
    const X_NOISE: f64 = 0.05;
    const Y_NOISE: f64 = 0.09;
    const XY_NOISE_COUPLING: f64 = 0.05;
    const G_COUPLING: f64 = 1.0; // Coupling in addition G terms

    const INIT_X_NOISE: f64 = 0.07;
    const INIT_Y_NOISE: f64 = 0.10;
    const INIT_XY_NOISE_CORRELATION: f64 = 0.; //0.4;
    const INIT_2_NOISE: f64 = 0.09;
    // Use zero for singular X
    const INIT_Y2_NOISE_CORRELATION: f64 = 0.5;

    // XY position of target - this is chosen so the observation angle is discontinues at -pi/+pi
    const TARGET: [f64; 2] = [-11., 0.];

    /// Coupled prediction model.
    fn f<D: super::Dim>(x: &VectorN<f64, D>) -> VectorN<f64, D>
    where
        DefaultAllocator: Allocator<f64, D>,
    {
        let mut xp = VectorN::zeros_generic(x.data.shape().0, U1);
        xp[0] = x[0];
        xp[1] = 0.1 * x[0] + 0.9 * x[1];
        xp[2] = x[2];
        xp
    }

    /// range, angle observation model.
    fn h<D: Dim>(x: &VectorN<f64, D>) -> Vector2<f64>
    where
        DefaultAllocator: Allocator<f64, D>,
    {
        let dx = TARGET[0] - x[0];
        let dy = TARGET[1] - x[1];

        Vector2::new((dx * dx + dy * dy).sqrt(), dy.atan2(dx))
    }

    fn normalize_angle(a: &mut f64, a0: f64) {
        let mut d = (*a - a0) % 2. * PI;
        if d >= PI {
            d -= 2. * PI
        } else if d < -PI {
            d += 2. * PI
        }
        *a += d;
    }

    fn h_normalize(z: &mut VectorN<f64, U2>, z0: &VectorN<f64, U2>) {
        normalize_angle(&mut z[1], z0[1]);
    }

    fn hx<D: Dim>(x: &VectorN<f64, D>) -> MatrixMN<f64, U2, D>
    where
        DefaultAllocator: Allocator<f64, U2, D> + Allocator<f64, D>,
    {
        let dx = TARGET[0] - x[0];
        let dy = TARGET[1] - x[1];

        let mut hx = MatrixMN::zeros_generic(U2, x.data.shape().0);
        let dist_sq = dx * dx + dy * dy;
        let dist = dist_sq.sqrt();
        hx[(0, 0)] = -dx / dist;
        hx[(0, 1)] = -dy / dist;
        hx[(1, 0)] = dy / dist_sq;
        hx[(1, 1)] = -dx / dist_sq;
        hx
    }

    /// Numerically test the estimation operations of a TestEstimator.
    ///
    /// Prediction und observation operations are performed and the expected KalmanState is checked.
    pub fn test_estimator<D: Dim>(est: &mut impl FatEstimator<D, U2, U2>)
    where
        DefaultAllocator: Allocator<f64, D, D>
            + Allocator<f64, D>
            + Allocator<f64, U1, D>
            + Allocator<f64, D, U1>
            + Allocator<f64, U1>
            + Allocator<f64, U2, D>
            + Allocator<f64, D, U2>
            + Allocator<f64, U2>
            + Allocator<f64, U2, U2>
            + Allocator<usize, D, D>
            + Allocator<usize, D>,
    {
        let d = est.dim();

        let linear_pred_model = new_copy(d, d, Matrix3::new(1., 0.1, 0., 0., 0.9, 0., 0., 0., 1.));
        let additive_noise = CoupledNoise {
            q: Vector2::new(X_NOISE.powi(2), Y_NOISE.powi(2)),
            G: new_copy(
                d,
                U2,
                Matrix3x2::new(
                    1.0,
                    XY_NOISE_COUPLING,
                    XY_NOISE_COUPLING,
                    1.0,
                    G_COUPLING,
                    G_COUPLING,
                ),
            ),
        };

        let co_obs_noise = CorrelatedNoise {
            Q: Matrix2::new(
                RANGE_NOISE.powi(2),
                RANGE_NOISE * ANGLE_NOISE * Z_CORRELATION,
                RANGE_NOISE * ANGLE_NOISE * Z_CORRELATION,
                ANGLE_NOISE.powi(2),
            ),
        };

        // True position
        let truth = Vector2::new(1., 0.);

        const INIT_XY_CORRELATION: f64 = INIT_X_NOISE * INIT_Y_NOISE * INIT_XY_NOISE_CORRELATION;
        const INIT_Y2_CORRELATION: f64 = INIT_Y_NOISE * INIT_2_NOISE * INIT_Y2_NOISE_CORRELATION;
        let init_state: KalmanState<f64, D> = KalmanState {
            // Choose initial y so that the estimated position should pass through 0 and thus the observed angle be discontinues
            x: new_copy(d, U1, Vector3::new(1., -0.2, 0.0)),
            X: new_copy(
                d,
                d,
                Matrix3::new(
                    INIT_X_NOISE.powi(2),
                    INIT_XY_CORRELATION,
                    0.,
                    INIT_XY_CORRELATION,
                    INIT_Y_NOISE.powi(2),
                    INIT_Y2_CORRELATION,
                    0.,
                    INIT_Y2_CORRELATION,
                    INIT_2_NOISE.powi(2),
                ),
            ),
        };

        est.init(&init_state).unwrap();

        let xx = est.kalman_state().unwrap();
        println!("init={:.6}{:.6}", xx.x, xx.X);
        est.trace_state();

        for _c in 0..2 {
            let predict_x = est.state().unwrap();
            let predict_xp = f(&predict_x);
            est.predict_fn(&predict_xp, f, &linear_pred_model, &additive_noise);
            let pp = KalmanEstimator::kalman_state(est).unwrap();
            println!("pred={:.6}{:.6}", pp.x, pp.X);
            est.trace_state();

            let z = h(&truth);
            let hx = hx(&pp.x);
            est.observe(&z, h, h_normalize, &hx, &co_obs_noise).unwrap();

            let oo = est.kalman_state().unwrap();
            println!("obs={:.6}{:.6}", oo.x, oo.X);
            est.trace_state();

            // Jump the true postion back an forth around the y axis, making the observes angle discontinues
            // truth[1] = -truth[1]
        }

        let est_state = Estimator::state(est).unwrap();
        let z = h(&truth);
        let hx = hx(&est_state);
        est.observe(&z, h, h_normalize, &hx, &co_obs_noise).unwrap();

        let xx = est.kalman_state().unwrap();
        println!("final={:.6}{:.6}", xx.x, xx.X);
    }
}
