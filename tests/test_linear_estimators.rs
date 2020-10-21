//! Test the numerical operations of linear and linearised estimator implementations.
//!
//! [`covariance`] [`information`] and [`ud`] estimator implementations are tested.
//!
//! [`covariance`]: ../filters/covariance.html
//! [`information`]: ../filters/information.html
//! [`ud`]: ../filters/ud.html
//!
//! Tests are performed with Dynamic matrices and matrices with fixed dimensions.

use na::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use na::base::storage::Storage;
use na::{allocator::Allocator, DefaultAllocator, U1, U2};
use na::{Dim, DimAdd, DimSum, Dynamic, VectorN};
use na::{Matrix, Matrix1, Matrix1x2, Matrix2, Matrix2x1, Vector1, Vector2};
use na::{MatrixMN, RealField};
use nalgebra as na;

use bayes_filter as bf;
use bf::estimators::ud::UDState;
use bf::models::{
    AdditiveNoise, InformationState, KalmanState, LinearObserveModel, LinearPredictModel,
};
use bf::models::{
    KalmanEstimator, LinearObserverCorrelated, LinearObserverUncorrelated, LinearPredictor,
};

use approx;
use bayes_filter::models::AdditiveCorrelatedNoise;
use bayes_filter::estimators::information_root::InformationRootState;

const DT: f64 = 0.01;
const V_NOISE: f64 = 0.1; // Velocity noise, giving mean squared error bound
const V_GAMMA: f64 = 1.; // Velocity correlation, giving velocity change time constant
                         // Filter's Initial state uncertainty: System state is unknown
const I_P_NOISE: f64 = 1000.;
const I_V_NOISE: f64 = 10.;
// Noise on observing system state
const OBS_NOISE: f64 = 0.001;

// Minimum allowable reciprocal condition number for PD Matrix factorisations
// Use 1e5  * epsilon give 5 decimal digits of headroom
const LIMIT_PD: f64 = std::f64::EPSILON * 1e5;


#[test]
fn test_covariance_u2() {
    test_estimator(&mut KalmanState::new(U2));
}

#[test]
fn test_information_u2() {
    test_estimator(&mut InformationState::new(U2));
}

#[test]
fn test_ud_u2() {
    test_estimator(&mut UDState::new(U2, U2.add(U1)));
}

#[test]
fn test_covariance_dynamic() {
    test_estimator(&mut KalmanState::new(Dynamic::new(2)));
}

#[test]
fn test_information_dynamic() {
    test_estimator(&mut InformationState::new(Dynamic::new(2)));
}

#[test]
fn test_ud_dynamic() {
    test_estimator(&mut UDState::new(Dynamic::new(2), Dynamic::new(3)));
}

#[test]
fn test_information_root_dynamic() {
    test_estimator(&mut InformationRootState::new(Dynamic::new(2)));
}


/// Checks a the reciprocal condition number exceeds a minimum.
///
/// IEC 559 NaN values are never true
fn check(res: Result<f64, &'static str>, what: &'static str) -> Result<f64, String> {
    match res {
        Ok(_) => {
            let rcond = res.unwrap();
            if rcond > LIMIT_PD {
                Result::Ok(rcond)
            } else {
                Result::Err(format!("{}: {}", what, rcond))
            }
        }
        Err(err) => Result::Err(err.to_string()),
    }
}

fn sqr(x: f64) -> f64 {
    x * x
}

/// Define the estimator operations to be tested.
trait TestEstimator<D: Dim>:
    KalmanEstimator<f64, D> + LinearPredictor<f64, D, U1>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
{
    fn trace_state(&self) {}

    /// Observation with uncorrelected noise
    fn observe_innov_un(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveNoise<f64, U1>,
        s: &Vector1<f64>,
        x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str>;

    /// Observation with correlected noise
    fn observe_linear_co(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1, U1>,
        s: &Vector1<f64>,
        _x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str>;
}

/// Test covariance estimator operations defined on a KalmanState.
impl<D: Dim> TestEstimator<D> for KalmanState<f64, D>
where
    Self: LinearObserverCorrelated<f64, D, U1, U1>,
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
{
    fn observe_innov_un(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveNoise<f64, U1>,
        s: &Vector1<f64>,
        _x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str>
    where
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        LinearObserverUncorrelated::observe_innovation(self, obs, noise, &s)
    }

    fn observe_linear_co(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1, U1>,
        s: &Vector1<f64>,
        _x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str>
    where
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        LinearObserverCorrelated::observe_innovation(self, obs, noise, s)
    }
}

/// Test information estimator operations defined on a InformationState.
impl<D: Dim> TestEstimator<D> for InformationState<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
{
    fn observe_innov_un(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveNoise<f64, U1>,
        s: &Vector1<f64>,
        x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str>
    where
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        let info = self.observe_innovation_un(&obs, noise, s, x)?;
        self.add_information(&info.1);
        Result::Ok(info.0)
    }

    fn observe_linear_co(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1, U1>,
        s: &Vector1<f64>,
        x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str>
    where
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        let info = self.observe_innovation_co(obs, noise, s, x)?;
        self.add_information(&info.1);
        Result::Ok(info.0)
    }
}

/// Test UD estimator operations defined on a UDState.
impl<D: DimAdd<U1>> TestEstimator<D> for UDState<f64, D, DimSum<D, U1>>
where
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, D>
        + Allocator<f64, D, DimSum<D, U1>>
        + Allocator<f64, DimSum<D, U1>>
        + Allocator<usize, D, DimSum<D, U1>>,
{
    fn trace_state(&self) {
        println!("{}", self.UD);
    }

    fn observe_innov_un(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveNoise<f64, U1>,
        s: &Vector1<f64>,
        _x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str>
    where
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        LinearObserverUncorrelated::observe_innovation(self, obs, noise, s)
    }

    fn observe_linear_co(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1, U1>,
        s: &Vector1<f64>,
        x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str> {
        let z = s + hx(x);

        self.observe_decorrelate::<U1, U1>(obs, noise, &z)
    }
}

/// Test information_root estimator operations defined on a InformationRootState.
impl<D: Dim> TestEstimator<D> for InformationRootState<f64, D>
    where
        DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
{
    fn observe_innov_un(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveNoise<f64, U1>,
        s: &Vector1<f64>,
        x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str>
        where
            DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        let info = self.observe_innovation_un(&obs, noise, s, x)?;
        Result::Ok(info.0)
    }

    fn observe_linear_co(
        &mut self,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1, U1>,
        s: &Vector1<f64>,
        x: &VectorN<f64, D>,
    ) -> Result<f64, &'static str>
        where
            DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        let info = self.observe_innovation_co(obs, noise, s, x)?;
        self.add_information(&info.1);
        Result::Ok(info.0)
    }
}


/// Simple prediction model.
fn fx<D: Dim>(x: &VectorN<f64, D>) -> VectorN<f64, D>
where
    DefaultAllocator: Allocator<f64, D>,
{
    let mut xp = (*x).clone();
    xp[0] += DT * x[1];
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
fn test_estimator<D: Dim>(est: &mut dyn TestEstimator<D>)
where
    ShapeConstraint: SameNumberOfRows<U2, D> + SameNumberOfColumns<U2, D>,
    ShapeConstraint: SameNumberOfRows<D, U2> + SameNumberOfColumns<D, U2>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, D, U1>
        + Allocator<f64, U1>
        + Allocator<f64, U2, U2>
        + Allocator<usize, D, D>
        + Allocator<usize, D>,
{
    let state = est.state().unwrap().1;
    let d = state.x.data.shape().0;

    let f_vv: f64 = (-DT * V_GAMMA).exp();

    let linear_pred_model = LinearPredictModel {
        Fx: new_copy(d, d, &Matrix2::new(1., DT, 0., f_vv)),
    };
    let additive_noise = AdditiveCorrelatedNoise {
        q: Vector1::new(DT * sqr((1. - f_vv) * V_NOISE)),
        G: new_copy(d, U1, &Matrix2x1::new(0.0, 1.0)),
    };

    let linear_obs_model = LinearObserveModel {
        Hx: new_copy(U1, d, &Matrix1x2::new(1.0, 0.0)),
    };
    let un_obs_noise = AdditiveNoise {
        q: Vector1::new(sqr(OBS_NOISE)),
    };
    let co_obs_noise = AdditiveCorrelatedNoise {
        G: Matrix1::new(1.0),
        q: Vector1::new(sqr(OBS_NOISE)),
    };
    let z = &Vector1::new(1000.);

    let init_state: KalmanState<f64, D> = KalmanState {
        x: new_copy(d, U1, &Vector2::new(900., 1.5)),
        X: new_copy(d, d, &Matrix2::new(sqr(I_P_NOISE), 0.0, 0.0, sqr(I_V_NOISE))),
    };

    check(est.init(&init_state), "init").unwrap();
    est.trace_state();

    let xx = est.state().unwrap().1;
    println!("init={:.6}{:.6}", xx.x, xx.X);

    for _c in 0..2 {
        let predict_x = est.state().unwrap().1.x;
        let predict_xp = fx(&predict_x);
        check(
            est.predict(&linear_pred_model, predict_xp, &additive_noise),
            "pred",
        )
        .unwrap();

        let obs_x = est.state().unwrap().1.x;
        let s = z - hx(&obs_x);
        check(
            est.observe_innov_un(&linear_obs_model, &un_obs_noise, &s, &obs_x),
            "obs",
        )
        .unwrap();
    }

    let obs_x = est.state().unwrap().1.x;
    let s = z - hx(&obs_x);
    check(
        est.observe_linear_co(&linear_obs_model, &co_obs_noise, &s, &obs_x),
        "obs_linear",
    )
    .unwrap();
    let xx = est.state().unwrap().1;
    println!("final={:.6}{:.6}", xx.x, xx.X);

    expect_state(&KalmanState::<f64, D> { x: xx.x, X: xx.X });
}

/// Test the KalmanState is as expected.
fn expect_state<D : Dim>(state: &KalmanState<f64, D>)
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
{
    let expect_x = Vector2::new(1000., 0.0152);
    approx::assert_relative_eq!(state.x[0], expect_x[0], max_relative = 0.00000001);
    approx::assert_relative_eq!(state.x[1], expect_x[1], max_relative = 0.01);

    approx::assert_abs_diff_eq!(state.X[(0,0)], 0.000000, epsilon = 0.000001);
    approx::assert_abs_diff_eq!(state.X[(0,1)], 0.000049, epsilon = 0.000001);
    approx::assert_abs_diff_eq!(state.X[(1,1)], 0.014701, epsilon = 0.000003);
}

/// Create a Dynamic or Static copy.
fn new_copy<N: RealField, R: Dim, C: Dim, R1: Dim, C1: Dim, S1: Storage<N, R1, C1>>(
    r: R,
    c: C,
    m: &Matrix<N, R1, C1, S1>,
) -> MatrixMN<N, R, C>
where
    DefaultAllocator: Allocator<N, R, C>,
    ShapeConstraint: SameNumberOfRows<R, R1> + SameNumberOfColumns<C, C1>,
{
    unsafe {
        let mut copy = MatrixMN::<N, R, C>::new_uninitialized_generic(r, c);
        copy.copy_from(m);
        copy
    }
}
