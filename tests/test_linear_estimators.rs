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
    InformationState, KalmanState, LinearObserveModel, LinearPredictModel, KalmanEstimator, LinearObserverCorrelated,
};

use approx;
use bayes_filter::models::{AdditiveCorrelatedNoise, AdditiveCoupledNoise, Estimator, LinearPredictor, FunctionPredictor, FunctionObserverCorrelated};
use bayes_filter::estimators::unscented::UnscentedKallmanState;

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
const LIMIT_PD: f64 = f64::EPSILON * 1e5;


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
fn test_unscented_dynamic() {
    test_estimator(&mut UnscentedKallmanState::new(Dynamic::new(2)));
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
    Estimator<f64, D> + KalmanEstimator<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
{
    fn dim(&self) -> D {
        return Estimator::state(self).unwrap().data.shape().0;
    }

    fn trace_state(&self) {}

    /// Prediction with additive noise
    fn predict_fn(
        &mut self,
        f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        pred: &LinearPredictModel<f64, D>,
        x_pred: VectorN<f64, D>,
        noise: &AdditiveCoupledNoise<f64, D, U1>)
        where
            DefaultAllocator: Allocator<f64, D, U1> + Allocator<f64, U1>;

    /// Observation with correlected noise
    fn observe(
        &mut self,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1>,
        s: &Vector1<f64>,
        _x: &VectorN<f64, D>,
    ) -> Result<(), &'static str>;
}

/// Test covariance estimator operations defined on a KalmanState.
impl<D: Dim> TestEstimator<D> for KalmanState<f64, D>
where
    Self: LinearObserverCorrelated<f64, D, U1>,
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D> + Allocator<f64, U1, D>,
{
    fn predict_fn(
        &mut self,
        _f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        pred: &LinearPredictModel<f64, D>,
        x_pred: VectorN<f64, D>,
        noise: &AdditiveCoupledNoise<f64, D, U1>)
    {
        LinearPredictor::<f64, D>::predict(self, pred, x_pred, &AdditiveCorrelatedNoise::from_coupled::<U1>(noise)).unwrap();
    }

    fn observe(
        &mut self,
        _h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1>,
        s: &Vector1<f64>,
        _x: &VectorN<f64, D>,
    ) -> Result<(), &'static str>
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
    fn dim(&self) -> D {
        return self.i.data.shape().0;
    }

    fn predict_fn(
        &mut self,
        _f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        pred: &LinearPredictModel<f64, D>,
        x_pred: VectorN<f64, D>,
        noise: &AdditiveCoupledNoise<f64, D, U1>)
    {
        LinearPredictor::<f64, D>::predict(self, pred, x_pred, &AdditiveCorrelatedNoise::from_coupled::<U1>(noise)).unwrap();
    }

    fn observe(
        &mut self,
        _h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1>,
        s: &Vector1<f64>,
        x: &VectorN<f64, D>,
    ) -> Result<(), &'static str>
    where
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        let info = self.observe_innovation_co(obs, noise, s, x)?;
        self.add_information(&info.1);
        Result::Ok(())
    }
}

/// Test UD estimator operations defined on a UDState.
impl<D: DimAdd<U1>> TestEstimator<D> for UDState<f64, D, DimSum<D, U1>>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D> + Allocator<f64, D, DimSum<D, U1>> + Allocator<f64, DimSum<D, U1>> + Allocator<usize, D, DimSum<D, U1>>,
{
    fn trace_state(&self) {
        println!("{}", self.UD);
    }

    fn predict_fn(
        &mut self,
        _f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        pred: &LinearPredictModel<f64, D>,
        x_pred: VectorN<f64, D>,
        noise: &AdditiveCoupledNoise<f64, D, U1>)
    {
        self.predict::<U1>(pred, x_pred, noise).unwrap();
    }

    fn observe(
        &mut self,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1>,
        s: &Vector1<f64>,
        x: &VectorN<f64, D>,
    ) -> Result<(), &'static str> {
        let z = s + h(x);

        self.observe_decorrelate::<U1>(obs, noise, &z).map(|_rcond| {})
    }
}

/// Test UD estimator operations defined on a UDState.
impl<D: DimAdd<U1>> TestEstimator<D> for UnscentedKallmanState<f64, D>
    where
        DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D> + Allocator<f64, D, Dynamic>
        + Allocator<usize, D, Dynamic> + Allocator<usize, D, DimSum<D, U1>>,
{
    fn trace_state(&self) {
        println!("{}", self.UU);
    }

    fn predict_fn(
        &mut self,
        f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        _pred: &LinearPredictModel<f64, D>,
        _x_pred: VectorN<f64, D>,
        noise: &AdditiveCoupledNoise<f64, D, U1>)
    {
        FunctionPredictor::predict(self, f, &AdditiveCorrelatedNoise::from_coupled::<U1>(noise)).unwrap();
    }

    fn observe(
        &mut self,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        _obs: &LinearObserveModel<f64, D, U1>,
        noise: &AdditiveCorrelatedNoise<f64, U1>,
        s: &Vector1<f64>,
        _x: &VectorN<f64, D>,
    ) -> Result<(), &'static str> {
        FunctionObserverCorrelated::<f64, D, U1>::observe_innovation(self, h, noise, s)
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
    let d = est.dim();

    let f_vv: f64 = (-DT * V_GAMMA).exp();

    let linear_pred_model = LinearPredictModel {
        Fx: new_copy(d, d, &Matrix2::new(1., DT, 0., f_vv)),
    };
    let additive_noise = AdditiveCoupledNoise {
        q: Vector1::new(DT * sqr((1. - f_vv) * V_NOISE)),
        G: new_copy(d, U1, &Matrix2x1::new(0.0, 1.0)),
    };

    let linear_obs_model = LinearObserveModel {
        Hx: new_copy(U1, d, &Matrix1x2::new(1.0, 0.0)),
    };
    let co_obs_noise = AdditiveCorrelatedNoise {
        Q: Matrix1::new(sqr(OBS_NOISE)),
    };
    let z = &Vector1::new(1000.);

    let init_state: KalmanState<f64, D> = KalmanState {
        x: new_copy(d, U1, &Vector2::new(900., 1.5)),
        X: new_copy(d, d, &Matrix2::new(sqr(I_P_NOISE), 0.0, 0.0, sqr(I_V_NOISE))),
    };

    check(est.init(&init_state), "init").unwrap();
    est.trace_state();

    let xx = KalmanEstimator::state(est).unwrap().1;
    println!("init={:.6}{:.6}", xx.x, xx.X);

    for _c in 0..2 {
        let predict_x = Estimator::state(est).unwrap();
        let predict_xp = fx(&predict_x);
        est.predict_fn(fx, &linear_pred_model, predict_xp, &additive_noise);
        let pp = KalmanEstimator::state(est).unwrap().1;
        println!("pred={:.6}{:.6}", pp.x, pp.X);

        let obs_x = Estimator::state(est).unwrap();
        let s = z - hx(&obs_x);
        est.observe(hx, &linear_obs_model, &co_obs_noise, &s, &obs_x).unwrap();

        let oo = KalmanEstimator::state(est).unwrap().1;
        println!("obs={:.6}{:.6}", oo.x, oo.X);
    }

    let obs_x = Estimator::state(est).unwrap();
    let s = z - hx(&obs_x);
    est.observe(hx, &linear_obs_model, &co_obs_noise, &s, &obs_x).unwrap();

    let xx = KalmanEstimator::state(est).unwrap().1;
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
