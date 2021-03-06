//! Test the numerical operations of linear and linearised estimator implementations.
//!
//! [`covariance`], [`information`], [`ud`] and [`unscented`] estimator implementations are tested.
//!
//! [`covariance`]: ../filters/covariance.html
//! [`information`]: ../filters/information.html
//! [`ud`]: ../filters/ud.html
//! [`unscented`]: ../filters/unscented.html
//!
//! Tests are performed with Dynamic matrices and matrices with fixed dimensions.

use approx;
use rand::RngCore;

use bayes_estimate::cholesky::UDU;
use bayes_estimate::estimators::information_root::InformationRootState;
use bayes_estimate::estimators::sir::{Likelihoods, SampleState, standard_resampler};
use bayes_estimate::models::{
    Estimator, ExtendedLinearObserver, ExtendedLinearPredictor,
    InformationState, KalmanEstimator, KalmanState, UDState
};
use bayes_estimate::noise::{CorrelatedFactorNoise, CorrelatedNoise, CoupledNoise};
use na::{allocator::Allocator, DefaultAllocator, U1, U2};
use na::{Dim, DimAdd, DimSum, Dynamic, VectorN};
use na::{Matrix, Matrix1, Matrix1x2, Matrix2, Matrix2x1, Vector1, Vector2};
use na::{MatrixMN, RealField};
use na::{DimMin, DimMinimum, DVector, MatrixN};
use na::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use na::base::storage::Storage;
use nalgebra as na;

#[test]
fn test_covariance_u2() {
    test_estimator(&mut KalmanState::new_zero(U2));
}

#[test]
fn test_information_u2() {
    test_estimator(&mut InformationState::new_zero(U2));
}

#[test]
fn test_information_root_u2() {
    test_estimator(&mut InformationRootState::new_zero(U2));
}

#[test]
fn test_ud_u2() {
    test_estimator(&mut UDState::new_zero(U2));
}

#[test]
fn test_unscented_u2() {
    test_estimator(&mut UnscentedKalmanState::new_zero(U2));
}

#[test]
fn test_sir_u2() {
    let mut s = Vec::with_capacity(10000);
    for _i in 0..10000 {
        s.push(VectorN::<f64, U2>::zeros());
    }
    let rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(1u64);
    test_estimator(&mut SampleState::new_equal_likelihood(s, Box::new(rng)));
}


#[test]
fn test_covariance_dynamic() {
    test_estimator(&mut KalmanState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_information_dynamic() {
    test_estimator(&mut InformationState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_information_root_dynamic() {
    test_estimator(&mut InformationRootState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_ud_dynamic() {
    test_estimator(&mut UDState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_unscented_dynamic() {
    test_estimator(&mut UnscentedKalmanState::new_zero(Dynamic::new(2)));
}

#[test]
fn test_sir_dynamic() {
    let mut s = Vec::with_capacity(10000);
    for _i in 0..10000 {
        s.push(DVector::zeros(2));
    }
    let rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(1u64);
    test_estimator(&mut SampleState::new_equal_likelihood(s, Box::new(rng)));
}


fn sqr(x: f64) -> f64 {
    x * x
}

/// Define the estimator operations to be tested.
trait TestEstimator<D: Dim>: Estimator<f64, D> + KalmanEstimator<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
{
    fn dim(&self) -> D {
        return Estimator::state(self).unwrap().data.shape().0;
    }

    fn allow_error_by(&self) -> f64 {
        1f64
    }

    fn trace_state(&self) {}

    /// Prediction with additive noise
    fn predict_fn(
        &mut self,
        x_pred: &VectorN<f64, D>,
        f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, U1>)
        where
            DefaultAllocator: Allocator<f64, D, U1> + Allocator<f64, U1>;

    /// Observation with correlected noise
    fn observe(
        &mut self,
        z: &Vector1<f64>,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        hx: &MatrixMN<f64, U1, D>,
        noise: &CorrelatedNoise<f64, U1>,
    ) -> Result<(), &'static str>;
}

/// Test covariance estimator operations defined on a KalmanState.
impl<D: Dim> TestEstimator<D> for KalmanState<f64, D>
where
    Self: ExtendedLinearObserver<f64, D, U1>,
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D> + Allocator<f64, U1, D>,
{
    fn predict_fn(
        &mut self,
        x_pred: &VectorN<f64, D>,
        _f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, U1>)
    {
        ExtendedLinearPredictor::<f64, D>::predict(self, x_pred, fx, &CorrelatedNoise::from_coupled::<U1>(noise)).unwrap();
    }

    fn observe(
        &mut self,
        z: &Vector1<f64>,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        hx: &MatrixMN<f64, U1, D>,
        noise: &CorrelatedNoise<f64, U1>,
    ) -> Result<(), &'static str>
    where
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        ExtendedLinearObserver::observe_innovation(self, &(z - h(&self.x)), hx, noise)
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
        x_pred: &VectorN<f64, D>,
        _f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, U1>)
    {
        ExtendedLinearPredictor::<f64, D>::predict(self, x_pred, fx, &CorrelatedNoise::from_coupled::<U1>(noise)).unwrap();
    }

    fn observe(
        &mut self,
        z: &Vector1<f64>,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        hx: &MatrixMN<f64, U1, D>,
        noise: &CorrelatedNoise<f64, U1>,
    ) -> Result<(), &'static str>
    where
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    {
        let s = z - h(&self.state().unwrap());
        ExtendedLinearObserver::observe_innovation(self, &s, hx, noise)?;
        Ok(())
    }
}

/// Test information estimator operations defined on a InformationRootState.
impl<D: Dim> TestEstimator<D> for InformationRootState<f64, D>
    where
        DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
        D: DimAdd<U1>,
        DefaultAllocator: Allocator<f64, DimSum<D, U1>, DimSum<D, U1>> + Allocator<f64, DimSum<D, U1>> + Allocator<f64, D, U1> + Allocator<f64, U1>,
        DimSum<D, U1>: DimMin<DimSum<D, U1>>,
        DefaultAllocator: Allocator<f64, DimMinimum<DimSum<D, U1>, DimSum<D, U1>>> + Allocator<f64, DimMinimum<DimSum<D, U1>, DimSum<D, U1>>, DimSum<D, U1>>
        + Allocator<usize, D, D>,
{
    fn trace_state(&self) {
        println!("{}", self.R);
    }

    fn dim(&self) -> D {
        return self.r.data.shape().0;
    }

    fn predict_fn(
        &mut self,
        x_pred: &VectorN<f64, D>,
        _f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, U1>)
    {
        InformationRootState::predict(self,  x_pred, fx, &noise).unwrap();
    }

    fn observe(
        &mut self,
        z: &Vector1<f64>,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        hx: &MatrixMN<f64, U1, D>,
        noise: &CorrelatedNoise<f64, U1>,
    ) -> Result<(), &'static str>
        where
            DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, U1> + Allocator<f64, U1, D> + Allocator<f64, D, U1> + Allocator<f64, D> + Allocator<f64, U1>
    {
        let s = &(z - h(&self.state().unwrap()));
        ExtendedLinearObserver::observe_innovation(self, s, hx, noise)?;

        Ok(())
    }
}

/// Test UD estimator operations defined on a UDState.
impl<D: DimAdd<U1>> TestEstimator<D> for UDState<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D> + Allocator<f64, D, DimSum<D, U1>> + Allocator<f64, DimSum<D, U1>>
    + Allocator<usize, D, D>,
{
    fn trace_state(&self) {
        println!("{}", self.UD);
    }

    fn predict_fn(
        &mut self,
        x_pred: &VectorN<f64, D>,
        _f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, U1>)
    {
        self.predict::<U1>(fx, x_pred, noise).unwrap();
    }

    fn observe(
        &mut self,
        z: &Vector1<f64>,
        _h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        hx: &MatrixMN<f64, U1, D>,
        noise: &CorrelatedNoise<f64, U1>,
    ) -> Result<(), &'static str> {
        let udu = UDU::new();
        let mut ud: MatrixMN<f64,U1,U1> = noise.Q.clone_owned();
        udu.UdUfactor_variant2(&mut ud, z.nrows());

        let noise_fac = CorrelatedFactorNoise::<f64, U1>{ UD: ud };
        let h_normalize = |_h: &mut VectorN<f64, U1>, _h0: &VectorN<f64, U1>| {};
        self.observe_correlated::<U1>(&z, hx, h_normalize, &noise_fac).map(|_rcond| {})
    }
}

/// Test Unscented estimator operations defined on an UnscentedKalmanState.
pub struct UnscentedKalmanState<N:RealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub kalman: KalmanState<N, D>,
    pub kappa: N,
}

impl<N: RealField, D: Dim> UnscentedKalmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub fn new_zero(d: D) -> UnscentedKalmanState<N, D> {
        UnscentedKalmanState {
            kalman: KalmanState::new_zero(d),
            kappa: N::from_usize(3 - d.value()).unwrap(),
        }
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for UnscentedKalmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        return Ok(self.kalman.x.clone());
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for UnscentedKalmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        self.kalman.x.copy_from(&state.x);
        self.kalman.X.copy_from(&state.X);

        Ok(())
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        return Ok(self.kalman.clone())
    }
}

impl<D: Dim> TestEstimator<D> for UnscentedKalmanState<f64, D>
    where
        DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D> + Allocator<f64, U1, D>,
        DefaultAllocator: Allocator<usize, D, D>,
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
        DefaultAllocator: Allocator<f64, D, U1>  {

    fn trace_state(&self) {
        println!("{:}", self.kalman.X);
    }

    fn predict_fn(
        &mut self,
        _x_pred: &VectorN<f64, D>,
        f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        _fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, U1>)
    {
        self.kalman.predict_unscented(f, &CorrelatedNoise::from_coupled::<U1>(noise), self.kappa).unwrap();
    }

    fn observe(
        &mut self,
        z: &Vector1<f64>,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        _hx: &MatrixMN<f64, U1, D>,
        noise: &CorrelatedNoise<f64, U1>,
    ) -> Result<(), &'static str> {
        let h_normalize = |_h: &mut VectorN<f64, U1>, _h0: &VectorN<f64, U1>| {};
        let s = &(z - h(&self.state().unwrap()));
        self.kalman.observe_unscented(h, h_normalize, noise, s, self.kappa)
    }
}

/// Test SIR estimator operations defined on a SampleState.
impl<D: Dim> TestEstimator<D> for SampleState<f64, D>
    where
        DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
{
    fn trace_state(&self) {
        // println!("{:?}\n{:?}", self.w, self.s);
    }

    fn allow_error_by(&self) -> f64 {
     1000f64
    }

    fn dim(&self) -> D {
        return self.s[0].data.shape().0;
    }

    fn predict_fn(
        &mut self,
        _x_pred: &VectorN<f64, D>,
        f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        _fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, U1>)
    {
        self.predict(f);
        let correlated_noise = CorrelatedNoise::from_coupled::<U1>(noise);
        let sampler = correlated_noise.sampler().unwrap();
        self.predict_sampled(move |x: &VectorN<f64,D>, rng: &mut dyn RngCore| -> VectorN<f64,D> {
            f(&x) + sampler(rng)
        });
    }

    fn observe(
        &mut self,
        z: &Vector1<f64>,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        _hx: &MatrixMN<f64, U1, D>,
        noise: &CorrelatedNoise<f64, U1>,
    ) -> Result<(), &'static str>
    {
        let zinv = noise.Q.cholesky().unwrap().inverse();
        let logdetz = noise.Q.determinant().ln() as f32;

        let z_likelihood = |x: &VectorN<f64, D>| -> f32 {
            let innov = z - h(x);
            let logl = innov.dot(&(&zinv * &innov)) as f32;
            (-0.5*(logl + logdetz)).exp()
        };
        self.observe(z_likelihood);

        let mut resampler = |w: &mut Likelihoods, rng: &mut dyn RngCore| {
            standard_resampler(w, rng)
        };
        let mut roughener= |s: &mut Vec<VectorN<f64, D>>, rng: &mut dyn RngCore| {
            SampleState::roughen_minmax(s, 1., rng)
        };
        self.update_resample(&mut resampler, &mut roughener)?;

        Ok(())
    }
}


const DT: f64 = 0.01;
const V_NOISE: f64 = 0.2; // Velocity noise, giving mean squared error bound
const V_GAMMA: f64 = 1.; // Velocity correlation, giving velocity change time constant
// Filter's Initial state uncertainty: System state is unknown
const I_P_NOISE: f64 = 2.;
const I_V_NOISE: f64 = 0.1;
// Noise on observing system state
const OBS_NOISE: f64 = 0.1;

/// Simple prediction model.
fn fx<D: Dim>(x: &VectorN<f64, D>) -> VectorN<f64, D>
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
fn test_estimator<D: Dim>(est: &mut dyn TestEstimator<D>)
where
    ShapeConstraint: SameNumberOfRows<U2, D> + SameNumberOfColumns<U2, D>,
    ShapeConstraint: SameNumberOfRows<D, U2> + SameNumberOfColumns<D, U2>,
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>
        + Allocator<f64, U1, D> + Allocator<f64, D, U1> + Allocator<f64, U1>
        + Allocator<f64, U2, U2>
        + Allocator<usize, D, D> + Allocator<usize, D>,
{
    let d = est.dim();

    let f_vv: f64 = (-DT * V_GAMMA).exp();

    let linear_pred_model = new_copy(d, d, &Matrix2::new(1., DT, 0., f_vv));
    let additive_noise = CoupledNoise {
        q: Vector1::new(DT * sqr((1. - f_vv) * V_NOISE)),
        G: new_copy(d, U1, &Matrix2x1::new(0.0, 1.0)),
    };

    let linear_obs_model = new_copy(U1, d, &Matrix1x2::new(1.0, 0.0));
    let co_obs_noise = CorrelatedNoise {
        Q: Matrix1::new(sqr(OBS_NOISE)),
    };
    let z = &Vector1::new(1000.);

    let init_state: KalmanState<f64, D> = KalmanState {
        x: new_copy(d, U1, &Vector2::new(1000., 1.5)),
        X: new_copy(d, d, &Matrix2::new(sqr(I_P_NOISE), 0.0, 0.0, sqr(I_V_NOISE))),
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
fn expect_state<D : Dim>(state: &KalmanState<f64, D>, allow_by: f64)
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
