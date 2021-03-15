//! Provide FatEstimator trait to allow for testing of many estimators with disparate requirements.
//!
//! This defines predict and observe operations using linearised models to be tested.
//!
//! Where necessary 'fat' estimator states are defined so that different operations of an estimator can be tested using
//! the FatEstimator trait.
//!
//! Implementation are provides for all the linearised estimators.
//! The [`sir`] sampled estimator is implemented so it can be tested using linearise models.

use na::{allocator::Allocator, DefaultAllocator, U1};
use na::{Dim, DimAdd, DimSum, VectorN};
use na::{MatrixMN, RealField};
use na::{DimMin, DimMinimum, MatrixN};
use na::base::storage::Storage;
use nalgebra as na;
use nalgebra::Vector1;
use rand::RngCore;

use bayes_estimate::estimators::information_root::InformationRootState;
use bayes_estimate::estimators::sir;
use bayes_estimate::estimators::ud::{CorrelatedFactorNoise, UDState};
use bayes_estimate::models::{
    Estimator, ExtendedLinearObserver, ExtendedLinearPredictor,
    InformationState, KalmanEstimator, KalmanState
};
use bayes_estimate::noise::{CorrelatedNoise, CoupledNoise, UncorrelatedNoise};

/// Define the estimator operations to be tested.
pub trait FatEstimator<D: Dim>: Estimator<f64, D> + KalmanEstimator<f64, D>
    where DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
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
impl<D: Dim> FatEstimator<D> for KalmanState<f64, D>
    where
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
impl<D: Dim> FatEstimator<D> for InformationState<f64, D>
    where DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
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
impl<D: Dim> FatEstimator<D> for InformationRootState<f64, D>
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
        InformationRootState::predict(self, x_pred, fx, &noise).unwrap();
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

/// Test UD estimator operations defined on a FatUDState.
pub struct FatUDState<N: RealField, D: Dim>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub ud: UDState<N, D>,
    pub obs_uncorrelated: bool,
}

impl<N: RealField, D: Dim> Estimator<N, D> for FatUDState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        self.ud.state()
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for FatUDState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        self.ud.init(state)
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        self.ud.kalman_state()
    }
}

impl<D: DimAdd<U1>> FatEstimator<D> for FatUDState<f64, D>
    where DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>
    + Allocator<f64, D, DimSum<D, U1>> + Allocator<f64, DimSum<D, U1>> + Allocator<usize, D, D>,
{
    fn trace_state(&self) {
        println!("{}", self.ud.UD);
    }

    fn predict_fn(
        &mut self,
        x_pred: &VectorN<f64, D>,
        _f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, U1>)
    {
        self.ud.predict::<U1>(fx, x_pred, noise).unwrap();
    }

    fn observe(
        &mut self,
        z: &Vector1<f64>,
        h: fn(&VectorN<f64, D>) -> VectorN<f64, U1>,
        hx: &MatrixMN<f64, U1, D>,
        noise: &CorrelatedNoise<f64, U1>,
    ) -> Result<(), &'static str> {
        if self.obs_uncorrelated {
            let s = &(z - h(&self.state().unwrap()));
            let noise_single = UncorrelatedNoise::<f64, U1> { q: Vector1::new(noise.Q[0]) };
            self.ud.observe_innovation::<U1>(s, hx, &noise_single).map(|_rcond| {})
        } else {
            let noise_fac = CorrelatedFactorNoise::from_correlated(&noise)?;
            let h_normalize = |_h: &mut VectorN<f64, U1>, _h0: &VectorN<f64, U1>| {};
            self.ud.observe_linear_correlated::<U1>(z, hx, h_normalize, &noise_fac).map(|_rcond| {})
        }
    }
}

/// Test Unscented estimator operations defined on a FatUnscentedState.
pub struct FatUnscentedState<N: RealField, D: Dim>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub kalman: KalmanState<N, D>,
    pub kappa: N,
}

impl<N: RealField, D: Dim> FatUnscentedState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub fn new_zero(d: D) -> FatUnscentedState<N, D> {
        FatUnscentedState {
            kalman: KalmanState::new_zero(d),
            kappa: N::from_usize(3 - d.value()).unwrap(),
        }
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for FatUnscentedState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        return Ok(self.kalman.x.clone());
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for FatUnscentedState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        self.kalman.x.copy_from(&state.x);
        self.kalman.X.copy_from(&state.X);

        Ok(())
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        return Ok(self.kalman.clone());
    }
}

impl<D: Dim> FatEstimator<D> for FatUnscentedState<f64, D>
    where DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D> + Allocator<f64, U1, D>,
        DefaultAllocator: Allocator<usize, D, D>,
        DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>, DefaultAllocator: Allocator<f64, D, U1>
{
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

/// Test SIR estimator operations defined on a FatSampleState.
pub struct FatSampleState<N: RealField, D: Dim>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub sample: sir::SampleState<N, D>,
    pub systematic_resampler: bool,
    pub kalman_roughening: bool
}

impl<N: RealField, D: Dim> Estimator<N, D> for FatSampleState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        self.sample.state()
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for FatSampleState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, U1, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        self.sample.init(state)
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        self.sample.kalman_state()
    }
}

impl<D: Dim> FatEstimator<D> for FatSampleState<f64, D>
    where
        DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, U1, D> + Allocator<f64, D>,
{
    fn trace_state(&self) {
        // println!("{:?}\n{:?}", self.w, self.s);
    }

    fn allow_error_by(&self) -> f64 {
        100000. / (self.sample.s.len() as f64).sqrt()   // sample error scales with sqrt number samples
    }

    fn dim(&self) -> D {
        return self.sample.s[0].data.shape().0;
    }

    fn predict_fn(
        &mut self,
        _x_pred: &VectorN<f64, D>,
        f: fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        _fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, U1>)
    {
        // Predict amd sample the noise
        let coupled_with_q_one = CoupledNoise::from_correlated(&CorrelatedNoise::from_coupled::<U1>(noise)).unwrap();
        let sampler = sir::noise_sampler_coupled(coupled_with_q_one.G);
        self.sample.predict_sampled(move |x: &VectorN<f64, D>, rng: &mut dyn RngCore| -> VectorN<f64, D> {
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
            (-0.5 * (logl + logdetz)).exp()
        };
        self.sample.observe(z_likelihood);

        let mut resampler = if self.systematic_resampler {
            |w: &mut sir::Likelihoods, rng: &mut dyn RngCore| {
                sir::standard_resampler(w, rng)
            }
        }
        else {
            |w: &mut sir::Likelihoods, rng: &mut dyn RngCore| {
                sir::systematic_resampler(w, rng)
            }
        };

        if self.kalman_roughening {
            let noise = CoupledNoise::from_correlated(&CorrelatedNoise {
                Q: self.kalman_state().unwrap().X
            }).unwrap();
            let mut roughener = move |s: &mut sir::Samples<f64, D>, rng: &mut dyn RngCore| {
                sir::roughen_noise(s, &noise, 1., rng)
            };
            self.sample.update_resample(&mut resampler, &mut roughener)?;
        }
        else {
            let mut roughener = |s: &mut sir::Samples<f64, D>, rng: &mut dyn RngCore| {
                sir::roughen_minmax(s, 1., rng)
            };
            self.sample.update_resample(&mut resampler, &mut roughener)?;
        };

        Ok(())
    }
}
