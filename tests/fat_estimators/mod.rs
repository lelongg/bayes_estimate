//! Provide FatEstimator trait to allow for testing of many estimators with disparate requirements.
//!
//! This defines predict and observe operations using linearised models to be tested.
//!
//! Where necessary 'fat' estimator states are defined so that different operations of an estimator can be tested using
//! the FatEstimator trait.
//!
//! Implementation are provides for all the linearised estimators.
//! The [`sir`] sampled estimator is implemented so it can be tested using linearise models.

use na::base::storage::Storage;
use na::{allocator::Allocator, DefaultAllocator, U1};
use na::{Dim, DimAdd, DimSum, VectorN};
use na::{DimMin, DimMinimum, MatrixN};
use na::{MatrixMN, RealField};
use nalgebra as na;
use rand::RngCore;

use bayes_estimate::estimators::information_root::InformationRootState;
use bayes_estimate::estimators::sir;
use bayes_estimate::estimators::ud::{CorrelatedFactorNoise, UdState};
use bayes_estimate::models::{
    Estimator, ExtendedLinearObserver, ExtendedLinearPredictor, InformationState, KalmanEstimator,
    KalmanState,
};
use bayes_estimate::noise::{CorrelatedNoise, CoupledNoise, UncorrelatedNoise};
use nalgebra::iter::MatrixIter;
use nalgebra::{Matrix, Scalar};
use std::iter::StepBy;

/// Define the estimator operations to be tested.
pub trait FatEstimator<D: Dim, QD: Dim, ZD: Dim>:
    Estimator<f64, D> + KalmanEstimator<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, QD, QD>
        + Allocator<f64, D, QD>
        + Allocator<f64, QD>
        + Allocator<f64, ZD, ZD>
        + Allocator<f64, ZD, D>
        + Allocator<f64, ZD>,
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
        f: impl Fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, QD>,
    ) where
        DefaultAllocator: Allocator<f64, D, U1> + Allocator<f64, U1>;

    /// Observation with correlected noise
    fn observe(
        &mut self,
        z: &VectorN<f64, ZD>,
        h: impl Fn(&VectorN<f64, D>) -> VectorN<f64, ZD>,
        h_normalize: impl Fn(&mut VectorN<f64, ZD>, &VectorN<f64, ZD>),
        hx: &MatrixMN<f64, ZD, D>,
        noise: &CorrelatedNoise<f64, ZD>,
    ) -> Result<(), &'static str>;
}

/// Test covariance estimator operations defined on a KalmanState.
impl<D: Dim, QD: Dim, ZD: Dim> FatEstimator<D, QD, ZD> for KalmanState<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, QD, QD>
        + Allocator<f64, D, QD>
        + Allocator<f64, QD>
        + Allocator<f64, ZD, ZD>
        + Allocator<f64, ZD, D>
        + Allocator<f64, ZD>,
    // observe_innovation
    DefaultAllocator: Allocator<f64, D, ZD>,
{
    fn predict_fn(
        &mut self,
        x_pred: &VectorN<f64, D>,
        _f: impl Fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, QD>,
    ) {
        ExtendedLinearPredictor::<f64, D>::predict(
            self,
            x_pred,
            fx,
            &CorrelatedNoise::from_coupled::<QD>(noise),
        )
        .unwrap();
    }

    fn observe(
        &mut self,
        z: &VectorN<f64, ZD>,
        h: impl Fn(&VectorN<f64, D>) -> VectorN<f64, ZD>,
        _h_normalize: impl Fn(&mut VectorN<f64, ZD>, &VectorN<f64, ZD>),
        hx: &MatrixMN<f64, ZD, D>,
        noise: &CorrelatedNoise<f64, ZD>,
    ) -> Result<(), &'static str> {
        ExtendedLinearObserver::observe_innovation(self, &(z - h(&self.x)), hx, noise)
    }
}

/// Test information estimator operations defined on a InformationState.
impl<D: Dim, QD: Dim, ZD: Dim> FatEstimator<D, QD, ZD> for InformationState<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, QD, QD>
        + Allocator<f64, D, QD>
        + Allocator<f64, QD>
        + Allocator<f64, ZD, ZD>
        + Allocator<f64, ZD, D>
        + Allocator<f64, ZD>,
    // observe_innovation
    DefaultAllocator: Allocator<f64, D, ZD>,
{
    fn dim(&self) -> D {
        self.i.data.shape().0
    }

    fn predict_fn(
        &mut self,
        x_pred: &VectorN<f64, D>,
        _f: impl Fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, QD>,
    ) {
        ExtendedLinearPredictor::<f64, D>::predict(
            self,
            x_pred,
            fx,
            &CorrelatedNoise::from_coupled::<QD>(noise),
        )
        .unwrap();
    }

    fn observe(
        &mut self,
        z: &VectorN<f64, ZD>,
        h: impl Fn(&VectorN<f64, D>) -> VectorN<f64, ZD>,
        _h_normalize: impl Fn(&mut VectorN<f64, ZD>, &VectorN<f64, ZD>),
        hx: &MatrixMN<f64, ZD, D>,
        noise: &CorrelatedNoise<f64, ZD>,
    ) -> Result<(), &'static str> {
        let s = z - h(&self.state().unwrap());
        ExtendedLinearObserver::observe_innovation(self, &s, hx, noise)?;
        Ok(())
    }
}

/// Test information estimator operations defined on a InformationRootState.
impl<D: Dim, QD: Dim, ZD: Dim> FatEstimator<D, QD, ZD> for InformationRootState<f64, D>
where
    // display
    DefaultAllocator: Allocator<usize, D, D>,

    // InformationRootState
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,

    // FatEstimator
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, ZD, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, D>
        + Allocator<f64, QD, QD>
        + Allocator<f64, ZD, ZD>
        + Allocator<f64, D, QD>
        + Allocator<f64, QD>
        + Allocator<f64, ZD>,

    // predict
    D: DimAdd<QD>,
    DefaultAllocator: Allocator<f64, DimSum<D, QD>, DimSum<D, QD>>
        + Allocator<f64, DimSum<D, QD>>
        + Allocator<f64, D, QD>
        + Allocator<f64, QD>,
    DimSum<D, QD>: DimMin<DimSum<D, QD>>,
    DefaultAllocator: Allocator<f64, DimMinimum<DimSum<D, QD>, DimSum<D, QD>>>
        + Allocator<f64, DimMinimum<DimSum<D, QD>, DimSum<D, QD>>, DimSum<D, QD>>,
    // observe
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, ZD, D>
        + Allocator<f64, ZD, ZD>
        + Allocator<f64, D>
        + Allocator<f64, ZD>,
    D: DimAdd<ZD> + DimAdd<U1>,
    DefaultAllocator: Allocator<f64, DimSum<D, ZD>, DimSum<D, U1>> + Allocator<f64, DimSum<D, ZD>>,
    DimSum<D, ZD>: DimMin<DimSum<D, U1>>,
    DefaultAllocator: Allocator<f64, DimMinimum<DimSum<D, ZD>, DimSum<D, U1>>>
        + Allocator<f64, DimMinimum<DimSum<D, ZD>, DimSum<D, U1>>, DimSum<D, U1>>,
{
    fn trace_state(&self) {
        println!("{}", self.R);
    }

    fn dim(&self) -> D {
        self.r.data.shape().0
    }

    fn predict_fn(
        &mut self,
        x_pred: &VectorN<f64, D>,
        _f: impl Fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, QD>,
    ) {
        InformationRootState::predict::<QD>(self, x_pred, fx, &noise).unwrap();
    }

    fn observe(
        &mut self,
        z: &VectorN<f64, ZD>,
        h: impl Fn(&VectorN<f64, D>) -> VectorN<f64, ZD>,
        _h_normalize: impl Fn(&mut VectorN<f64, ZD>, &VectorN<f64, ZD>),
        hx: &MatrixMN<f64, ZD, D>,
        noise: &CorrelatedNoise<f64, ZD>,
    ) -> Result<(), &'static str>
    where
        DefaultAllocator: Allocator<f64, D, D>
            + Allocator<f64, U1, U1>
            + Allocator<f64, U1, D>
            + Allocator<f64, D, U1>
            + Allocator<f64, D>
            + Allocator<f64, U1>,
    {
        let s = &(z - h(&self.state().unwrap()));
        ExtendedLinearObserver::observe_innovation(self, s, hx, noise)?;

        Ok(())
    }
}

/// Test UD estimator operations defined on a FatUDState.
pub struct FatUdState<N: RealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub ud: UdState<N, D>,
    pub obs_uncorrelated: bool,
}

impl<N: RealField, D: Dim> Estimator<N, D> for FatUdState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        self.ud.state()
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for FatUdState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        self.ud.init(state)
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        self.ud.kalman_state()
    }
}

impl<D: DimAdd<U1>, QD: Dim, ZD: Dim> FatEstimator<D, QD, ZD> for FatUdState<f64, D>
where
    DefaultAllocator: Allocator<usize, D, D>
        + Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, QD, QD>
        + Allocator<f64, D, QD>
        + Allocator<f64, QD>
        + Allocator<f64, ZD, ZD>
        + Allocator<f64, ZD, D>
        + Allocator<f64, ZD>
        + Allocator<f64, D, DimSum<D, U1>>,
    D: DimAdd<QD>,
    DefaultAllocator: Allocator<f64, DimSum<D, QD>, U1>,
{
    fn trace_state(&self) {
        println!("{}", self.ud.UD);
    }

    fn predict_fn(
        &mut self,
        x_pred: &VectorN<f64, D>,
        _f: impl Fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, QD>,
    ) {
        self.ud.predict::<QD>(fx, x_pred, noise).unwrap();
    }

    fn observe(
        &mut self,
        z: &VectorN<f64, ZD>,
        h: impl Fn(&VectorN<f64, D>) -> VectorN<f64, ZD>,
        _h_normalize: impl Fn(&mut VectorN<f64, ZD>, &VectorN<f64, ZD>),
        hx: &MatrixMN<f64, ZD, D>,
        noise: &CorrelatedNoise<f64, ZD>,
    ) -> Result<(), &'static str> {
        if self.obs_uncorrelated {
            let s = &(z - h(&self.state().unwrap()));
            let noise_single = UncorrelatedNoise::<f64, ZD> {
                q: noise.Q.diagonal(),
            };
            self.ud
                .observe_innovation::<ZD>(s, hx, &noise_single)
                .map(|_rcond| {})
        } else {
            let noise_fac = CorrelatedFactorNoise::from_correlated(&noise)?;
            let h_normalize = |_h: &mut VectorN<f64, ZD>, _h0: &VectorN<f64, ZD>| {};
            self.ud
                .observe_linear_correlated(z, hx, h_normalize, &noise_fac)
                .map(|_rcond| {})
        }
    }
}

/// Test Unscented estimator operations defined on a FatUnscentedState.
pub struct FatUnscentedState<N: RealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub kalman: KalmanState<N, D>,
    pub kappa: N,
}

impl<N: RealField, D: Dim> FatUnscentedState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub fn new_zero(d: D) -> FatUnscentedState<N, D> {
        FatUnscentedState {
            kalman: KalmanState::new_zero(d),
            kappa: N::from_usize(3 - d.value()).unwrap(),
        }
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for FatUnscentedState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        Ok(self.kalman.x.clone())
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for FatUnscentedState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        self.kalman.x.copy_from(&state.x);
        self.kalman.X.copy_from(&state.X);

        Ok(())
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        Ok(self.kalman.clone())
    }
}

impl<D: Dim, QD: Dim, ZD: Dim> FatEstimator<D, QD, ZD> for FatUnscentedState<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, QD, QD>
        + Allocator<f64, D, QD>
        + Allocator<f64, QD>
        + Allocator<f64, ZD, ZD>
        + Allocator<f64, ZD, D>
        + Allocator<f64, ZD>,
    // predict_unscented
    DefaultAllocator: Allocator<f64, U1, D>,
    // observe_unscented
    DefaultAllocator: Allocator<f64, D, ZD>,
    DefaultAllocator: Allocator<f64, U1, ZD>,
    //
    DefaultAllocator: Allocator<f64, U1, U1> + Allocator<f64, U1>,
    DefaultAllocator: Allocator<f64, D, U1>,
    DefaultAllocator: Allocator<usize, D, D>,
{
    fn trace_state(&self) {
        println!("{:}", self.kalman.X);
    }

    fn predict_fn(
        &mut self,
        _x_pred: &VectorN<f64, D>,
        f: impl Fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        _fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, QD>,
    ) {
        self.kalman
            .predict_unscented(f, &CorrelatedNoise::from_coupled::<QD>(noise), self.kappa)
            .unwrap();
    }

    fn observe(
        &mut self,
        z: &VectorN<f64, ZD>,
        h: impl Fn(&VectorN<f64, D>) -> VectorN<f64, ZD>,
        h_normalize: impl Fn(&mut VectorN<f64, ZD>, &VectorN<f64, ZD>),
        _hx: &MatrixMN<f64, ZD, D>,
        noise: &CorrelatedNoise<f64, ZD>,
    ) -> Result<(), &'static str> {
        let s = &(z - h(&self.state().unwrap()));
        self.kalman
            .observe_unscented(h, h_normalize, noise, s, self.kappa)
    }
}

/// Test SIR estimator operations defined on a FatSampleState.
pub struct FatSampleState<N: RealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub sample: sir::SampleState<N, D>,
    pub systematic_resampler: bool,
    pub kalman_roughening: bool,
}

impl<N: RealField, D: Dim> Estimator<N, D> for FatSampleState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        self.sample.state()
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for FatSampleState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, U1, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        self.sample.init(state)
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        self.sample.kalman_state()
    }
}

impl<D: Dim, QD: Dim, ZD: Dim> FatEstimator<D, QD, ZD> for FatSampleState<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, QD, QD>
        + Allocator<f64, D, QD>
        + Allocator<f64, QD>
        + Allocator<f64, ZD, ZD>
        + Allocator<f64, ZD, D>
        + Allocator<f64, ZD>,
    // sample
    DefaultAllocator: Allocator<f64, U1, D>,
{
    fn trace_state(&self) {
        // println!("{:?}\n{:?}", self.w, self.s);
    }

    fn allow_error_by(&self) -> f64 {
        100000. / (self.sample.s.len() as f64).sqrt() // sample error scales with sqrt number samples
    }

    fn dim(&self) -> D {
        self.sample.s[0].data.shape().0
    }

    fn predict_fn(
        &mut self,
        _x_pred: &VectorN<f64, D>,
        f: impl Fn(&VectorN<f64, D>) -> VectorN<f64, D>,
        _fx: &MatrixN<f64, D>,
        noise: &CoupledNoise<f64, D, QD>,
    ) {
        // Predict amd sample the noise
        let coupled_with_q_one =
            CoupledNoise::from_correlated(&CorrelatedNoise::from_coupled::<QD>(noise)).unwrap();
        let sampler = sir::normal_noise_sampler_coupled(coupled_with_q_one.G);
        self.sample.predict_sampled(
            move |x: &VectorN<f64, D>, rng: &mut dyn RngCore| -> VectorN<f64, D> {
                f(&x) + sampler(rng)
            },
        );
    }

    fn observe(
        &mut self,
        z: &VectorN<f64, ZD>,
        h: impl Fn(&VectorN<f64, D>) -> VectorN<f64, ZD>,
        _h_normalize: impl Fn(&mut VectorN<f64, ZD>, &VectorN<f64, ZD>),
        _hx: &MatrixMN<f64, ZD, D>,
        noise: &CorrelatedNoise<f64, ZD>,
    ) -> Result<(), &'static str> {
        self.sample
            .observe(sir::gaussian_observation_likelihood(z, h, noise));

        let mut resampler = if self.systematic_resampler {
            |w: &mut sir::Likelihoods, rng: &mut dyn RngCore| sir::standard_resampler(w, rng)
        } else {
            |w: &mut sir::Likelihoods, rng: &mut dyn RngCore| sir::systematic_resampler(w, rng)
        };

        if self.kalman_roughening {
            let noise = CoupledNoise::from_correlated(&CorrelatedNoise {
                Q: self.kalman_state().unwrap().X,
            })
            .unwrap();
            let mut roughener = move |s: &mut sir::Samples<f64, D>, rng: &mut dyn RngCore| {
                sir::roughen_noise(s, &noise, 1., rng)
            };
            self.sample
                .update_resample(&mut resampler, &mut roughener)?;
        } else {
            let mut roughener = |s: &mut sir::Samples<f64, D>, rng: &mut dyn RngCore| {
                sir::roughen_minmax(s, 1., rng)
            };
            self.sample
                .update_resample(&mut resampler, &mut roughener)?;
        };

        Ok(())
    }
}

trait Diagonal<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> {
    fn diagonal_iter(&self) -> StepBy<MatrixIter<N, R, C, S>>;
}

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Diagonal<N, R, C, S> for Matrix<N, R, C, S> {
    fn diagonal_iter(&self) -> StepBy<MatrixIter<N, R, C, S>> {
        self.iter().step_by(self.nrows() + 1)
    }
}
