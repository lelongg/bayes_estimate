#![allow(non_snake_case)]

//! Bayesian estimation models.
//!
//! Defines a hierarchy of traits that model discrete systems estimation operations.
//! State representations are defined by structs

use na::{allocator::Allocator, DefaultAllocator, Dim, Dynamic, MatrixMN, MatrixN, VectorN};
use na::SimdRealField;
use na::storage::Storage;
use nalgebra as na;

use crate::mine::matrix::MatrixUDU;

/// Kalman State.
///
/// Linear representation as a state vector and the state covariance (symmetric positive semi-definite) matrix.
#[derive(PartialEq, Clone)]
pub struct KalmanState<N: SimdRealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// State vector
    pub x: VectorN<N, D>,
    /// State covariance matrix (symmetric positive semi-definite)
    pub X: MatrixN<N, D>,
}

/// Information State.
///
/// Linear representation as a information state vector and the information (symmetric positive semi-definite) matrix.
#[derive(PartialEq, Clone)]
pub struct InformationState<N: SimdRealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Information state vector
    pub i: VectorN<N, D>,
    /// Information matrix (symmetric positive semi-definite)
    pub I: MatrixN<N, D>,
}

/// A state estimator.
///
pub trait Estimator<N: SimdRealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D>,
{
    /// The estimator's estimate of the system's state.
    fn state(&self) -> Result<VectorN<N, D>, &'static str>;
}

/// A Kalman filter (estimator).
///
/// The linear Kalman state representation x,X is used to represent the system.
pub trait KalmanEstimator<N: SimdRealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Initialise the estimator with a KalmanState.
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str>;

    /// The estimator's estimate of the system's KalmanState.
    fn state(&self) -> Result<(N, KalmanState<N, D>), &'static str>;
}

/// A linear predictor.
///
/// Uses a Linear model with additive noise.
pub trait LinearPredictor<N: SimdRealField, D: Dim, QD: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, QD> + Allocator<N, D> + Allocator<N, QD>,
{
    /// State prediction with a linear prediction model and additive noise.
    fn predict(
        &mut self,
        pred: &LinearPredictModel<N, D>,
        x_pred: VectorN<N, D>,
        noise: &AdditiveCoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str>;
}

/// A functional predictor.
///
/// Uses a function model with additive noise.
pub trait FunctionPredictor<N: SimdRealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn predict(
        &mut self,
        f: fn(&VectorN<N, D>) -> VectorN<N, D>,
        noise: &AdditiveCorrelatedNoise<N, D>);
}

/// A linear observer with uncorrelated observation noise.
///
/// Uses a Linear observation model with uncorrelated additive observation noise.
pub trait LinearObserverUncorrelated<N: SimdRealField, D: Dim, ZD: Dim>
where
    DefaultAllocator: Allocator<N, ZD, D> + Allocator<N, ZD>
{
    /// State observation with a linear observation model, additive observation noise and
    /// the observation innovation.
    ///
    /// The observation innovation is the difference between the observation and the predicted observation.
    fn observe_innovation(
        &mut self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &AdditiveNoise<N, ZD>,
        s: &VectorN<N, ZD>,
    ) -> Result<N, &'static str>;
}

/// A linear observer with correlated observation noise.
///
/// Uses a Linear observation model with correlated additive observation noise.
pub trait LinearObserverCorrelated<N: SimdRealField, D: Dim, ZD: Dim>
where
    DefaultAllocator:
        Allocator<N, ZD, D> + Allocator<N, ZD, ZD> + Allocator<N, ZD>
{
    fn observe_innovation(
        &mut self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &AdditiveCorrelatedNoise<N, ZD>,
        s: &VectorN<N, ZD>,
    ) -> Result<N, &'static str>;
}

/// A functional observer with correlated observation noise.
///
/// Uses a function model with correlated additive observation noise.
pub trait FunctionObserverCorrelated<N: SimdRealField, D: Dim, ZD: Dim>
    where
        DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, D, Dynamic> + Allocator<N, ZD, Dynamic> + Allocator<N, ZD>
{
    fn observe_innovation(
        &mut self,
        h: fn(&MatrixMN<N, D, Dynamic>) -> MatrixMN<N, ZD, Dynamic>,
        noise: &AdditiveCorrelatedNoise<N, ZD>,
        s: &VectorN<N, ZD>)
        -> Result<N, &'static str>;
}

/// Additive noise.
///
/// Linear additive noise represented as a the noise variance vector.
pub struct AdditiveNoise<N: SimdRealField, QD: Dim>
where
    DefaultAllocator: Allocator<N, QD>
{
    /// Noise variance
    pub q: VectorN<N, QD>,
}

/// Additive noise.
///
/// Linear additive noise represented as a the noise covariance as a factorised UdU' matrix.
pub struct AdditiveCorrelatedNoise<N: SimdRealField, QD: Dim>
    where
        DefaultAllocator: Allocator<N, QD, QD>
{
    /// Noise covariance
    pub Q: MatrixUDU<N, QD>
}

/// Additive noise.
///
/// Linear additive noise represented as a the noise variance vector and a noise coupling matrix.
/// The noise covariance is G.q.G'.
pub struct AdditiveCoupledNoise<N: SimdRealField, D: Dim, QD: Dim>
where
    DefaultAllocator: Allocator<N, D, QD> + Allocator<N, QD>,
{
    /// Noise variance
    pub q: VectorN<N, QD>,
    /// Noise coupling
    pub G: MatrixMN<N, D, QD>,
}

/// Linear prediction model.
///
/// Prediction is represented by a state transition matrix.
pub struct LinearPredictModel<N: SimdRealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    /// State tramsition matrix
    pub Fx: MatrixN<N, D>,
}

/// Linear observation model.
///
/// Observation is represented by an observation matrix.
pub struct LinearObserveModel<N: SimdRealField, D: Dim, ZD: Dim>
where
    DefaultAllocator: Allocator<N, ZD, D>,
{
    /// Observation matrix
    pub Hx: MatrixMN<N, ZD, D>,
}

impl<'a, N: SimdRealField, QD: Dim> AdditiveCorrelatedNoise<N, QD>
where
    DefaultAllocator: Allocator<N, QD, QD> + Allocator<N, QD>
{
    /// Creates a AdditiveCorrelatedNoise from an AdditiveNoise.
    pub fn from_uncorrelated(uncorrelated: &'a AdditiveNoise<N, QD>) -> Self {
        let z_size = uncorrelated.q.data.shape().0;
        let mut correlated = AdditiveCorrelatedNoise {
            Q: MatrixUDU::zeros_generic(z_size, z_size)
        };
        for i in 0..uncorrelated.q.nrows() {
            correlated.Q[(i,i)] = uncorrelated.q[i];
        }

        correlated
    }
}
