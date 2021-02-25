#![allow(non_snake_case)]

//! Bayesian estimation models.
//!
//! Linear models are represented as structs.
//! Common Bayesian discrete system estimation operations are defined as traits.

use nalgebra as na;
use na::{allocator::Allocator, DefaultAllocator, Dim, MatrixMN, MatrixN, VectorN};
use na::{SimdRealField};

pub use crate::noise::{CorrelatedNoise};

pub use crate::estimators::ud::UDState;

/// Kalman state.
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

/// Information state.
///
/// Linear representation as a information state vector and the information (symmetric positive semi-definite) matrix.
/// For a given [KalmanState] the information state I == inverse(X), i == I.x
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
pub trait Estimator<N: SimdRealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D>,
{
    /// The estimator's estimate of the system's state.
    fn state(&self) -> Result<VectorN<N, D>, &'static str>;
}

/// A Kalman estimator.
///
/// The linear Kalman state representation x,X is used to represent the system.
pub trait KalmanEstimator<N: SimdRealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Initialise the estimator with a KalmanState.
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str>;

    /// The estimator's estimate of the system's KalmanState.
    fn kalman_state(&self) -> Result<(N, KalmanState<N, D>), &'static str>;
}

/// An extended linear predictor.
///
/// Uses a non-linear state prediction with linear estimation model with additive noise.
pub trait ExtendedLinearPredictor<N: SimdRealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// Uses a non-linear state prediction with linear estimation model with additive noise.
    fn predict(
        &mut self,
        x_pred: &VectorN<N, D>,
        Fx: &MatrixN<N, D>, // State tramsition matrix
        noise: &CorrelatedNoise<N, D>,
    ) -> Result<(), &'static str>;
}

/// A functional predictor with correlated observation noise.
///
/// Uses a function model with additive noise.
pub trait FunctionPredictor<N: SimdRealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn predict(
        &mut self,
        f: fn(&VectorN<N, D>) -> VectorN<N, D>,
        noise: &CorrelatedNoise<N, D>) -> Result<(), &'static str>;
}

/// A extended linear observer with correlated observation noise.
///
/// Uses a non-linear state observation with linear estimation model with additive noise.
pub trait ExtendedLinearObserver<N: SimdRealField, D: Dim, ZD: Dim>
where
    DefaultAllocator:
        Allocator<N, ZD, D> + Allocator<N, ZD, ZD> + Allocator<N, ZD>
{
    /// Uses a non-linear state observation with linear estimation model with additive noise.
    fn observe_innovation(
        &mut self,
        s: &VectorN<N, ZD>,
        Hx: &MatrixMN<N, ZD, D>, // Observation matrix
        noise: &CorrelatedNoise<N, ZD>,
    ) -> Result<(), &'static str>;
}

/// A functional observer with correlated observation noise.
///
/// Uses a function model with correlated additive observation noise.
pub trait FunctionObserver<N: SimdRealField, D: Dim, ZD: Dim>
    where
        DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, D> + Allocator<N, ZD>
{
    fn observe_innovation(
        &mut self,
        z: &VectorN<N, ZD>,
        h: fn(&VectorN<N, D>) -> VectorN<N, ZD>,
        h_normalize: fn(&mut VectorN<N, D>, VectorN<N, D>),
        noise: &CorrelatedNoise<N, ZD>)
        -> Result<(), &'static str>;
}
