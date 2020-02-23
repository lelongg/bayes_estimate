#![allow(non_snake_case)]

/// Bayesian estimation models.
///
/// Defines a hierarchy of traits that model discrete systems estimation operations.
/// State representions are definied by structs

use nalgebra as na;

use na::RealField;
use na::{DefaultAllocator, allocator::Allocator, Dim, MatrixMN, MatrixN, VectorN};


/// Kalman State.
/// Linear respresentation as a state vector and the state covariance (Symetric Positive Definate) matrix.
#[derive(PartialEq, Clone)]
pub struct KalmanState<N: RealField, D: Dim>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// state vector
    pub x: VectorN<N, D>,
    /// state covariance Symetric Positive Definate Matrix
    pub X: MatrixN<N, D>
}

/// Information State.
/// Linear respresentation as a information state vector and the information (Symetric Positive Definate) matrix .
#[derive(PartialEq, Clone)]
pub struct InformationState<N: RealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// information state vector
    pub i: VectorN<N, D>,
    /// Symetric Positive Definate infromation matrix

    pub I: MatrixN<N, D>
}

/// A Kalman filter (estimator).
/// The linear Kalman state representation x,X is used to represent the system.
pub trait KalmanEstimator<N: RealField, D: Dim>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// Initialise the estimator with a KalmanState.
    fn init(&mut self, state : &KalmanState<N, D>) -> Result<N, &'static str>;

    // The estimator estimate of the systems KalmanState.
    fn state(&self) -> Result<(N, KalmanState<N, D>), &'static str>;
}

/// A linear estimator.
///
/// Common to the linear estimators.
pub trait LinearEstimator<N: RealField>
{}

/// A linear predictor.
///
/// Linear or linearised prediction model with additive noise.
pub trait LinearPredictor<N: RealField, D: Dim, QD: Dim> : LinearEstimator<N>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, QD> + Allocator<N, D> + Allocator<N, QD>
{
    fn predict(&mut self, pred: &LinearPredictModel<N, D>, x_pred: VectorN<N, D>, noise: &AdditiveNoise<N, D, QD>) -> Result<N, &'static str>;
}

/// A linear observation.
///
/// Linear or linearised observation model with uncorrelated obseration noise.
pub trait LinearObservationUncorrelated<N: RealField, D: Dim, ZD: Dim> : LinearEstimator<N>
    where DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D> + Allocator<N, ZD>
{
    fn observe_innovation(&mut self, obs: &LinearUncorrelatedObserveModel<N, D, ZD>, s: &VectorN<N, ZD>) -> Result<N, &'static str>;
}

/// A linear observation.
///
/// Linear or linearised observation model with ccorrelated obseration noise.
pub trait LinearObservationCorrelated<N: RealField, D: Dim, ZD: Dim> : LinearEstimator<N>
    where DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D> + Allocator<N, ZD>
{
    fn observe_innovation(&mut self, obs: &LinearCorrelatedObserveModel<N, D, ZD>, s: &VectorN<N, ZD>) -> Result<N, &'static str>;
}

/// Additive noise.
///
/// Linear additive noise represented as a the noise variance vector and a noise coupling matrix.
/// The noise coveriance is G.q.G'.
pub struct AdditiveNoise<N: RealField, D: Dim, QD: Dim>
    where DefaultAllocator: Allocator<N, D, QD> + Allocator<N, QD>
{
    /// Noise variance
    pub q: VectorN<N, QD>,
    // Noise coupling
    pub G: MatrixMN<N, D, QD>
}

/// Linear prediction model.
///
/// Prediction is represented by a state transaition matrix.
pub struct LinearPredictModel<N: RealField, D: Dim>
    where DefaultAllocator: Allocator<N, D, D>
{
    // State tramsition matrix
    pub Fx: MatrixN<N, D>
}

/// Linear prediction model.
///
/// Prediction is represented by a observation matrix.
pub struct LinearUncorrelatedObserveModel<N: RealField, D: Dim, ZD: Dim>
    where DefaultAllocator: Allocator<N, ZD, D> + Allocator<N, D> + Allocator<N, ZD>
{
    pub Hx: MatrixMN<N, ZD, D>,
    pub Zv: VectorN<N, ZD>
}

/// Linear prediction model.
///
/// Prediction is represented by a observation matrix.
pub struct LinearCorrelatedObserveModel<N: RealField, D: Dim, ZD: Dim>
    where DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D>
{
    pub Hx: MatrixMN<N, ZD, D>,
    pub Z: MatrixN<N, ZD>
}

impl <'a, N: RealField, D: Dim, ZD: Dim> LinearCorrelatedObserveModel< N, D, ZD>
    where
        DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D> + Allocator<N, ZD>
{
    pub fn from_uncorrelated(uncorrelated : &'a LinearUncorrelatedObserveModel<N, D, ZD>) -> Self {
        LinearCorrelatedObserveModel {
            Hx: uncorrelated.Hx.clone(),
            Z: MatrixMN::from_diagonal(&uncorrelated.Zv)
        }
    }
}