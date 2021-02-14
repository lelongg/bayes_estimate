#![allow(non_snake_case)]
#![allow(unused_variables)]

//! A 'null' implementation of a filter with a Kalman state representation [`KalmanState`].
//!
//! No numerical computations are performed, the traits have implmentations which do nothing.
//!
//! [`KalmanState`]: ../models/struct.KalmanState.html
//!
use bayes_estimate;
use nalgebra as na;

use bayes_estimate::models::{CorrelatedNoise, UncorrelatedNoise, KalmanState, LinearObserver, LinearObserverUncorrelated, LinearObserveModel, LinearPredictModel, LinearPredictor, CoupledNoise};
use na::{allocator::Allocator, DefaultAllocator, Dim, RealField, VectorN};
use std::marker::PhantomData;

pub struct NullState<N: RealField, D: Dim> {
    _phantoms: PhantomData<(N, D)>,
}

impl<N: RealField, D: Dim> NullState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub fn new(state: KalmanState<N, D>) -> NullState<N, D> {
        NullState {
            _phantoms: PhantomData,
        }
    }
}

impl<N: RealField, D: Dim, QD: Dim> LinearPredictor<N, D, QD> for NullState<N, D>
where
    DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, QD, QD>
        + Allocator<N, D, QD>
        + Allocator<N, QD, D>
        + Allocator<N, D>
        + Allocator<N, QD>,
{
    fn predict(
        &mut self,
        pred: &LinearPredictModel<N, D>,
        x_pred: VectorN<N, D>,
        noise: &CoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str> {
        Ok(N::one())
    }
}

impl<N: RealField, D: Dim, ZD: Dim> LinearObserver<N, D, ZD>
    for NullState<N, D>
where
    DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, ZD, ZD>
        + Allocator<N, ZD, D>
        + Allocator<N, D, ZD>
        + Allocator<N, D>
        + Allocator<N, ZD>
{
    fn observe_innovation(
        &mut self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &CorrelatedNoise<N, ZD>,
        s: &VectorN<N, ZD>,
    ) -> Result<N, &'static str> {
        Ok(N::one())
    }
}

impl<N: RealField, D: Dim, ZD: Dim> LinearObserverUncorrelated<N, D, ZD>
    for NullState<N, D>
where
    DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, ZD, ZD>
        + Allocator<N, ZD, D>
        + Allocator<N, D, ZD>
        + Allocator<N, D>
        + Allocator<N, ZD>
{
    fn observe_innovation(
        &mut self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &UncorrelatedNoise<N, ZD>,
        s: &VectorN<N, ZD>,
    ) -> Result<N, &'static str> {
        Ok(N::one())
    }
}
