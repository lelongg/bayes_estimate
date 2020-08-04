#![allow(non_snake_case)]
#![allow(unused_variables)]

//! A 'null' implementation of a filter with a Kalman state representation [`KalmanState`].
//!
//! No numerical computations are performed, the traits have implmentations which do nothing.
//!
//! [`KalmanState`]: ../models/struct.KalmanState.html
//!
use bayes_filter;
use nalgebra as na;

use bayes_filter::models::{
    AdditiveCorrelatedNoise, AdditiveNoise, KalmanState,
    LinearObservationCorrelated, LinearObservationUncorrelated, LinearObserveModel,
    LinearPredictModel, LinearPredictor,
};
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
        noise: &AdditiveCorrelatedNoise<N, D, QD>,
    ) -> Result<N, &'static str> {
        Result::Ok(N::one())
    }
}

impl<N: RealField, D: Dim, ZD: Dim, ZQD: Dim> LinearObservationCorrelated<N, D, ZD, ZQD>
    for NullState<N, D>
where
    DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, ZD, ZD>
        + Allocator<N, ZD, D>
        + Allocator<N, D, ZD>
        + Allocator<N, ZD, ZQD>
        + Allocator<N, D>
        + Allocator<N, ZD>
        + Allocator<N, ZQD>,
{
    fn observe_innovation(
        &mut self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &AdditiveCorrelatedNoise<N, ZD, ZQD>,
        s: &VectorN<N, ZD>,
    ) -> Result<N, &'static str> {
        Result::Ok(N::one())
    }
}

impl<N: RealField, D: Dim, ZD: Dim, ZQD: Dim> LinearObservationUncorrelated<N, D, ZD, ZQD>
    for NullState<N, D>
where
    DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, ZD, ZD>
        + Allocator<N, ZD, D>
        + Allocator<N, D, ZD>
        + Allocator<N, ZD, ZQD>
        + Allocator<N, D>
        + Allocator<N, ZD>
        + Allocator<N, ZQD>,
{
    fn observe_innovation(
        &mut self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &AdditiveNoise<N, ZQD>,
        s: &VectorN<N, ZD>,
    ) -> Result<N, &'static str> {
        Result::Ok(N::one())
    }
}
