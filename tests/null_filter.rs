#![allow(non_snake_case)]
#![allow(unused_variables)]

use nalgebra as na;
use bayes_filter;

use na::{RealField, allocator::Allocator, DefaultAllocator, Dim, DimSub, Dynamic, VectorN};
use bayes_filter::models::{LinearEstimator, KalmanState, LinearPredictor, AdditiveNoise, LinearObservationCorrelated, LinearCorrelatedObserveModel, LinearUncorrelatedObserveModel, LinearObservationUncorrelated, LinearPredictModel};
use std::marker::PhantomData;


pub struct NullState<N: RealField, D: Dim>
{
    _phantoms: PhantomData<(N, D)>
}

impl<N: RealField, D: Dim> NullState<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub fn new(state: KalmanState<N, D>) -> NullState<N, D> {
        NullState {
            _phantoms: PhantomData
        }
    }
}

impl<N: RealField, D: Dim> LinearEstimator<N> for NullState<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{}

impl<N: RealField, D: Dim, QD: Dim> LinearPredictor<N, D, QD> for NullState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, QD, QD> + Allocator<N, D, QD> + Allocator<N, QD, D>
        + Allocator<N, D> + Allocator<N, QD>
{
    fn predict(&mut self, pred: &LinearPredictModel<N, D>, x_pred: VectorN<N, D>, noise: &AdditiveNoise<N, D, QD>) -> Result<N, &'static str> {
        Result::Ok(N::one())
    }
}

impl<N: RealField, D: Dim, ZD: DimSub<Dynamic>> LinearObservationCorrelated<N, D, ZD> for NullState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D, ZD>
        + Allocator<N, D> + Allocator<N, ZD>,
{
    fn observe_innovation(&mut self, obs: &LinearCorrelatedObserveModel<N, D, ZD>, s: &VectorN<N, ZD>) -> Result<N, &'static str> {
        Result::Ok(N::one())
    }
}

impl<N: RealField, D: Dim, ZD: DimSub<Dynamic>> LinearObservationUncorrelated<N, D, ZD> for NullState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D, ZD>
        + Allocator<N, D> + Allocator<N, ZD>,
{
    fn observe_innovation(&mut self, obs: &LinearUncorrelatedObserveModel<N, D, ZD>, s: &VectorN<N, ZD>) -> Result<N, &'static str> {
        Result::Ok(N::one())
    }
}
