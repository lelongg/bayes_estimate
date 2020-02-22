#![allow(non_snake_case)]

use nalgebra as na;

use na::RealField;
use na::{DefaultAllocator, allocator::Allocator, Dim, MatrixMN, MatrixN, VectorN};


#[derive(PartialEq, Clone)]
pub struct KalmanState<N: RealField, D: Dim>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub x: VectorN<N, D>,
    pub X: MatrixN<N, D>
}

#[derive(PartialEq, Clone)]
pub struct InformationState<N: RealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub i: VectorN<N, D>,
    pub I: MatrixN<N, D>
}

pub trait KalmanEstimator<N: RealField, D: Dim>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn init(&mut self, state : &KalmanState<N, D>) -> Result<N, &'static str>;
    fn state(&self) -> Result<(N, KalmanState<N, D>), &'static str>;
}

pub trait LinearEstimator<N: RealField>
{
}

pub trait LinearPredictor<N: RealField, D: Dim, QD: Dim>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, QD> + Allocator<N, D> + Allocator<N, QD>
{
    fn predict(&mut self, pred: &LinearPredictModel<N, D>, x_pred: VectorN<N, D>, noise: &AdditiveNoise<N, D, QD>) -> Result<N, &'static str>;
}

pub trait LinearObservationUncorrelated<N: RealField, D: Dim, ZD: Dim>
    where DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D> + Allocator<N, ZD>
{
    fn observe_innovation(&mut self, obs: &LinearUncorrelatedObserveModel<N, D, ZD>, s: &VectorN<N, ZD>) -> Result<N, &'static str>;
}

pub trait LinearObservationCorrelated<N: RealField, D: Dim, ZD: Dim>
    where DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D> + Allocator<N, ZD>
{
    fn observe_innovation(&mut self, obs: &LinearCorrelatedObserveModel<N, D, ZD>, s: &VectorN<N, ZD>) -> Result<N, &'static str>;
}

pub struct AdditiveNoise<N: RealField, D: Dim, QD: Dim>
    where DefaultAllocator: Allocator<N, D, QD> + Allocator<N, QD>
{
    pub q: VectorN<N, QD>, // Noise variance
    pub G: MatrixMN<N, D, QD> // Noise Coupling
}

pub struct LinearPredictModel<N: RealField, D: Dim>
    where DefaultAllocator: Allocator<N, D, D>
{
    pub Fx: MatrixN<N, D>
}

pub struct LinearUncorrelatedObserveModel<N: RealField, D: Dim, ZD: Dim>
    where DefaultAllocator: Allocator<N, ZD, D> + Allocator<N, D> + Allocator<N, ZD>
{
    pub Hx: MatrixMN<N, ZD, D>,
    pub Zv: VectorN<N, ZD>
}

pub struct LinearCorrelatedObserveModel<N: RealField, D: Dim, ZD: Dim>
    where
        DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D>
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