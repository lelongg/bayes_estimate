#![allow(non_snake_case)]

use nalgebra as na;

use na::{RealField, Dim, DimSub, Dynamic, U1, DefaultAllocator, allocator::Allocator, MatrixN, VectorN};
use na::storage::Storage;
use crate::models::{LinearEstimator, KalmanState, LinearPredictor, KalmanEstimator, LinearObservationCorrelated, LinearObservationUncorrelated,
                    LinearPredictModel, AdditiveNoise, AdditiveCorrelatedNoise, LinearObserveModel};
use crate::mine::matrix::{prod_spd, check_positive};
use crate::linalg::cholesky;


impl<N: RealField, D: Dim> KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub fn new(d : D) -> KalmanState<N, D> {
        KalmanState {
            x: VectorN::zeros_generic(d, U1),
            X: MatrixN::zeros_generic(d, d)
        }
    }
}

impl<N: RealField, D: Dim> LinearEstimator<N> for KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for KalmanState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn init(&mut self, state : &KalmanState<N, D>) -> Result<N, &'static str> {
        self.x.copy_from(&state.x);
        self.X.copy_from(&state.X);
        check_positive(cholesky::UDU::UdUrcond(&self.X), "X not PD")?;

        Result::Ok(N::one())
    }

    fn state(&self) -> Result<(N, KalmanState<N, D>), &'static str> {
        Result::Ok((N::one(), KalmanState{
            x: self.x.clone(),
            X: self.X.clone()
        }))
    }
}

impl<N: RealField, D: Dim, QD: Dim> LinearPredictor<N, D, QD> for KalmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, QD, QD> + Allocator<N, D, QD> + Allocator<N, QD, D>
        + Allocator<N, D> + Allocator<N, QD>
{
    fn predict(&mut self, pred: &LinearPredictModel<N, D>, x_pred : VectorN<N, D>, noise: &AdditiveCorrelatedNoise<N, D, QD>) -> Result<N, &'static str> {
        self.x = x_pred;
        self.X = &pred.Fx * self.X.clone() * pred.Fx.transpose() + prod_spd(&noise.G, &MatrixN::from_diagonal(&noise.q));

        Result::Ok(N::one())
    }
}

impl<N: RealField, D: Dim, ZD: DimSub<Dynamic>, ZQD: Dim> LinearObservationCorrelated<N, D, ZD, ZQD> for KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D, ZD>
    + Allocator<N, ZQD, ZQD> + Allocator<N, ZD, ZQD> + Allocator<N, ZQD, ZD>
    + Allocator<N, D> + Allocator<N, ZD> + Allocator<N, ZQD>
{
    fn observe_innovation(&mut self, obs: &LinearObserveModel<N, D, ZD>, noise : &AdditiveCorrelatedNoise<N, ZD, ZQD>, s: &VectorN<N, ZD>) -> Result<N, &'static str> {
        let XHt = &self.X * obs.Hx.transpose();
        let S = &obs.Hx * &XHt + prod_spd(&noise.G, &MatrixN::from_diagonal(&noise.q));
        let S2 = S.clone();

        // Inverse innovation covariance
        let SI = S.cholesky().ok_or("S not PD in observe")?.inverse();

        // Kalman gain, X*Hx'*SI
        let W = XHt * SI;

        // State update
        self.x += &W * s;
        self.X -= prod_spd(&W, &S2);

        Result::Ok(N::one())
    }
}

impl<N: RealField, D: Dim, ZD: DimSub<Dynamic>, ZQD: Dim> LinearObservationUncorrelated<N, D, ZD, ZQD> for KalmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D, ZD>
        + Allocator<N, ZQD, ZQD> + Allocator<N, ZD, ZQD> + Allocator<N, ZQD, ZD>
        + Allocator<N, D> + Allocator<N, ZD> + Allocator<N, ZQD>
{
    fn observe_innovation(&mut self, obs: &LinearObserveModel<N, D, ZD>, noise : &AdditiveNoise<N, ZQD>, s: &VectorN<N, ZD>) -> Result<N, &'static str> {
        let dim : ZD = obs.Hx.data.shape().0;
        let correlated = AdditiveCorrelatedNoise::<N,ZD,ZQD>::from_uncorrelated(noise, dim);
        LinearObservationCorrelated::observe_innovation(self, obs, &correlated, s)
    }
}
