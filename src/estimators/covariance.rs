#![allow(non_snake_case)]

//! Covariance state estimation.
//!
//! A discrete Bayesian estimator that uses a Kalman state representation [`KalmanState`] of the system for estimation.
//! The Kalman state is simply the x,X pair the dimensions of both are the dimensions of the system.
//!
//! The linear Kalman state representation can also be used for non-linear system by using linearised forms of the system model.
//!
//! [`KalmanState`]: ../models/struct.KalmanState.html

use na::{
    allocator::Allocator, DefaultAllocator, Dim, MatrixN, RealField, VectorN, U1,
};
use nalgebra as na;

use crate::linalg::cholesky;
use crate::mine::matrix::{prod_spd_udu, check_non_negativ};
use crate::models::{AdditiveCorrelatedNoise, KalmanEstimator, KalmanState, LinearObserverCorrelated, LinearObserveModel, LinearPredictModel, LinearPredictor, Estimator};

impl<N: RealField, D: Dim> KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub fn new(d: D) -> KalmanState<N, D> {
        KalmanState {
            x: VectorN::zeros_generic(d, U1),
            X: MatrixN::zeros_generic(d, d),
        }
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for KalmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        Result::Ok(self.x.clone())
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str> {
        self.x.copy_from(&state.x);
        self.X.copy_from(&state.X);
        check_non_negativ(cholesky::UDU::UdUrcond(&self.X), "X not PSD")?;

        Result::Ok(N::one())
    }

    fn state(&self) -> Result<(N, KalmanState<N, D>), &'static str> {
        Result::Ok((
            N::one(),
            KalmanState {
                x: self.x.clone(),
                X: self.X.clone(),
            },
        ))
    }
}

impl<N: RealField, D: Dim> LinearPredictor<N, D> for KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn predict(
        &mut self,
        pred: &LinearPredictModel<N, D>,
        x_pred: VectorN<N, D>,
        noise: &AdditiveCorrelatedNoise<N, D>,
    ) -> Result<(), &'static str> {
        self.x = x_pred;
        // X = Fx.X.FX' + Q
        self.X.quadform_tr(N::one(), &pred.Fx, &self.X.clone(), N::zero());
        self.X += &noise.Q;

        Result::Ok(())
    }
}

impl<N: RealField, D: Dim, ZD: Dim> LinearObserverCorrelated<N, D, ZD>
    for KalmanState<N, D>
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
        noise: &AdditiveCorrelatedNoise<N, ZD>,
        s: &VectorN<N, ZD>,
    ) -> Result<(), &'static str> {
        let XHt = &self.X * obs.Hx.transpose();
        // S = Hx.X.Hx' + G.q.G'
        let S = &obs.Hx * &XHt + prod_spd_udu(&noise.Q);

        // Inverse innovation covariance
        let SI = S.clone().cholesky().ok_or("S not PD in observe")?.inverse();

        // Kalman gain, X*Hx'*SI
        let W = XHt * SI;

        // State update
        self.x += &W * s;
        // X -= W.S.W'
        self.X.quadform_tr(N::one().neg(), &W, &S, N::one());

        Result::Ok(())
    }
}

