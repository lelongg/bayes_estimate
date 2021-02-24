#![allow(non_snake_case)]

//! Covariance state estimation.
//!
//! A discrete Bayesian estimator that uses the 'Kalman' linear representation [`KalmanState`] of the system for estimation.
//!
//! The linear representation can also be used for non-linear systems by using linearised models.
//!
//! [`KalmanState`]: ../models/struct.KalmanState.html

use nalgebra as na;
use na::{allocator::Allocator, DefaultAllocator, Dim, U1, MatrixN, RealField, VectorN};

use crate::linalg::cholesky;
use crate::mine::matrix::{check_non_negativ};
use crate::models::{KalmanEstimator, KalmanState, ExtendedLinearObserver, ExtendedLinearPredictor, Estimator};
use crate::noise::{CorrelatedNoise};
use nalgebra::MatrixMN;

impl<N: RealField, D: Dim> KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub fn new_zero(d: D) -> KalmanState<N, D> {
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
        Ok(self.x.clone())
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str> {
        self.x = state.x.clone();
        self.X = state.X.clone();
        let rcond = cholesky::UDU::UdUrcond(&self.X);
        check_non_negativ(rcond, "X not PSD")?;

        Ok(rcond)
    }

    fn kalman_state(&self) -> Result<(N, KalmanState<N, D>), &'static str> {
        Ok((
            N::one(),
            self.clone()
        ))
    }
}

impl<N: RealField, D: Dim> ExtendedLinearPredictor<N, D> for KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn predict(
        &mut self,
        x_pred: VectorN<N, D>,
        fx: &MatrixN<N, D>,
        noise: &CorrelatedNoise<N, D>,
    ) -> Result<(), &'static str> {
        self.x = x_pred;
        // X = Fx.X.FX' + Q
        self.X.quadform_tr(N::one(), &fx, &self.X.clone(), N::zero());
        self.X += &noise.Q;

        Ok(())
    }
}

impl<N: RealField, D: Dim, ZD: Dim> ExtendedLinearObserver<N, D, ZD> for KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D, ZD> + Allocator<N, D> + Allocator<N, ZD>
{
    fn observe_innovation(
        &mut self,
        s: &VectorN<N, ZD>,
        hx: &MatrixMN<N, ZD, D>,
        noise: &CorrelatedNoise<N, ZD>,
    ) -> Result<(), &'static str> {
        let XHt = &self.X * hx.transpose();
        // S = Hx.X.Hx' + Q
        let S = hx * &XHt + &noise.Q;

        // Inverse innovation covariance
        let SI = S.clone().cholesky().ok_or("S not PD in observe")?.inverse();
        // Kalman gain, X*Hx'*SI
        let W = &XHt * SI;

        // State update
        self.x += &W * s;
        // X -= W.S.W'
        self.X.quadform_tr(N::one().neg(), &W, &S, N::one());

        Ok(())
    }
}

