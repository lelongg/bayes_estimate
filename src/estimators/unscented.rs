#![allow(non_snake_case)]

//! Unscented state estimation.
//!

use std::ops::SubAssign;

use simba::simd::SimdComplexField;

use na::{
    allocator::Allocator, DefaultAllocator, Dim, RealField, U1, VectorN,
};
use nalgebra as na;
use nalgebra::{Dynamic, MatrixMN};
use nalgebra::storage::Storage;

use crate::linalg::cholesky::UDU;
use crate::models::{AdditiveCorrelatedNoise, KalmanEstimator, KalmanState, FunctionPredictor, FunctionObserverCorrelated, Estimator};
use crate::mine::matrix::check_non_negativ;
use crate::mine::matrix;
use crate::linalg::cholesky;

pub struct UnscentedKallmanState<N: RealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, Dynamic> + Allocator<N, D, D> + Allocator<N, D>
{
    pub xX: KalmanState<N, D>,
    /// Unscented state
    pub UU: MatrixMN<N, D, Dynamic>,
    pub kappa: N
}

impl<N: RealField, D: Dim> UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, Dynamic> + Allocator<N, D, D> + Allocator<N, U1, D> + Allocator<N, D>,
{
    pub fn new(d: D) -> UnscentedKallmanState<N, D> {
        UnscentedKallmanState {
            xX: KalmanState::new(d),
            UU: MatrixMN::zeros_generic(d, Dynamic::new(d.value() * 2 + 1)),
            kappa: N::from_usize(3 - d.value()).unwrap()
        }
    }
}

impl<N: RealField, D: Dim> FunctionPredictor<N, D> for UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, Dynamic> + Allocator<N, U1, D> + Allocator<N, D>,
{
    // State prediction with a linear prediction model and additive noise.
    fn predict(
        &mut self,
        f: fn(&VectorN<N, D>) -> VectorN<N, D>,
        noise: &AdditiveCorrelatedNoise<N, D>) -> Result<(), &'static str> {
        // Create Unscented distribution
        let x_kappa = N::from_usize(self.xX.x.nrows()).unwrap() + self.kappa;
        let _rcond = unscented(&mut self.UU, &self.xX, x_kappa)?;

        // Predict points of XX using supplied predict model
        for c in 0..(self.UU.ncols()) {
            let UUc = self.UU.column(c).clone_owned();
            let ff = f(&UUc);
            self.UU.column_mut(c).copy_from(&ff);
        }

        //  State covariance
        kalman(&mut self.xX, &self.UU, self.kappa);
        // Additive Noise
        self.xX.X += &noise.Q;

        Result::Ok(())
    }
}

impl<N: RealField, D: Dim, ZD: Dim> FunctionObserverCorrelated<N, D, ZD> for UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, ZD> + Allocator<N, ZD, ZD> + Allocator<N, D, Dynamic> + Allocator<N, ZD, Dynamic> + Allocator<N, U1, ZD> + Allocator<N, D> + Allocator<N, ZD> {
    fn observe_innovation(&mut self, h: fn(&VectorN<N, D>) -> VectorN<N, ZD>, noise: &AdditiveCorrelatedNoise<N, ZD>, s: &VectorN<N, ZD>) -> Result<(), &'static str> {
        // Create Unscented distribution
        unscented(&mut self.UU, &self.xX, self.kappa)?;

        // Predict points of ZZ using supplied observation model
        let mut ZZ = matrix::as_zeros((s.data.shape().0, self.UU.data.shape().1));
        for z in 0..ZZ.data.shape().1.value() {
            ZZ.column_mut(z).copy_from(&h(&self.UU.column(z).clone_owned()))
        }

        let mut zZ = KalmanState::<N, ZD>::new(ZZ.data.shape().0);
        kalman(&mut zZ, &ZZ, self.kappa);

        let two = N::from_u32(2).unwrap();

        // Correlation of state with observation: Xxz
        // Center point, premult here by 2 for efficiency
        let mut XZ;
        {
            let XX0 = (self.UU.column(0) - &self.xX.x).clone_owned();
            let ZZ0t = ZZ.column(0).transpose();
            XZ = XX0 * ZZ0t;
            XZ *= two * self.kappa;
        }

        // Remaining Unscented points
        for i in 1..ZZ.ncols() {
            let XXi = (self.UU.column(i) - &self.xX.x).clone_owned();
            let ZZit = ZZ.column(i).transpose();
            XZ += XXi * ZZit;
        }
        XZ /= two * self.kappa;

        let S = zZ.X + &noise.Q;

        // Inverse innovation covariance
        let SI = S.clone().cholesky().ok_or("S not PD in observe")?.inverse();

        // Kalman gain, X*Hx'*SI
        let W = XZ * SI;

        // State update
        self.xX.x += &W * s;
        // X -= W.S.W'
        self.xX.X.quadform_tr(N::one().neg(), &W, &S, N::one());

        Result::Ok(())
    }
}

pub fn unscented<N: RealField, D: Dim>(UU: &mut MatrixMN<N, D, Dynamic>, xX: &KalmanState<N, D>, scale: N) -> Result<N, &'static str>
    where
        DefaultAllocator: Allocator<N, D, Dynamic> + Allocator<N, D, D> + Allocator<N, D>
{
    let xsize = UU.nrows();
    let udu = UDU::new();

    let mut sigma = xX.X.clone();
    let rcond = udu.UCfactor_n(&mut sigma, xX.x.nrows());
    check_non_negativ(rcond, "Unscented X not PSD")?;
    sigma *= scale.simd_sqrt();

    // Generate XX with the same sample Mean and Covariance
    UU.column_mut(0).copy_from(&xX.x);

    for c in 0..xX.x.nrows() {
        let sigmaCol = sigma.column(c);
        let xp: VectorN<N, D> = &xX.x + &sigmaCol;
        UU.column_mut(c + 1).copy_from(&xp);
        let xm: VectorN<N, D> = &xX.x - &sigmaCol;
        UU.column_mut(c + 1 + xsize).copy_from(&xm);
    }

    Result::Ok(rcond)
}

pub fn kalman<N: RealField, D: Dim>(state: &mut KalmanState<N, D>, XX: &MatrixMN<N, D, Dynamic>, scale: N)
    where
        DefaultAllocator: Allocator<N, D, Dynamic> + Allocator<N, D, D> + Allocator<N, D> + Allocator<N, U1, D>
{
    let two = N::from_u32(2).unwrap();
    let half = N::one() / two;

    let mut tXX = XX.clone();
    let xsize = XX.nrows();
    let x_scale = N::from_usize(xsize).unwrap() + scale;
    // Mean of predicted distribution: x
    state.x = tXX.column(0) * scale;
    for i in 1..tXX.ncols() {
        state.x += tXX.column(i).scale(half);
    }
    state.x /= x_scale;
    // Covariance of distribution: X
    // Subtract mean from each point in tXX
    for i in 0..tXX.ncols() {
        &tXX.column_mut(i).sub_assign(&state.x);
    }
    // Center point, premult here by 2 for efficiency
    {
        let XX0 = tXX.column(0).clone_owned();
        let XX0t = XX0.transpose() * two * scale;
        state.X.copy_from(&(XX0 * XX0t));
    }
    // Remaining Unscented points
    for i in 1..xsize {
        let XXi = tXX.column(i);
        let XXit = XXi.transpose();
        state.X += &(XXi * XXit);
    }
    state.X /= two * x_scale;
}

impl<N: RealField, D: Dim> Estimator<N, D> for UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, Dynamic> + Allocator<N, U1, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N,D>, &'static str> {
        return Result::Ok(self.xX.x.clone());
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, Dynamic> + Allocator<N, U1, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str> {
        self.xX.x.copy_from(&state.x);
        self.xX.X.copy_from(&state.X);
        check_non_negativ(cholesky::UDU::UdUrcond(&self.xX.X), "X not PSD")?;

        Result::Ok(N::one())
    }

    fn state(&self) -> Result<(N, KalmanState<N, D>), &'static str> {
        return Result::Ok(
            (N::one(), self.xX.clone())
        );
    }
}

