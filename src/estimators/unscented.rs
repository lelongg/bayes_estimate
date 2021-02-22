#![allow(non_snake_case)]

//! Unscented state estimation.
//!

use simba::simd::SimdComplexField;

use na::{
    allocator::Allocator, DefaultAllocator, Dim, RealField, U1, VectorN,
};
use nalgebra as na;
use nalgebra::storage::Storage;

use crate::linalg::cholesky::UDU;
use crate::models::{KalmanEstimator, KalmanState, FunctionPredictor, FunctionObserver, Estimator};
use crate::noise::{CorrelatedNoise};
use crate::mine::matrix::{check_non_negativ};
use crate::linalg::cholesky;


pub struct UnscentedKallmanState<N:RealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub xX: KalmanState<N, D>,
    /// Unscented state
    pub UU: Vec<VectorN<N, D>>,
    pub kappa: N,
}

impl<N: RealField, D: Dim> UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub fn new_zero(d: D) -> UnscentedKallmanState<N, D> {
        let usize = 2 * d.value() + 1;
        let mut UU: Vec<VectorN<N, D>> = Vec::with_capacity(usize);
        let zu: VectorN<N, D> = VectorN::zeros_generic(d, U1);
        for _u in 0..usize {
            UU.push(zu.clone());
        }
        UnscentedKallmanState {
            xX: KalmanState::new_zero(d),
            UU,
            kappa: N::from_usize(3 - d.value()).unwrap(),
        }
    }
}

impl<N: RealField, D: Dim> FunctionPredictor<N, D> for UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, U1, D>
{
    // State prediction with a linear prediction model and additive noise.
    fn predict(&mut self, f: fn(&VectorN<N, D>) -> VectorN<N, D>, noise: &CorrelatedNoise<N, D>) -> Result<(), &'static str>
    {
        // Create Unscented distribution
        let x_kappa = N::from_usize(self.xX.x.nrows()).unwrap() + self.kappa;
        let _rcond = unscented(&mut self.UU, &self.xX, x_kappa)?;

        // Predict points of XX using supplied predict model
        for c in 0..(self.UU.len()) {
            let ff = f(&&self.UU[c]);
            self.UU[c].copy_from(&ff);
        }

        // State covariance
        kalman(&mut self.xX, &self.UU, self.kappa);
        // Additive Noise
        self.xX.X += &noise.Q;

        Ok(())
    }
}

pub fn unscented<N: RealField, D: Dim>(UU: &mut Vec<VectorN<N, D>>, xX: &KalmanState<N, D>, scale: N) -> Result<N, &'static str>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    let xsize = xX.x.nrows();
    let udu = UDU::new();

    let mut sigma = xX.X.clone();
    let rcond = udu.UCfactor_n(&mut sigma, xX.x.nrows());

    check_non_negativ(rcond, "Unscented X not PSD")?;
    sigma *= scale.simd_sqrt();

    // Generate XX with the same sample Mean and Covariance
    UU[0] = xX.x.clone();

    for c in 0..xsize {
        let sigmaCol = sigma.column(c);
        let xp: VectorN<N, D> = &xX.x + &sigmaCol;
        UU[c + 1] = xp;
        let xm: VectorN<N, D> = &xX.x - &sigmaCol;
        UU[c + 1 + xsize] = xm;
    }

    Ok(rcond)
}

impl<N: RealField, D: Dim, ZD: Dim> FunctionObserver<N, D, ZD> for UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
        DefaultAllocator: Allocator<N, D, ZD> + Allocator<N, ZD, ZD> + Allocator<N, U1, ZD> + Allocator<N, ZD> {

    fn observe_innovation(&mut self, h: fn(&VectorN<N, D>, &VectorN<N, D>) -> VectorN<N, ZD>, noise: &CorrelatedNoise<N, ZD>, s: &VectorN<N, ZD>) -> Result<(), &'static str> {
        // Create Unscented distribution
        let x_kappa = N::from_usize(self.xX.x.nrows()).unwrap() + self.kappa;
        unscented(&mut self.UU, &self.xX, x_kappa)?;

        // Predict points of ZZ using supplied observation model
        let usize = self.UU.len();
        let mut ZZ: Vec<VectorN<N, ZD>> = Vec::with_capacity(usize);
        let xm = &self.xX.x;
        for i in 0..usize {
            ZZ.push(h(&self.UU[i], xm))
        }

        // Mean and covarnaic of observation distribution
        let mut zZ = KalmanState::<N, ZD>::new_zero(s.data.shape().0);
        kalman(&mut zZ, &ZZ, self.kappa);
        for i in 0..ZZ.len() {
            ZZ[i] -= &zZ.x;
        }

        let two = N::from_u32(2).unwrap();

        // Correlation of state with observation: Xxz
        // Center point, premult here by 2 for efficiency
        let x = &self.xX.x;
        let mut XZ;
        {
            let XX0 = &self.UU[0] - x;
            let ZZ0t = ZZ[0].transpose();
            XZ = XX0 * ZZ0t;
            XZ *= two * self.kappa;
        }

        // Remaining Unscented points
        for i in 1..ZZ.len() {
            let XXi = (&self.UU[i] - x).clone_owned();
            let ZZit = ZZ[i].transpose();
            XZ += XXi * ZZit;
        }
        XZ /= two * x_kappa;

        let S = zZ.X + &noise.Q;

        // Inverse innovation covariance
        let SI = S.clone().cholesky().ok_or("S not PD in observe")?.inverse();

        // Kalman gain, X*Hx'*SI
        let W = &XZ * SI;

        // State update
        self.xX.x += &W * s;
        // X -= W.S.W'
        self.xX.X.quadform_tr(N::one().neg(), &W, &S, N::one());

        Ok(())
    }
}

pub fn kalman<N: RealField, D: Dim>(state: &mut KalmanState<N, D>, XX: &Vec<VectorN<N, D>>, scale: N)
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, U1, D>
{
    let two = N::from_u32(2).unwrap();
    let half = N::one() / two;

    let mut tXX = XX.clone();
    let x_scale = N::from_usize((XX.len()-1)/2).unwrap() + scale;
    // Mean of predicted distribution: x
    state.x = &tXX[0] * scale;
    for i in 1..tXX.len() {
        state.x += tXX[i].scale(half);
    }
    state.x /= x_scale;
    // Covariance of distribution: X
    // Subtract mean from each point in tXX
    for i in 0..tXX.len() {
        tXX[i] -= &state.x;
    }
    // Center point, premult here by 2 for efficiency
    {
        let XX0 = &tXX[0];
        let XX0t = XX0.transpose() * two * scale;
        state.X.copy_from(&(XX0 * XX0t));
    }
    // Remaining Unscented points
    for i in 1..tXX.len() {
        let XXi = &tXX[i];
        let XXit = XXi.transpose();
        state.X += &(XXi * XXit);
    }
    state.X /= two * x_scale;
}

impl<N: RealField, D: Dim> Estimator<N, D> for UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, U1, D>
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        return Ok(self.xX.x.clone());
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for UnscentedKallmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, U1, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str> {
        self.xX.x.copy_from(&state.x);
        self.xX.X.copy_from(&state.X);
        check_non_negativ(cholesky::UDU::UdUrcond(&self.xX.X), "X not PSD")?;

        Ok(N::one())
    }

    fn kalman_state(&self) -> Result<(N, KalmanState<N, D>), &'static str> {
        return Ok(
            (N::one(), self.xX.clone())
        );
    }
}

