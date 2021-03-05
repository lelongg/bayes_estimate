#![allow(non_snake_case)]

//! Julier-Uhlmann 'Unscented' state estimation.
//!
//! A discrete Bayesian estimator that uses the [`KalmanState`] linear representation of the system.
//! The 'Unscented' transform is used for non-linear state predictions and observation.
//!
//! The 'Unscented' transforma interpolates the non-linear predict and observe function.
//! Unscented transforms can be optimised for particular functions by vary the Kappa parameter from its usual value of 1.
//! Implements the classic Duplex 'Unscented' transform.

use nalgebra as na;
use na::{allocator::Allocator, DefaultAllocator, Dim, RealField, U1, VectorN, storage::Storage};

use crate::models::KalmanState;
use crate::noise::{CorrelatedNoise};
use crate::linalg::rcond;


impl<N: RealField, D: Dim> KalmanState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, U1, D>
{
    /// State prediction with a functional prediction model and additive noise.
    pub fn predict_unscented(&mut self, f: fn(&VectorN<N, D>) -> VectorN<N, D>, noise: &CorrelatedNoise<N, D>, kappa: N) -> Result<(), &'static str>
    {
        // Create Unscented distribution
        let x_kappa = N::from_usize(self.x.nrows()).unwrap() + kappa;
        let (mut UU, _rcond) = unscented(&self, x_kappa)?;

        // Predict points of XX using supplied predict model
        for c in 0..(UU.len()) {
            UU[c] = f(&UU[c]);
        }

        // State covariance
        kalman(self, &UU, kappa);
        // Additive Noise
        self.X += &noise.Q;

        Ok(())
    }

    pub fn observe_unscented<ZD: Dim>(
        &mut self,
        h: fn(&VectorN<N, D>) -> VectorN<N, ZD>,
        h_normalise: fn(&mut VectorN<N, ZD>, &VectorN<N, ZD>),
        noise: &CorrelatedNoise<N, ZD>, s:
        &VectorN<N, ZD>, kappa: N)
        -> Result<(), &'static str>
        where
            DefaultAllocator: Allocator<N, D, ZD> + Allocator<N, ZD, ZD> + Allocator<N, U1, ZD> + Allocator<N, ZD>
    {
        // Create Unscented distribution
        let x_kappa = N::from_usize(self.x.nrows()).unwrap() + kappa;
        let (UU, _rcond) = unscented(&self, x_kappa)?;

        // Predict points of ZZ using supplied observation model
        let usize = UU.len();
        let mut ZZ: Vec<VectorN<N, ZD>> = Vec::with_capacity(usize);
        ZZ.push(h(&UU[0]));
        for i in 1..usize {
            let mut zi = h(&UU[i]);
            h_normalise(&mut zi, &ZZ[0]);
            ZZ.push(zi);
        }

        // Mean and covariance of observation distribution
        let mut zZ = KalmanState::<N, ZD>::new_zero(s.data.shape().0);
        kalman(&mut zZ, &ZZ, kappa);
        for i in 0..usize {
            ZZ[i] -= &zZ.x;
        }

        let two = N::from_u32(2).unwrap();

        // Correlation of state with observation: Xxz
        // Center point, premult here by 2 for efficiency
        let x = &self.x;
        let mut XZ;
        {
            let XX0 = &UU[0] - x;
            XZ = XX0 * ZZ[0].transpose() * two * kappa;
        }

        // Remaining Unscented points
        for i in 1..ZZ.len() {
            let XXi = (&UU[i] - x).clone_owned();
            XZ += XXi * ZZ[i].transpose();
        }
        XZ /= two * x_kappa;

        let S = zZ.X + &noise.Q;

        // Inverse innovation covariance
        let SI = S.clone().cholesky().ok_or("S not PD in observe")?.inverse();

        // Kalman gain, X*Hx'*SI
        let W = &XZ * SI;

        // State update
        self.x += &W * s;
        // X -= W.S.W'
        self.X.quadform_tr(N::one().neg(), &W, &S, N::one());

        Ok(())
    }
}

pub fn unscented<N: RealField, D: Dim>(xX: &KalmanState<N, D>, scale: N) -> Result<(Vec<VectorN<N, D>>, N), &'static str>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    let sigma = xX.X.clone().cholesky().ok_or("unscented X not PSD")?.l() * scale.sqrt();

    // Generate UU with the same sample Mean and Covariance
    let mut UU: Vec<VectorN<N, D>> = Vec::with_capacity(2 * xX.x.nrows() + 1);
    UU.push(xX.x.clone());

    for c in 0..xX.x.nrows() {
        let sigmaCol = sigma.column(c);
        UU.push(&xX.x + &sigmaCol);
        UU.push(&xX.x - &sigmaCol);
    }

    Ok((UU, rcond::rcond_symetric(&xX.X)))
}

pub fn kalman<N: RealField, D: Dim>(state: &mut KalmanState<N, D>, XX: &Vec<VectorN<N, D>>, scale: N)
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, U1, D>
{
    let two = N::from_u32(2).unwrap();
    let half = N::one() / two;

    let x_scale = N::from_usize((XX.len() - 1) / 2).unwrap() + scale;
    // Mean of predicted distribution: x
    state.x = &XX[0] * scale;
    for i in 1..XX.len() {
        state.x += XX[i].scale(half);
    }
    state.x /= x_scale;

    // Covariance of distribution: X
    // Center point, premult here by 2 for efficiency
    {
        let XX0 = &XX[0] - &state.x;
        let XX0t = XX0.transpose() * two * scale;
        state.X = XX0 * XX0t;
    }
    // Remaining Unscented points
    for i in 1..XX.len() {
        let XXi = &XX[i] - &state.x;
        let XXit = XXi.transpose();
        state.X += XXi * XXit;
    }
    state.X /= two * x_scale;
}
