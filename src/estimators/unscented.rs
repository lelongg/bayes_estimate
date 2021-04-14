#![allow(non_snake_case)]

//! Julier-Uhlmann 'Unscented' state estimation.
//!
//! A discrete Bayesian estimator that uses the [`KalmanState`] linear representation of the system.
//! The 'Unscented' transform is used for non-linear state predictions and observation.
//!
//! The 'Unscented' transforma interpolates the non-linear predict and observe function.
//! Unscented transforms can be optimised for particular functions by vary the Kappa parameter from its usual value of 1.
//! Implements the classic Duplex 'Unscented' transform.

use na::{allocator::Allocator, storage::Storage, DefaultAllocator, Dim, RealField, VectorN, U1};
use nalgebra as na;

use crate::linalg::rcond;
use crate::matrix::quadform_tr_x;
use crate::models::KalmanState;
use crate::noise::CorrelatedNoise;

impl<N: RealField, D: Dim> KalmanState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, U1, D>,
{
    /// State prediction with a functional prediction model and additive noise.
    pub fn predict_unscented(
        &mut self,
        f: fn(&VectorN<N, D>) -> VectorN<N, D>,
        noise: &CorrelatedNoise<N, D>,
        kappa: N,
    ) -> Result<(), &'static str> {
        // Create Unscented distribution
        let x_kappa = N::from_usize(self.x.nrows()).unwrap() + kappa;
        let (mut UU, _rcond) = unscented(&self, x_kappa)?;

        // Predict points of XX using supplied predict model
        for uu in &mut UU {
            *uu = f(uu);
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
        noise: &CorrelatedNoise<N, ZD>,
        s: &VectorN<N, ZD>,
        kappa: N,
    ) -> Result<(), &'static str>
    where
        DefaultAllocator:
            Allocator<N, D, ZD> + Allocator<N, ZD, ZD> + Allocator<N, U1, ZD> + Allocator<N, ZD>,
    {
        // Create Unscented distribution
        let x_kappa = N::from_usize(self.x.nrows()).unwrap() + kappa;
        let (UU, _rcond) = unscented(&self, x_kappa)?;

        // Predict points of ZZ using supplied observation model
        let usize = UU.len();
        let mut ZZ: Vec<VectorN<N, ZD>> = Vec::with_capacity(usize);
        ZZ.push(h(&UU[0]));
        for uu in &UU {
            let mut zi = h(uu);
            h_normalise(&mut zi, &ZZ[0]);
            ZZ.push(zi);
        }

        // Mean and covariance of observation distribution
        let mut zZ = KalmanState::<N, ZD>::new_zero(s.data.shape().0);
        kalman(&mut zZ, &ZZ, kappa);
        for zz in &mut ZZ {
            *zz -= &zZ.x;
        }

        let two = N::from_u32(2).unwrap();

        // Correlation of state with observation: Xxz
        // Center point, premult here by 2 for efficiency
        let x = &self.x;
        let mut XZ = (&UU[0] - x) * ZZ[0].transpose() * two * kappa;
        // Remaining Unscented points
        for i in 1..ZZ.len() {
            XZ += &(&UU[i] - x) * ZZ[i].transpose();
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

#[allow(clippy::type_complexity)]
pub fn unscented<N: RealField, D: Dim>(
    xX: &KalmanState<N, D>,
    scale: N,
) -> Result<(Vec<VectorN<N, D>>, N), &'static str>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    let sigma = xX.X.clone().cholesky().ok_or("unscented X not PSD")?.l() * scale.sqrt();

    // Generate UU with the same sample Mean and Covariance
    let mut UU: Vec<VectorN<N, D>> = Vec::with_capacity(2 * xX.x.nrows() + 1);
    UU.push(xX.x.clone());

    for c in 0..xX.x.nrows() {
        let sigmaCol = sigma.column(c);
        UU.push(&xX.x + sigmaCol);
        UU.push(&xX.x - sigmaCol);
    }

    Ok((UU, rcond::rcond_symetric(&xX.X)))
}

pub fn kalman<N: RealField, D: Dim>(state: &mut KalmanState<N, D>, XX: &[VectorN<N, D>], scale: N)
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, U1, D>,
{
    let two = N::from_u32(2).unwrap();

    let x_scale = N::from_usize((XX.len() - 1) / 2).unwrap() + scale;
    // Mean of predicted distribution: x
    state.x = &XX[0] * two * scale;
    for xx in XX {
        state.x += xx;
    }
    state.x /= two * x_scale;

    // Covariance of distribution: X
    // Center point, premult here by 2 for efficiency
    quadform_tr_x(&mut state.X, two * scale, &(&XX[0] - &state.x), N::zero());
    // Remaining Unscented points
    for xx in XX {
        quadform_tr_x(&mut state.X, N::one(), &(xx - &state.x), N::one());
    }
    state.X /= two * x_scale;
}
