#![allow(non_snake_case)]

//! Information 'square root' state estimation.
//!
//! A discrete Bayesian estimator that uses a linear information root representation [`InformationRootState`] of the system for estimation.
//!
//! The linear representation can also be used for non-linear systems by using linearised models.

use nalgebra as na;
use na::{allocator::Allocator, DefaultAllocator, DMatrix, MatrixMN, MatrixN, RealField, VectorN, QR};
use na::{SimdRealField, Dim, DimName, Dynamic, U1};

use crate::linalg::cholesky::UDU;
use crate::models::{KalmanState, InformationState, KalmanEstimator, LinearObserveModel, LinearPredictModel, ExtendedLinearPredictor, ExtendedLinearObserver, Estimator};
use crate::noise::{CorrelatedNoise, CoupledNoise};
use crate::linalg::cholesky;
use crate::mine::matrix::check_positive;

/// Information State.
///
/// Linear representation as a information root state vector and the information root (upper triangular) matrix.
/// For a given [KalmanState] the information root state R.R' == inverse(X), r == R.x
#[derive(PartialEq, Clone)]
pub struct InformationRootState<N: SimdRealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Information root state vector
    pub r: VectorN<N, D>,
    /// Information root matrix (upper triangular)
    pub R: MatrixN<N, D>,
}

impl<N: RealField, D: Dim> InformationRootState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub fn new(d: D) -> InformationRootState<N, D> {
        InformationRootState {
            r: VectorN::zeros_generic(d, U1),
            R: MatrixN::zeros_generic(d, d),
        }
    }

    // pub fn init(&mut self, state: &InformationState<N, D>) -> Result<N, &'static str> {
    //     // Information Root
    //     self.R = state.I.clone().cholesky().ok_or("X not PD")?.l().transpose();
    //
    //     // Information Root state r=R*x
    //     self.r = &self.R.transpose() * &state.i;
    //
    //     Result::Ok(cholesky::UDU::UdUrcond(&state.I))
    // }
    //
    pub fn information_state(&self) -> Result<(N, InformationState<N, D>), &'static str> {
        let I = &self.R.transpose() * &self.R ; // I = R'R
        let x = self.state()?;
        let i = &I * x;

        Result::Ok((N::one(), InformationState { i, I }))
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for InformationRootState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        self.kalman_state().map(|res| res.1.x)
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for InformationRootState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str> {
        // Information Root
        let udu = UDU::new();
        self.R.copy_from(&state.X);
        let rcond = udu.UCfactor_n(&mut self.R, state.X.nrows());
        check_positive(rcond, "X not PD")?;
        let singular = udu.UTinverse(&mut self.R);
        assert!(!singular, "singular R");   // unexpected check_positive should prevent singular

        // Information Root state r=R*x
        self.r = &self.R * &state.x;

        Result::Ok(cholesky::UDU::UdUrcond(&state.X))
    }

    fn kalman_state(&self) -> Result<(N, KalmanState<N, D>), &'static str> {
        let mut RI = self.R.clone();    // Invert Cholesky factor
        let singular = UDU::new().UTinverse(&mut RI);
        if singular {
            return Result::Err("R singular");
        }

        let X = &RI * &RI.transpose();        // X = RI*RI'
        let x = RI * &self.r;

        Result::Ok((N::one(), KalmanState { x, X }))
    }
}

impl<N: RealField, D: Dim> ExtendedLinearPredictor<N, D> for InformationRootState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn predict(
        &mut self,
        x_pred: VectorN<N, D>,
        pred: &LinearPredictModel<N, D>,
        noise: &CorrelatedNoise<N, D>,
    ) -> Result<(), &'static str>
    {
        let pred_inv = pred.Fx.clone().cholesky().ok_or("Fx not PD in predict")?.inverse();

        self.predict(x_pred, &LinearPredictModel{ Fx: pred_inv}, noise).map(|_rcond| {})
    }
}

impl<N: RealField, D: DimName, ZD: DimName> ExtendedLinearObserver<N, D, ZD> for InformationRootState<N, D>
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
        s: &VectorN<N, ZD>,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &CorrelatedNoise<N, ZD>,
    ) -> Result<(), &'static str>
    {
        let noise_inv = noise.Q.clone().cholesky().ok_or("Q not PD in predict")?.inverse();
        let x = self.state()?;

        self.observe_info(obs, &noise_inv, &(s + &obs.Hx * x))
    }
}

impl<N: RealField, D: DimName> InformationRootState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, D>
{
    pub fn predict<QD: DimName>(
        &mut self,
        pred_inv: MatrixN<N, D>, // Inverse of linear prediction model Fx
        x_pred: VectorN<N, D>,
        noise: &CoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str>
        where
            DefaultAllocator: Allocator<N, QD> + Allocator<N, D, QD>
    {
        // Require Root of correlated predict noise (may be semidefinite)
        let mut Gqr = noise.G.clone();

        for qi in 0..noise.q.nrows() {
            if noise.q[qi] < N::zero() {
                return Result::Err("Predict q Not PSD");
            }
            let mut ZZ = Gqr.column_mut(qi);
            ZZ *= N::sqrt(noise.q[qi]);
        }

        // Form Augmented matrix for factorisation
        let x_size = x_pred.nrows();
        let q_size = noise.q.nrows();
        let RFxI: MatrixMN<N, D, D> = &self.R * &pred_inv;

        let mut A = DMatrix::<N>::identity(q_size + x_size, q_size + x_size); // Prefill with identity for top left and zero's in off diagonals
        let x: MatrixMN<N, D, QD> = &RFxI * &Gqr;
        A.slice_mut((q_size, 0), (x_size, q_size)).copy_from(&x);
        A.slice_mut((q_size, q_size), (x_size, x_size)).copy_from(&RFxI);

        // Calculate factorisation so we have and upper triangular R
        let qr = QR::new(A);
        // Extract the roots, junk in strict lower triangle
        let qri = qr.qr_internal();
        let res = qri.slice((q_size, q_size), (x_size, x_size)).upper_triangle();
        self.R.copy_from(&res);

        self.r = &self.R * &x_pred;    // compute r from x_pred

        return Result::Ok(UDU::new().UCrcond(&self.R));    // compute rcond of result
    }

    pub fn observe_info<ZD: DimName>(
        &mut self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise_inv: &MatrixN<N, ZD>, // Inverse of correlated noise model
        z: &VectorN<N, ZD>,
    ) -> Result<(), &'static str>
        where
            DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, ZD> + Allocator<N, ZD, D> + Allocator<N, ZD, ZD>
            + Allocator<N, D, Dynamic> + Allocator<N, ZD, Dynamic>
            + Allocator<N, ZD, U1>
            + Allocator<N, D> + Allocator<N, ZD>
    {
        let x_size = self.r.nrows();
        let z_size = z.nrows();
        // Size consistency, z to model
        if z_size != obs.Hx.nrows() {
            return Result::Err("observation and model size inconsistent");
        }

        // Form Augmented matrix for factorisation
        let mut A = DMatrix::<N>::zeros(x_size + z_size, x_size+1); // Prefill with identity for top left and zero's in off diagonals
        A.slice_mut((0, 0), (x_size, x_size)).copy_from(&self.R);
        A.slice_mut((0, x_size), (x_size, 1)).copy_from(&self.r);
        let B = noise_inv * &obs.Hx;
        A.slice_mut((x_size, 0), (z_size, x_size)).copy_from(&B);
        let C = noise_inv * z;
        A.slice_mut((x_size, x_size),(z_size,1)).copy_from(&C);

        // Calculate factorisation so we have and upper triangular R
        let qr = QR::new(A);
        // Extract the roots, junk in strict lower triangle
        let qri = qr.qr_internal();
        let res = qri.slice((0, 0), (x_size, x_size)).upper_triangle();

        // Extract the roots, junk in strict lower triangle
        self.R.copy_from(&res);
        self.r.copy_from(&qri.slice((0, x_size), (x_size,1)));

        Result::Ok(())
    }
}