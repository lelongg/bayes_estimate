#![allow(non_snake_case)]

//! Information 'square root' state estimation.
//!
//! A discrete Bayesian estimator that uses a linear information root representation [`InformationRootState`] of the system for estimation.
//!
//! The linear representation can also be used for non-linear systems by using linearised models.

use na::storage::Storage;
use na::{allocator::Allocator, DefaultAllocator, MatrixMN, MatrixN, RealField, VectorN, QR};
use na::{Dim, SimdRealField, U1};
use na::{DimAdd, DimMin, DimMinimum, DimSum};
use nalgebra as na;

use crate::linalg::cholesky;
use crate::linalg::cholesky::UdU;
use crate::matrix::check_positive;
use crate::models::{
    Estimator, ExtendedLinearObserver, InformationState, KalmanEstimator, KalmanState,
};
use crate::noise::{CorrelatedNoise, CoupledNoise};

/// Information State.
///
/// Linear representation as a information root state vector and the information root (upper triangular) matrix.
/// For a given [KalmanState] the information root state inverse(R).inverse(R)' == X, r == R.x
/// For a given [InformationState] the information root state R'.R == I, r == invserse(R).i
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
    pub fn new_zero(d: D) -> InformationRootState<N, D> {
        InformationRootState {
            r: VectorN::zeros_generic(d, U1),
            R: MatrixN::zeros_generic(d, d),
        }
    }

    pub fn init_information(&mut self, state: &InformationState<N, D>) -> Result<N, &'static str> {
        // Information Root, R'.R = I
        self.R = state
            .I
            .clone()
            .cholesky()
            .ok_or("I not PD")?
            .l()
            .transpose();

        // Information Root state, r=inv(R)'.i
        let shape = self.R.data.shape();
        let mut RI = MatrixN::identity_generic(shape.0, shape.1);
        self.R.solve_upper_triangular_mut(&mut RI);
        self.r = RI.tr_mul(&state.i);

        Result::Ok(cholesky::UdU::UdUrcond(&state.I))
    }

    pub fn information_state(&self) -> Result<InformationState<N, D>, &'static str> {
        // Information, I = R.R'
        let I = self.R.tr_mul(&self.R);
        let x = self.state()?;
        // Information state, i = I.x
        let i = &I * x;

        Result::Ok(InformationState { i, I })
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for InformationRootState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        self.kalman_state().map(|res| res.x)
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for InformationRootState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        // Information Root, inv(R).inv(R)' = X
        let udu = UdU::new();
        self.R.copy_from(&state.X);
        let rcond = udu.UCfactor_n(&mut self.R, state.X.nrows());
        check_positive(rcond, "X not PD")?;
        let singular = udu.UTinverse(&mut self.R);
        assert!(!singular, "singular R"); // unexpected check_positive should prevent singular

        // Information Root state, r=R*x
        self.r = &self.R * &state.x;

        Result::Ok(())
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        let shape = self.R.data.shape();
        let mut RI = MatrixN::identity_generic(shape.0, shape.1);
        self.R.solve_upper_triangular_mut(&mut RI);

        // Covariance X = inv(R).inv(R)'
        let X = &RI * &RI.transpose();
        // State, x= inv(R).r
        let x = RI * &self.r;

        Result::Ok(KalmanState { x, X })
    }
}

impl<N: RealField, D: Dim, ZD: Dim> ExtendedLinearObserver<N, D, ZD> for InformationRootState<N, D>
where
    DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, ZD, D>
        + Allocator<N, ZD, ZD>
        + Allocator<N, D>
        + Allocator<N, ZD>,
    D: DimAdd<ZD> + DimAdd<U1>,
    DefaultAllocator: Allocator<N, DimSum<D, ZD>, DimSum<D, U1>> + Allocator<N, DimSum<D, ZD>>,
    DimSum<D, ZD>: DimMin<DimSum<D, U1>>,
    DefaultAllocator: Allocator<N, DimMinimum<DimSum<D, ZD>, DimSum<D, U1>>>
        + Allocator<N, DimMinimum<DimSum<D, ZD>, DimSum<D, U1>>, DimSum<D, U1>>,
{
    fn observe_innovation(
        &mut self,
        s: &VectorN<N, ZD>,
        hx: &MatrixMN<N, ZD, D>,
        noise: &CorrelatedNoise<N, ZD>,
    ) -> Result<(), &'static str> {
        let udu = UdU::new();
        let mut QI = noise.Q.clone();
        let rcond = udu.UCfactor_n(&mut QI, s.nrows());
        check_positive(rcond, "Q not PD")?;
        let singular = udu.UTinverse(&mut QI);
        assert!(!singular, "singular QI"); // unexpected check_positive should prevent singular

        let x = self.state()?;
        self.observe_info(&(s + hx * x), hx, &QI)
    }
}

impl<N: RealField, D: Dim> InformationRootState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub fn predict<QD: Dim>(
        &mut self,
        x_pred: &VectorN<N, D>,
        fx: &MatrixN<N, D>,
        noise: &CoupledNoise<N, D, QD>,
    ) -> Result<(), &'static str>
    where
        D: DimAdd<QD>,
        DefaultAllocator: Allocator<N, DimSum<D, QD>, DimSum<D, QD>>
            + Allocator<N, DimSum<D, QD>>
            + Allocator<N, D, QD>
            + Allocator<N, QD>,
        DimSum<D, QD>: DimMin<DimSum<D, QD>>,
        DefaultAllocator: Allocator<N, DimMinimum<DimSum<D, QD>, DimSum<D, QD>>>
            + Allocator<N, DimMinimum<DimSum<D, QD>, DimSum<D, QD>>, DimSum<D, QD>>,
    {
        let mut Fx_inv = fx.clone();
        let invertable = Fx_inv.try_inverse_mut();
        if !invertable {
            return Err("Fx not invertable");
        }

        self.predict_inv_model(x_pred, &Fx_inv, noise)
            .map(|_rcond| {})
    }

    pub fn predict_inv_model<QD: Dim>(
        &mut self,
        x_pred: &VectorN<N, D>,
        fx_inv: &MatrixN<N, D>, // Inverse of linear prediction model Fx
        noise: &CoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str>
    where
        D: DimAdd<QD>,
        DefaultAllocator: Allocator<N, DimSum<D, QD>, DimSum<D, QD>>
            + Allocator<N, DimSum<D, QD>>
            + Allocator<N, D, QD>
            + Allocator<N, QD>,
        DimSum<D, QD>: DimMin<DimSum<D, QD>>,
        DefaultAllocator: Allocator<N, DimMinimum<DimSum<D, QD>, DimSum<D, QD>>>
            + Allocator<N, DimMinimum<DimSum<D, QD>, DimSum<D, QD>>, DimSum<D, QD>>,
    {
        // Require Root of correlated predict noise (may be semidefinite)
        let mut Gqr = noise.G.clone();

        for qi in 0..noise.q.nrows() {
            let mut ZZ = Gqr.column_mut(qi);
            ZZ *= noise.q[qi].sqrt();
        }

        // Form Augmented matrix for factorisation
        let dqd = noise.G.data.shape().0.add(noise.q.data.shape().0);
        let mut A = MatrixMN::identity_generic(dqd, dqd); // Prefill with identity for top left and zero's in off diagonals
        let RFxI: MatrixMN<N, D, D> = &self.R * fx_inv;
        let x: MatrixMN<N, D, QD> = &RFxI * &Gqr;
        let x_size = x_pred.data.shape().0;
        let q_size = noise.q.data.shape().0;
        A.generic_slice_mut((q_size.value(), 0), (x_size, q_size))
            .copy_from(&x);
        A.generic_slice_mut((q_size.value(), q_size.value()), (x_size, x_size))
            .copy_from(&RFxI);
        A.generic_slice_mut((q_size.value(), 0), (x_size, q_size))
            .copy_from(&x);
        A.generic_slice_mut((q_size.value(), q_size.value()), (x_size, x_size))
            .copy_from(&RFxI);

        // Calculate factorisation so we have and upper triangular R
        let qr = QR::new(A);
        // Extract the roots
        let r = qr.r();
        self.R
            .copy_from(&r.generic_slice((q_size.value(), q_size.value()), (x_size, x_size)));

        self.r = &self.R * x_pred; // compute r from x_pred

        Result::Ok(UdU::new().UCrcond(&self.R)) // compute rcond of result
    }

    pub fn observe_info<ZD: Dim>(
        &mut self,
        z: &VectorN<N, ZD>,
        hx: &MatrixMN<N, ZD, D>,
        noise_inv: &MatrixN<N, ZD>, // Inverse of correlated noise model
    ) -> Result<(), &'static str>
    where
        DefaultAllocator: Allocator<N, D, D>
            + Allocator<N, ZD, D>
            + Allocator<N, ZD, ZD>
            + Allocator<N, D>
            + Allocator<N, ZD>,
        D: DimAdd<ZD> + DimAdd<U1>,
        DefaultAllocator: Allocator<N, DimSum<D, ZD>, DimSum<D, U1>> + Allocator<N, DimSum<D, ZD>>,
        DimSum<D, ZD>: DimMin<DimSum<D, U1>>,
        DefaultAllocator: Allocator<N, DimMinimum<DimSum<D, ZD>, DimSum<D, U1>>>
            + Allocator<N, DimMinimum<DimSum<D, ZD>, DimSum<D, U1>>, DimSum<D, U1>>,
    {
        let x_size = self.r.data.shape().0;
        let z_size = z.data.shape().0;
        // Size consistency, z to model
        if z_size != hx.data.shape().0 {
            return Result::Err("observation and model size inconsistent");
        }

        // Form Augmented matrix for factorisation
        let xd = self.r.data.shape().0;
        let mut A = MatrixMN::identity_generic(xd.add(z.data.shape().0), xd.add(U1)); // Prefill with identity for top left and zero's in off diagonals
        A.generic_slice_mut((0, 0), (x_size, x_size))
            .copy_from(&self.R);
        A.generic_slice_mut((0, x_size.value()), (x_size, U1))
            .copy_from(&self.r);
        A.generic_slice_mut((x_size.value(), 0), (z_size, x_size))
            .copy_from(&(noise_inv * hx));
        A.generic_slice_mut((x_size.value(), x_size.value()), (z_size, U1))
            .copy_from(&(noise_inv * z));

        // Calculate factorisation so we have and upper triangular R
        let qr = QR::new(A);
        // Extract the roots
        let r = qr.r();

        // Extract the roots
        self.R.copy_from(&r.generic_slice((0, 0), (x_size, x_size)));
        self.r
            .copy_from(&r.generic_slice((0, x_size.value()), (x_size, U1)));

        Result::Ok(())
    }
}
