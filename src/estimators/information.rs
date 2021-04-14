#![allow(non_snake_case)]

//! Information state estimation.
//!
//! A discrete Bayesian estimator that uses a linear information representation [`InformationState`] of the system for estimation.
//!
//! A fundamental property of the information state is that information is additive. So if there is more information
//! about the system (such as by an observation) this can simply be added to i,I of the information state.
//!
//! The linear representation can also be used for non-linear systems by using linearised models.
//!
//! [`InformationState`]: ../models/struct.InformationState.html

use na::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, MatrixMN, MatrixN, RealField,
    VectorN, U1,
};
use nalgebra as na;

use crate::linalg::rcond;
use crate::matrix::check_positive;
use crate::models::{
    Estimator, ExtendedLinearObserver, ExtendedLinearPredictor, InformationState, KalmanEstimator,
    KalmanState,
};
use crate::noise::{CorrelatedNoise, CoupledNoise};

impl<N: RealField, D: Dim> InformationState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub fn new_zero(d: D) -> InformationState<N, D> {
        InformationState {
            i: VectorN::zeros_generic(d, U1),
            I: MatrixN::zeros_generic(d, d),
        }
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for InformationState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        KalmanEstimator::kalman_state(self).map(|r| r.x)
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for InformationState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        // Information
        self.I = state.X.clone().cholesky().ok_or("X not PD")?.inverse();
        // Information state
        self.i = &self.I * &state.x;

        Ok(())
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        // Covariance
        let X = self.I.clone().cholesky().ok_or("Y not PD")?.inverse();
        // State
        let x = &X * &self.i;

        Ok(KalmanState { x, X })
    }
}

impl<N: RealField, D: Dim> ExtendedLinearPredictor<N, D> for InformationState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn predict(
        &mut self,
        x_pred: &VectorN<N, D>,
        fx: &MatrixN<N, D>,
        noise: &CorrelatedNoise<N, D>,
    ) -> Result<(), &'static str> {
        // Covariance
        let mut X = self
            .I
            .clone()
            .cholesky()
            .ok_or("I not PD in predict")?
            .inverse();

        // Predict information matrix, and state covariance
        X.quadform_tr(N::one(), &fx, &X.clone(), N::zero());
        X += &noise.Q;

        self.init(&KalmanState {
            x: x_pred.clone_owned(),
            X,
        })?;

        Ok(())
    }
}

impl<N: RealField, D: Dim, ZD: Dim> ExtendedLinearObserver<N, D, ZD> for InformationState<N, D>
where
    DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, D, ZD>
        + Allocator<N, ZD, D>
        + Allocator<N, ZD, ZD>
        + Allocator<N, D>
        + Allocator<N, ZD>,
{
    fn observe_innovation(
        &mut self,
        s: &VectorN<N, ZD>,
        hx: &MatrixMN<N, ZD, D>,
        noise: &CorrelatedNoise<N, ZD>,
    ) -> Result<(), &'static str> {
        let x = self.state().unwrap();
        let noise_inv = noise
            .Q
            .clone()
            .cholesky()
            .ok_or("Q not PD in observe")?
            .inverse();
        let info = self.observe_info(hx, &noise_inv, &(s + hx * x));
        self.add_information(&info);

        Ok(())
    }
}

impl<N: RealField, D: Dim> InformationState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Linear information predict.
    ///
    /// Computation is through information state i,I only.
    /// Uses Fx.x instead of extended f(x)
    ///
    /// The numerical solution used is particularly flexible. It takes
    /// particular care to avoid invertibility requirements for the noise and noise coupling g,Q
    /// Therefore both zero noises and zeros in the couplings can be used.
    pub fn predict_linear<QD: Dim>(
        &mut self,
        pred_inv: MatrixN<N, D>, // Inverse of linear prediction model Fx
        noise: &CoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str>
    where
        DefaultAllocator:
            Allocator<N, QD, QD> + Allocator<N, D, QD> + Allocator<N, QD, D> + Allocator<N, QD>,
    {
        let I_shape = self.I.data.shape();

        // A = invFx'*Y*invFx ,Inverse Predict covariance
        let A = (&self.I * &pred_inv).tr_mul(&pred_inv);
        // B = G'*A*G+invQ , A in coupled additive noise space
        let mut B = (&A * &noise.G).tr_mul(&noise.G);
        for i in 0..noise.q.nrows() {
            B[(i, i)] += N::one() / noise.q[i];
        }

        // invert B ,additive noise
        B = B.cholesky().ok_or("B not PD")?.inverse();
        let rcond = rcond::rcond_symetric(&B);
        check_positive(rcond, "(G'invFx'.I.inv(Fx).G + inv(Q)) not PD")?;

        // G*invB*G' ,in state space
        self.I.quadform_tr(N::one(), &noise.G, &B, N::zero());
        // I - A* G*invB*G', information gain
        let ig = MatrixMN::identity_generic(I_shape.0, I_shape.1) - &A * &self.I;
        // Information
        self.I = &ig * &A;
        // Information state
        let y = pred_inv.tr_mul(&self.i);
        self.i = &ig * &y;

        Ok(rcond)
    }

    pub fn add_information(&mut self, information: &InformationState<N, D>) {
        self.i += &information.i;
        self.I += &information.I;
    }

    pub fn observe_info<ZD: Dim>(
        &self,
        hx: &MatrixMN<N, ZD, D>,
        noise_inv: &MatrixN<N, ZD>, // Inverse of correlated noise model
        z: &VectorN<N, ZD>,
    ) -> InformationState<N, D>
    where
        DefaultAllocator:
            Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D, ZD> + Allocator<N, ZD>,
    {
        // Observation Information
        let HxTZI = hx.tr_mul(noise_inv);
        // Calculate EIF i = Hx'*ZI*z
        let ii = &HxTZI * z;
        // Calculate EIF I = Hx'*ZI*Hx
        let II = &HxTZI * hx; // use column matrix trans(HxT)

        InformationState { i: ii, I: II }
    }
}
