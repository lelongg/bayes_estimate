#![allow(non_snake_case)]

//! Information state estimation.
//!
//! A discrete Bayesian estimator that uses a linear information representation [`InformationState`] of the system for estimation.
//! The information state is simply the i,I pair the dimensions of both are the dimensions of the system.
//!
//! The Kalman state and Information state are equivalent:
//! I == inverse(X), i = I.x, since both I and X are PSD a conversion is numerically possible except with singular I or X.
//!
//! A fundamental property of the Information state is that Information is additive. So if there is more information
//! about the system (such as by an observation) this can simply be added to i,I Information state.
//!
//! The linear information state representation can also be used for non-linear system by using linearised forms of the system model.
//!
//! [`InformationState`]: ../models/struct.InformationState.html

use na::base::storage::Storage;
use na::{allocator::Allocator, DefaultAllocator, Dim, MatrixMN, MatrixN, RealField, VectorN, U1};
use nalgebra as na;

use crate::linalg::cholesky::UDU;
use crate::mine::matrix::{check_positive};
use crate::models::{InformationState, KalmanEstimator, KalmanState, LinearObserveModel, LinearPredictModel, LinearPredictor, Estimator};
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

    pub fn init_information(&mut self, information: &InformationState<N, D>) {
        self.i = information.i.clone();
        self.I = information.I.clone();
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for InformationState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        KalmanEstimator::kalman_state(self).map(|r| r.1.x)
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for InformationState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str> {
        // Information
        self.I = state.X.clone();
        let rcond = UDU::new().UdUinversePD(&mut self.I);
        check_positive(rcond, "X not PD")?;
        // Information state
        self.i = &self.I * &state.x;

        Ok(rcond)
    }

    fn kalman_state(&self) -> Result<(N, KalmanState<N, D>), &'static str> {
        // Covariance
        let mut X = self.I.clone();
        let rcond = UDU::new().UdUinversePD(&mut X);
        check_positive(rcond, "Y not PD")?;
        // State
        let x = &X * &self.i;

        Ok((rcond, KalmanState { x, X }))
    }
}

impl<N: RealField, D: Dim> LinearPredictor<N, D> for InformationState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn predict(
        &mut self,
        pred: &LinearPredictModel<N, D>,
        x_pred: VectorN<N, D>,
        noise: &CorrelatedNoise<N, D>,
    ) -> Result<(), &'static str> {
        // Covariance
        let mut X = self.I.clone();
        let rcond = UDU::new().UdUinversePD(&mut X);
        check_positive(rcond, "I not PD in predict")?;

        // Predict information matrix, and state covariance
        X.quadform_tr(N::one(), &pred.Fx, &X.clone(), N::zero());
        X += &noise.Q;

        self.init(&KalmanState { x: x_pred, X })?;

        return Ok(())
    }
}

impl<N: RealField, D: Dim> InformationState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Linear information predict.
    ///
    /// Computation is through information state i,I only.
    /// Uses x(k+1|k) = Fx * x(k|k) instead of extended x(k+1|k) = f(x(k|k))
    ///
    /// The numerical solution used is particularly flexible. It takes
    /// particular care to avoid invertibility requirements for the noise and noise coupling g,Q
    /// Therefore both zero noises and zeros in the couplings can be used.
    pub fn predict_linear_invertible<QD: Dim>(
        &mut self,
        pred_inv: &LinearPredictModel<N, D>,
        noise: &CoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str>
    where
        DefaultAllocator:
            Allocator<N, QD, QD> + Allocator<N, D, QD> + Allocator<N, QD, D> + Allocator<N, QD>,
    {
        let I_shape = self.I.data.shape();

        // A = invFx'*Y*invFx ,Inverse Predict covariance
        let A = (&self.I * &pred_inv.Fx).transpose() * &pred_inv.Fx;
        // B = G'*A*G+invQ , A in coupled additive noise space
        let mut B = (&A * &noise.G).transpose() * &noise.G;
        for i in 0..noise.q.nrows() {
            if noise.q[i] < N::zero() {
                // allow PSD q, let infinity propagate into B
                return Err("q not PSD");
            }
            B[(i, i)] += N::one() / noise.q[i];
        }

        // invert B ,additive noise
        let rcond = UDU::new().UdUinversePDignoreInfinity(&mut B);
        check_positive(rcond, "(G'invFx'.I.inv(Fx).G + inv(Q)) not PD")?;

        // G*invB*G' ,in state space
        self.I.quadform_tr(N::one(), &noise.G, &B, N::zero());
        // I - A* G*invB*G' ,information gain
        let ig = MatrixMN::identity_generic(I_shape.0, I_shape.1) - &A * &self.I;
        // Information
        self.I = &ig * &A;
        // Information state
        let y = pred_inv.Fx.transpose() * &self.i;
        self.i = &ig * &y;

        Ok(rcond)
    }

    pub fn add_information(&mut self, information: &InformationState<N, D>) {
        self.i += &information.i;
        self.I += &information.I;
    }

    pub fn observe_info<ZD: Dim>(
        &self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise_inverted: &CorrelatedNoise<N, ZD>,
        z: &VectorN<N, ZD>
    ) -> InformationState<N, D>
    where
        DefaultAllocator: Allocator<N, ZD, ZD>
            + Allocator<N, ZD, D>
            + Allocator<N, D, ZD>
            + Allocator<N, ZD>
    {
        // Observation Information
        let HxTZI = obs.Hx.transpose() * &noise_inverted.Q;
        // Calculate EIF i = Hx'*ZI*z
        let ii = &HxTZI * z;
        // Calculate EIF I = Hx'*ZI*Hx
        let II = &HxTZI * &obs.Hx; // use column matrix trans(HxT)

        InformationState { i: ii, I: II }
    }
}
