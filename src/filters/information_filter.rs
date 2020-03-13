#![allow(non_snake_case)]

//! Information filter.
//!
//! A Bayesian filter that uses a linear information representation [`InformationState`] of the system for filtering.
//! The information state is simply the i,I pair the dimensions of both are the dimensions of the system.
//!
//! The Kalman state and Information state are equivilent:
//! I == inverse(X), i = I.x, since both I and X are PSD a conversion is numerically possible except with singular I or X.
//!
//! A fundamental property of the Information state is that Information is additive. So if there is more information
//! about the system (such as by an observation) this can simply be added to i,I Information state.
//!
//! The linear information state representation can also be used for non-linear system by using linearised
//! forms of the system model.
//!
//! [`InformationState`]: ../models/struct.InformationState.html

use na::{allocator::Allocator, DefaultAllocator, Dim, MatrixMN, MatrixN, RealField, U1, VectorN};
use na::base::storage::Storage;
use nalgebra as na;

use crate::linalg::cholesky::UDU;
use crate::mine::matrix::{check_positive, prod_spd, prod_spdt};
use crate::models::{AdditiveCorrelatedNoise, AdditiveNoise, InformationState, KalmanEstimator, KalmanState, LinearEstimator, LinearObserveModel, LinearPredictModel, LinearPredictor};


impl<N: RealField, D: Dim> InformationState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    pub fn new(d: D) -> InformationState<N, D> {
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

impl<N: RealField, D: Dim> LinearEstimator<N> for InformationState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for InformationState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str> {
        // Information
        self.I = state.X.clone();
        let rcond = UDU::new().UdUinversePD(&mut self.I);
        check_positive(rcond, "X not PD")?;
        // Information state
        self.i = &self.I * &state.x;

        Result::Ok(rcond)
    }

    fn state(&self) -> Result<(N, KalmanState<N, D>), &'static str> {
        // Covariance
        let mut X = self.I.clone();
        let rcond = UDU::new().UdUinversePD(&mut X);
        check_positive(rcond, "Y not PD")?;
        // State
        let x = &X * &self.i;

        Result::Ok((rcond, KalmanState { x, X }))
    }
}

impl<N: RealField, D: Dim, QD: Dim> LinearPredictor<N, D, QD> for InformationState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, QD, QD> + Allocator<N, D, QD> + Allocator<N, QD, D>
        + Allocator<N, D> + Allocator<N, QD>
{
    fn predict(&mut self, pred: &LinearPredictModel<N, D>, x_pred: VectorN<N, D>, noise: &AdditiveCorrelatedNoise<N, D, QD>) -> Result<N, &'static str> {
        // Covariance
        let mut X = self.I.clone();
        let rcond = UDU::new().UdUinversePD(&mut X);
        check_positive(rcond, "I not PD in predict")?;

        // Predict information matrix, and state covariance
        X = prod_spd(&pred.Fx, &X);
        X += prod_spd(&noise.G, &MatrixN::from_diagonal(&noise.q));

        self.init(&KalmanState { x: x_pred, X })
    }
}

impl<N: RealField, D: Dim> InformationState<N, D>
    where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /* Linear information predict
     *  Computation is through information state i,I only
     *  Uses x(k+1|k) = Fx * x(k|k) instead of extended x(k+1|k) = f(x(k|k))
     * Requires i(k|k), I(k|k)
     * Predicts i(k+1|k), I(k+1|k)
     *
     * The numerical solution used is particularly flexible. It takes
     * particular care to avoid invertibility requirements for the noise and noise coupling g,Q
     * Therefore both zero noises and zeros in the couplings can be used
     */
    pub fn predict_linear_invertable<QD: Dim>(&mut self, pred_inv: &LinearPredictModel<N, D>, noise: &AdditiveCorrelatedNoise<N, D, QD>) -> Result<N, &'static str>
        where DefaultAllocator: Allocator<N, QD, QD> + Allocator<N, D, QD> + Allocator<N, QD, D> + Allocator<N, QD>
    {
        let I_shape = self.I.data.shape();

        // A = invFx'*Y*invFx ,Inverse Predict covariance
        let A = prod_spdt(&pred_inv.Fx, &self.I);
        // B = G'*A*G+invQ , A in coupled additive noise space
        let mut B = prod_spdt(&noise.G, &A);
        for i in 0..noise.q.nrows()
        {
            if noise.q[i] < N::zero() {    // allow PSD q, let infinity propagate into B
                return Result::Err("q not PSD");
            }
            B[(i, i)] += N::one() / noise.q[i];
        }

        // invert B ,additive noise
        let rcond = UDU::new().UdUinversePDignoreInfinity(&mut B);
        check_positive(rcond, "(G'invFx'.I.inv(Fx).G + inv(Q)) not PD")?;

        // G*invB*G' ,in state space
        self.I = prod_spd(&noise.G, &B);
        // I - A* G*invB*G' ,information gain
        let ig = MatrixMN::identity_generic(I_shape.0, I_shape.1) - &A * &self.I;
        // Information
        self.I = &ig * &A;
        // Information state
        let y = pred_inv.Fx.transpose() * &self.i;
        self.i = &ig * &y;

        Result::Ok(rcond)
    }

    pub fn add_information(&mut self, information: &InformationState<N, D>) {
        self.i += &information.i;
        self.I += &information.I;
    }

    pub fn observe_innovation_co<ZD: Dim, ZQD: Dim>(&self, obs: &LinearObserveModel<N, D, ZD>, noise: &AdditiveCorrelatedNoise<N, ZD, ZQD>,
                                                    s: &VectorN<N, ZD>, x: &VectorN<N, D>) -> Result<(N, InformationState<N, D>), &'static str>
        where
            DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D, ZD>
            + Allocator<N, ZQD, ZQD> + Allocator<N, ZQD, ZD> + Allocator<N, ZD, ZQD>
            + Allocator<N, ZD> + Allocator<N, ZQD>
    {
        let zz = s + &obs.Hx * x;        // Strange EIF observation object

        // Observation Information, TODO use inverse directly on q, D factors
        let mut ZI = prod_spd(&noise.G, &MatrixN::from_diagonal(&noise.q));
        let rcond = UDU::new().UdUinversePD(&mut ZI);
        check_positive(rcond, "Z not PD")?;

        let HxTZI = obs.Hx.transpose() * ZI;
        // Calculate EIF i = Hx'*ZI*zz
        let i = &HxTZI * zz;
        // Calculate EIF I = Hx'*ZI*Hx
        let I = &HxTZI * &obs.Hx;                // use column matrix trans(HxT)

        Result::Ok((rcond, InformationState { i, I }))
    }

    pub fn observe_innovation_un<ZD: Dim, ZQD: Dim>(&self, obs: &LinearObserveModel<N, D, ZD>, noise: &AdditiveNoise<N, ZQD>, s: &VectorN<N, ZD>, x: &VectorN<N, D>)
                                                    -> Result<(N, InformationState<N, D>), &'static str>
        where
            DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, D, ZD> + Allocator<N, ZD> + Allocator<N, ZQD>
    {
        let zz = s + &obs.Hx * x;        // Strange EIF observation object

        // Observation Information
        let rcond = UDU::UdUrcond_vec(&noise.q);
        check_positive(rcond, "Zv not PD")?;
        // HxTZI = Hx'*inverse(Z)
        let mut HxTZI = obs.Hx.transpose();
        for w in 0..noise.q.nrows() {
            let mut HxTZI_w = HxTZI.column_mut(w);
            HxTZI_w *= N::one() / noise.q[w];
        }

        // Calculate EIF i = Hx'*ZI*zz
        let i = &HxTZI * zz;
        // Calculate EIF I = Hx'*ZI*Hx
        let I = &HxTZI * &obs.Hx;                // use column matrix trans(HxT)

        Result::Ok((rcond, InformationState { i, I }))
    }
}
