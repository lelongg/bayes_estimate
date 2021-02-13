#![allow(non_snake_case)]

//! UD 'square root' state estimation.
//!
//! A discrete Bayesian filter that uses a 'square root' factorisation of the Kalman state representation [`KalmanState`] of the system for estimation.
//!
//! The state covariance is represented as a U.d.U' factorisation, where U is upper triangular matrix (0 diagonal) and
//! d is a diagonal vector.
//! Numerically the this 'square root' factorisation is advantageous as condition of when inverting is improved by the square root.
//!
//! The linear representation can also be used for non-linear system by using linearised forms of the system model.
//!
//! [`KalmanState`]: ../models/struct.KalmanState.html

use na::base::storage::Storage;
use na::{allocator::Allocator, DefaultAllocator};
use na::{DMatrix, Dynamic, MatrixMN, MatrixN, VectorN, U1};
use na::{Dim, RealField};
use nalgebra as na;

use crate::linalg::cholesky::UDU;
use crate::mine::matrix;
use crate::models::{AdditiveCorrelatedNoise, AdditiveNoise, KalmanEstimator, KalmanState, LinearObserveModel, LinearPredictModel, AdditiveCoupledNoise, Estimator};

/// UD State representation.
///
/// Linear representation as a state vector and 'square root' factorisation of the state covariance matrix.
///
/// The state covariance X is factorised with a modified Cholesky factorisation so U.d.U' == X, where U is upper triangular matrix (0 diagonal) and
/// d is a diagonal vector. U and d are packed into a single UD Matrix, the lower Triangle ist not part of state representation.
pub struct UDState<N: RealField, D: Dim, XUD: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, XUD> + Allocator<N, D>,
{
    /// State vector
    pub x: VectorN<N, D>,
    /// UD matrix representation of state covariance
    pub UD: MatrixMN<N, D, XUD>,
    // UDU instance for factorisations
    udu: UDU<N>,
}

impl<N: RealField, D: Dim, XUD: Dim> UDState<N, D, XUD>
where
    DefaultAllocator:
        Allocator<N, D, D> + Allocator<N, D, XUD> + Allocator<N, D> + Allocator<N, XUD>,
{
    /// Create a UDState for given state dimensions.
    ///
    /// d is the size of states vector and rows in UD.
    ///
    /// XUD is the number of columns in UD. This will be large then d to accommodate the matrix
    /// dimensions of the prediction model. The extra columns are used for the prediction computation.
    pub fn new(d: D, xud: XUD) -> Self {
        assert!(xud.value() >= d.value(), "xud must be >= d");

        UDState {
            UD: MatrixMN::<N, D, XUD>::zeros_generic(d, xud),
            x: VectorN::zeros_generic(d, U1),
            udu: UDU::new(),
        }
    }

    pub fn predict<QD: Dim>(
        &mut self,
        pred: &LinearPredictModel<N, D>,
        x_pred: VectorN<N, D>,
        noise: &AdditiveCoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str>
        where
            DefaultAllocator: Allocator<N, D, QD> + Allocator<N, QD>
    {
        let mut scratch = self.new_predict_scratch();
        self.predict_use_scratch(&mut scratch, pred, x_pred, noise)
    }

    /// Implement observe using sequential observation updates.
    ///
    /// Uncorrelated observations are applied sequentially in the order they appear in z.
    ///
    /// Therefore the model of each observation needs to be computed sequentially. Generally this
    /// is inefficient and observe (UD_sequential_observe_model&) should be used instead
    //// Return: Minimum rcond of all sequential observe
    pub fn observe_innovation<ZD: Dim>(
        &mut self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &AdditiveNoise<N, ZD>,
        s: &VectorN<N, ZD>,
    ) -> Result<N, &'static str>
    where
        DefaultAllocator: Allocator<N, ZD, D> + Allocator<N, ZD>
    {
        let mut scratch = self.new_observe_scratch();

        // Predict UD from model
        UDState::observe_innovation_use_scratch(self, &mut scratch, obs, noise, s)
    }

}

impl<N: RealField, D: Dim, XUD: Dim> Estimator<N, D> for UDState<N, D, XUD>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, XUD> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        KalmanEstimator::state(self).map(|r| r.1.x)
    }
}

impl<N: RealField, D: Dim, XUD: Dim> KalmanEstimator<N, D> for UDState<N, D, XUD>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, XUD> + Allocator<N, D>,
{
    /// Initialise the UDState with a KalmanState.
    ///
    /// The covariance matrix X is factorised into a U.d.U' as a UD matrix.
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str> {
        self.x = state.x.clone();

        // Factorise X into UD
        let rows = self.UD.nrows();
        matrix::copy_from(&mut self.UD.columns_mut(0, rows), &state.X);
        let rcond = self.udu.UdUfactor_variant2(&mut self.UD, rows);
        matrix::check_non_negativ(rcond, "X not PSD")?;

        Ok(self.udu.one)
    }

    /// Derive the KalmanState from the UDState.
    ///
    /// The covariance matrix X is recomposed from U.d.U' in the UD matrix.
    fn state(&self) -> Result<(N, KalmanState<N, D>), &'static str> {
        // assign elements of common left block of M into X
        let x_shape = self.x.data.shape().0;
        let mut X = matrix::as_zeros((x_shape, x_shape));
        matrix::copy_from(&mut X, &self.UD.columns(0, self.UD.nrows()));
        UDU::UdUrecompose(&mut X);

        Ok((
            self.udu.one,
            KalmanState {
                x: self.x.clone(),
                X,
            },
        ))
    }
}

/// Prediction Scratch.
///
/// Provides temporary variables for prediction calculation.
pub struct PredictScratch<N: RealField, XUD: Dim>
where
    DefaultAllocator: Allocator<N, XUD>,
{
    pub d: VectorN<N, XUD>,
    pub dv: VectorN<N, XUD>,
    pub v: VectorN<N, XUD>,
}

/// Observe Scratch.
///
/// Provides temporary variables for observe calculation.
pub struct ObserveScratch<N: RealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D>,
{
    pub w: VectorN<N, D>,
    pub a: VectorN<N, D>,
    pub b: VectorN<N, D>,
}

impl<N: RealField, D: Dim, XUD: Dim> UDState<N, D, XUD>
where
    DefaultAllocator:
        Allocator<N, D, D> + Allocator<N, D, XUD> + Allocator<N, D> + Allocator<N, XUD>,
{
    pub fn predict_use_scratch<QD: Dim>(
        &mut self,
        scratch: &mut PredictScratch<N, XUD>,
        pred: &LinearPredictModel<N, D>,
        x_pred: VectorN<N, D>,
        noise: &AdditiveCoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str>
    where
        DefaultAllocator: Allocator<N, D, QD> + Allocator<N, QD>,
    {
        self.x = x_pred;

        // Predict UD from model
        let rcond = UDState::predictGq(self, scratch, &pred.Fx, &noise.G, &noise.q);

        matrix::check_non_negativ(rcond, "X not PSD")
    }

    pub fn observe_innovation_use_scratch<ZD: Dim>(
        &mut self,
        scratch: &mut ObserveScratch<N, D>,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &AdditiveNoise<N, ZD>,
        s: &VectorN<N, ZD>,
    ) -> Result<N, &'static str>
    where
        DefaultAllocator:
            Allocator<N, ZD, D> + Allocator<N, ZD>
    {
        let z_size = s.nrows();

        // Apply observations sequentially as they are decorrelated
        let mut rcondmin = N::max_value();

        for o in 0..z_size {
            // Check noise precondition
            if noise.q[o] < self.udu.zero {
                return Err("Zv not PSD in observe");
            }
            // Update UD and extract gain
            let mut S = self.udu.zero;
            obs.Hx.row(o).transpose_to(&mut scratch.a);
            let rcond = UDState::observeUD(self, scratch, &mut S, noise.q[o]);
            if rcond < rcondmin {
                rcondmin = rcond;
            }
            // State update using normalised non-linear innovation
            scratch.w *= s[0];
            self.x += &scratch.w;
        }
        Ok(rcondmin)
    }

    /// Special Linear Hx observe for correlated Z.
    ///
    /// Z must be PD and will be decorrelated
    /// Applies observations sequentially in the order they appear in z
    /// Creates temporary Vec and Matrix to decorrelate z,Z
    ///
    /// Return: Minimum rcond of all sequential observe
    pub fn observe_decorrelate<ZD: Dim>(
        &mut self,
        obs: &LinearObserveModel<N, D, ZD>,
        noise: &AdditiveCorrelatedNoise<N, ZD>,
        z: &VectorN<N, ZD>,
    ) -> Result<N, &'static str>
    where
        DefaultAllocator: Allocator<N, ZD, ZD>
            + Allocator<N, ZD, D>
            + Allocator<N, ZD>
    {
        let x_size = self.x.nrows();
        let z_size = z.nrows();

        let mut scratch = self.new_observe_scratch();

        let mut zp = &obs.Hx * &self.x;
        let mut zpdecol = zp.clone();

        // Observation prediction and normalised observation
        let mut GIHx = obs.Hx.clone();
        {
            // Solve G* GIHx = Hx for GIHx in-place
            for j in 0..x_size {
                for i in (0..z_size).rev() {
                    for k in i + 1..z_size {
                        let t = noise.Q[(i, k)] * GIHx[(k, j)];
                        GIHx[(i, j)] -= t;
                    }
                }
            }

            // Solve G zp~ = z, G z~ = z  for zp~,z~ in-place
            for i in (0..z_size).rev() {
                for k in i + 1..z_size {
                    let zpt = noise.Q[(i, k)] * zp[k];
                    zp[i] -= zpt;
                    let zpdt = noise.Q[(i, k)] * zpdecol[k];
                    zpdecol[i] -= zpdt;
                }
            }
        }

        // Apply observations sequential as they are decorrelated
        let mut rcondmin = N::max_value();
        for o in 0..z_size {
            // Update UD and extract gain
            let mut S = self.udu.zero;
            GIHx.row(o).transpose_to(&mut scratch.a);
            let rcond = UDState::observeUD(self, &mut scratch, &mut S, noise.Q[(o, o)]);
            matrix::check_positive(rcond, "S not PD in observe")?; // -1 implies S singular
            if rcond < rcondmin {
                rcondmin = rcond;
            }
            // State update using linear innovation
            let s = z[o] - zpdecol[o];
            self.x += &scratch.w * s;
            println!("{:?} {:?} {:?}", self.x, scratch.w, s)
        }
        Ok(rcondmin)
    }
}

impl<N: RealField, D: Dim, XUD: Dim> UDState<N, D, XUD>
where
    DefaultAllocator:
        Allocator<N, D, D> + Allocator<N, D, XUD> + Allocator<N, XUD> + Allocator<N, D>,
{
    pub fn new_predict_scratch(&self) -> PredictScratch<N, XUD> {
        let ud_col_vec_shape = (self.UD.data.shape().1, U1);
        PredictScratch {
            d: matrix::as_zeros(ud_col_vec_shape),
            dv: matrix::as_zeros(ud_col_vec_shape),
            v: matrix::as_zeros(ud_col_vec_shape),
        }
    }

    pub fn new_observe_scratch(&self) -> ObserveScratch<N, D> {
        let x_vec_shape = self.x.data.shape();
        ObserveScratch {
            w: matrix::as_zeros(x_vec_shape),
            a: matrix::as_zeros(x_vec_shape),
            b: matrix::as_zeros(x_vec_shape),
        }
    }
}

impl<N: RealField> UDState<N, Dynamic, Dynamic> {
    pub fn new_dynamic(state: KalmanState<N, Dynamic>, q_maxsize: usize) -> Self {
        let x_size = state.x.nrows();
        UDState {
            x: state.x,
            UD: DMatrix::zeros(x_size, x_size + q_maxsize),
            udu: UDU::new(),
        }
    }
}

impl<N: RealField, D: Dim, XUD: Dim> UDState<N, D, XUD>
where
    DefaultAllocator:
        Allocator<N, D, D> + Allocator<N, D, XUD> + Allocator<N, D> + Allocator<N, XUD>,
{
    /// MWG-S prediction from Bierman p.132
    ///
    /// q can have order less then x and a matching G so GqG' has order of x
    ///
    /// Return: reciprocal condition number, -1 if negative, 0 if semi-definite (including zero)
    fn predictGq<UD: Dim>(
        &mut self,
        scratch: &mut PredictScratch<N, XUD>,
        Fx: &MatrixN<N, D>,
        G: &MatrixMN<N, D, UD>,
        q: &VectorN<N, UD>,
    ) -> N
    where
        DefaultAllocator: Allocator<N, D, UD> + Allocator<N, UD>,
    {
        let n = self.x.nrows();
        let Nq = q.nrows();
        let NN = n + Nq;

        if n > 0
        // Simplify reverse loop termination
        {
            // Augment d with q, UD with G
            for i in 0..Nq
            // 0..Nq-1
            {
                scratch.d[i + n] = q[i];
            }
            for j in 0..n
            // 0..n-1
            {
                for i in 0..Nq {
                    // 0..Nq-1
                    self.UD[(j, i + n)] = G[(j, i)];
                }
            }

            // U=Fx*U and diagonals retrieved
            for j in (1..n).rev()
            // n-1..1
            {
                // Prepare d(0)..d(j) as temporary
                for i in 0..=j {
                    // 0..j
                    scratch.d[i] = self.UD[(i, j)];
                }

                // Lower triangle of UD is implicitly empty
                for i in 0..n
                // 0..n-1
                {
                    self.UD[(i, j)] = Fx[(i, j)];
                    for k in 0..j {
                        // 0..j-1
                        self.UD[(i, j)] += Fx[(i, k)] * scratch.d[k];
                    }
                }
            }
            scratch.d[0] = self.UD[(0, 0)];

            //  Complete U = Fx*U
            for j in 0..n
            // 0..n-1
            {
                self.UD[(j, 0)] = Fx[(j, 0)];
            }

            // The MWG-S algorithm on UD transpose
            for j in (0..n).rev() {
                // n-1..0
                let mut e = self.udu.zero;
                for k in 0..NN
                // 0..N-1
                {
                    scratch.v[k] = self.UD[(j, k)];
                    scratch.dv[k] = scratch.d[k] * scratch.v[k];
                    e += scratch.v[k] * scratch.dv[k];
                }
                // Check diagonal element
                if e > self.udu.zero {
                    // Positive definite
                    self.UD[(j, j)] = e;

                    let diaginv = self.udu.one / e;
                    for k in 0..j
                    // 0..j-1
                    {
                        e = self.udu.zero;
                        for i in 0..NN {
                            // 0..N-1
                            e += self.UD[(k, i)] * scratch.dv[i];
                        }
                        e *= diaginv;
                        self.UD[(j, k)] = e;

                        for i in 0..NN {
                            // 0..N-1
                            self.UD[(k, i)] -= e * scratch.v[i]
                        }
                    }
                }
                //PD
                else if e == self.udu.zero {
                    // Possibly semi-definite, check not negative
                    self.UD[(j, j)] = e;

                    // 1 / e is infinite
                    for k in 0..j
                    // 0..j-1
                    {
                        for i in 0..NN
                        // 0..N-1
                        {
                            e = self.UD[(k, i)] * scratch.dv[i];
                            if e != self.udu.zero {
                                return self.udu.minus_one;
                            }
                        }
                        // UD(j,k) unaffected
                    }
                }
                //PD
                else {
                    // Negative
                    return self.udu.minus_one;
                }
            } //MWG-S loop

            // Transpose and Zero lower triangle
            for j in 1..n
            // 0..n-1
            {
                for i in 0..j {
                    self.UD[(i, j)] = self.UD[(j, i)];
                    self.UD[(j, i)] = self.udu.zero; // Zeroing unnecessary as lower only used as a scratch
                }
            }
        }

        // Estimate the reciprocal condition number from upper triangular part
        UDU::UdUrcond(&self.UD)
    }

    /// Linear UD factorisation update from Bierman p.100
    ///
    /// # Input
    ///  h observation coefficients
    ///  r observation variance
    /// # Output
    ///  gain  observation Kalman gain
    ///  alpha observation innovation variance
    /// # Variables with physical significance
    ///  gamma becomes covariance of innovation
    /// # Precondition
    ///  r is PSD (not checked)
    /// # Return
    ///  reciprocal condition number of UD, -1 if alpha singular (negative or zero)
    fn observeUD(&mut self, scratch: &mut ObserveScratch<N, D>, alpha: &mut N, r: N) -> N {
        let n = self.UD.nrows();
        // a(n) is U'a
        // b(n) is Unweighted Kalman gain

        // Compute b = DU'h, a = U'h
        for j in (1..n).rev()
        // n-1..1
        {
            for k in 0..j
            // 0..j-1
            {
                let t = self.UD[(k, j)] * scratch.a[k];
                scratch.a[j] += t;
            }
            scratch.b[j] = self.UD[(j, j)] * scratch.a[j];
        }
        scratch.b[0] = self.UD[(0, 0)] * scratch.a[0];

        // Update UD(0,0), d(0) modification
        *alpha = r + scratch.b[0] * scratch.a[0];
        if *alpha <= self.udu.zero {
            return self.udu.minus_one;
        }
        let mut gamma = self.udu.one / *alpha;
        self.UD[(0, 0)] *= r * gamma;
        // Update rest of UD and gain b
        for j in 1..n {
            // 1..n-1 {
            // d modification
            let alpha_jm1 = *alpha; // alpha at j-1
            *alpha += scratch.b[j] * scratch.a[j];
            let lamda = -scratch.a[j] * gamma;
            if *alpha <= self.udu.zero {
                return self.udu.minus_one;
            }
            gamma = self.udu.one / *alpha;
            self.UD[(j, j)] *= alpha_jm1 * gamma;
            // U modification
            for i in 0..j
            // 0..j-1
            {
                let UD_jm1 = self.UD[(i, j)];
                self.UD[(i, j)] = UD_jm1 + lamda * scratch.b[i];
                let t = scratch.b[j] * UD_jm1;
                scratch.b[i] += t;
            }
        }
        // Update gain from b
        scratch.w.copy_from(&(&scratch.b * gamma));

        // Estimate the reciprocal condition number from upper triangular part
        UDU::UdUrcond(&self.UD)
    }
}
