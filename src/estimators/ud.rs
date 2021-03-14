#![allow(non_snake_case)]

//! UD 'square root' state estimation.
//!
//! A discrete Bayesian estimator that uses a 'square root' factorisation of the Kalman state representation [`UDState`] of the system for estimation.
//!
//! The linear representation can also be used for non-linear system by using linearised forms of the system model.

use nalgebra as na;
use na::{allocator::Allocator, DefaultAllocator, storage::Storage};
use na::{Dim, RealField, U1, MatrixMN, MatrixN, VectorN};

use crate::linalg::cholesky::UDU;
use crate::matrix;
use crate::models::{KalmanEstimator, KalmanState, Estimator};
use crate::noise::{UncorrelatedNoise, CoupledNoise, CorrelatedNoise};
use nalgebra::{DimAdd, DimSum};
use crate::matrix::{check_non_negativ};


/// UD State representation.
///
/// Linear representation as a state vector and 'square root' factorisation of the state covariance matrix.
/// Numerically the this 'square root' factorisation is advantageous as conditioning for inverting is improved by the square root.
///
/// The state covariance is represented as a U.d.U' factorisation, where U is upper triangular matrix (0 diagonal) and d is a diagonal vector.
/// U and d are packed into a single UD Matrix, the lower Triangle ist not part of state representation.
pub struct UDState<N: RealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{


    /// State vector
    pub x: VectorN<N, D>,
    /// UD matrix representation of state covariance
    pub UD: MatrixN<N, D>,
    // UDU instance for factorisations
    udu: UDU<N>,
}

impl<N: RealField, D: Dim> UDState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Create a UDState for given state dimensions.
    ///
    /// D is the size of states vector and rows in UD.
    pub fn new(UD: MatrixN<N, D>, x: VectorN<N, D>) -> Self {
        assert!(x.nrows() == UD.nrows(), "x rows must be == UD rows");

        UDState {
            UD,
            x,
            udu: UDU::new(),
        }
    }

    /// Create a UDState for given state dimensions.
    ///
    /// d is the size of states vector and rows in UD.
    pub fn new_zero(d: D) -> Self {
        UDState {
            UD: MatrixN::<N, D>::zeros_generic(d, d),
            x: VectorN::zeros_generic(d, U1),
            udu: UDU::new(),
        }
    }

    pub fn predict<QD: Dim>(
        &mut self,
        fx: &MatrixN<N, D>,
        x_pred: &VectorN<N, D>,
        noise: &CoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str>
        where
            D: DimAdd<QD>,
            DefaultAllocator: Allocator<N, DimSum<D, QD>, U1> + Allocator<N, D, QD> + Allocator<N, QD>
    {
        let mut scratch = self.new_predict_scratch(noise.q.data.shape().0);
        self.predict_use_scratch(&mut scratch, x_pred, fx, noise)
    }

    /// Implement observe using sequential observation updates.
    ///
    /// Uncorrelated observations are applied sequentially in the order they appear in z.
    ///
    //// Return: Minimum rcond of all sequential observe
    pub fn observe_innovation<ZD: Dim>(
        &mut self,
        s: &VectorN<N, ZD>,
        hx: &MatrixMN<N, ZD, D>,
        noise: &UncorrelatedNoise<N, ZD>,
    ) -> Result<N, &'static str>
        where
            DefaultAllocator: Allocator<N, ZD, D> + Allocator<N, ZD>
    {
        let mut scratch = self.new_observe_scratch();
        UDState::observe_innovation_use_scratch(self, &mut scratch, s, hx, noise)
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for UDState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        KalmanEstimator::kalman_state(self).map(|r| r.x)
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for UDState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Initialise the UDState with a KalmanState.
    ///
    /// The covariance matrix X is factorised into a U.d.U' as a UD matrix.
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        self.x = state.x.clone();

        // Factorise X into UD
        let rows = self.UD.nrows();
        self.UD.copy_from(&state.X);
        let rcond = self.udu.UdUfactor_variant2(&mut self.UD, rows);
        matrix::check_non_negativ(rcond, "X not PSD")?;

        Ok(())
    }

    /// Derive the KalmanState from the UDState.
    ///
    /// The covariance matrix X is recomposed from U.d.U' in the UD matrix.
    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        // assign elements of common left block of M into X
        let x_shape = self.x.data.shape().0;

        let mut X = self.UD.columns_generic(0, x_shape).into_owned();
        UDU::UdUrecompose(&mut X);

        Ok(KalmanState { x: self.x.clone(), X })
    }
}

/// Additive noise.
///
/// Noise represented as a the noise covariance as a factorised UdU' matrix.
pub struct CorrelatedFactorNoise<N: RealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D>
{
    /// Noise covariance
    pub UD: MatrixN<N, D>
}

impl<N: RealField, D: Dim> CorrelatedFactorNoise<N, D>
    where
        DefaultAllocator: Allocator<N, D, D>,
{
    /// Creates a CorrelatedFactorNoise from an CorrelatedNoise.
    /// The CorrelatedNoise must be PSD.
    pub fn from_correlated(correlated: &CorrelatedNoise<N, D>) -> Result<Self, &'static str> {
        let udu = UDU::new();
        let mut ud: MatrixN<N, D> = correlated.Q.clone_owned();
        let rcond = udu.UdUfactor_variant2(&mut ud, correlated.Q.nrows());
        check_non_negativ(rcond, "Q not PSD")?;

        Ok(CorrelatedFactorNoise{ UD: ud })
    }
}

impl<N: RealField, D: Dim> UDState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Special Linear 'hx' observe for correlated factorised noise.
    ///
    /// Observation predictions are made with the linear 'hx'. This allows the observation noise
    /// to be decorrelated. Observations can then be applied for each element in the order they appear in z.
    ///
    /// Return: Minimum rcond of all sequential observations
    pub fn observe_linear_correlated<ZD: Dim>(
        &mut self,
        z: &VectorN<N, ZD>,
        hx: &MatrixMN<N, ZD, D>,
        h_normalize: fn(&mut VectorN<N, ZD>, &VectorN<N, ZD>),
        noise_factor: &CorrelatedFactorNoise<N, ZD>,
    ) -> Result<N, &'static str>
        where
            DefaultAllocator: Allocator<N, ZD, ZD> + Allocator<N, ZD, D> + Allocator<N, ZD>
    {
        let x_size = self.x.nrows();
        let z_size = z.nrows();

        let mut scratch = self.new_observe_scratch();

        let mut zp = hx * &self.x;
        h_normalize(&mut zp, z);
        let mut zpdecol = zp.clone();

        // Observation prediction and normalised observation
        let mut GIHx = hx.clone();
        {
            // Solve G* GIHx = Hx for GIHx in-place
            for j in 0..x_size {
                for i in (0..z_size).rev() {
                    let UDi = noise_factor.UD.row(i);
                    let mut t = N::zero();
                    for k in i + 1..z_size {
                        t += UDi[k] * GIHx[(k, j)];
                    }
                    GIHx[(i, j)] -= t;
                }
            }

            // Solve G zp~ = z, G z~ = z  for zp~,z~ in-place
            for i in (0..z_size).rev() {
                let UDi = noise_factor.UD.row(i);
                for k in i + 1..z_size {
                    let UDik = UDi[k];
                    let zpt = UDik * zp[k];
                    zp[i] -= zpt;
                    let zpdt = UDik * zpdecol[k];
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
            let rcond = UDState::observeUD(self, &mut scratch, &mut S, noise_factor.UD[(o, o)]);
            matrix::check_positive(rcond, "S not PD in observe")?; // -1 implies S singular
            if rcond < rcondmin {
                rcondmin = rcond;
            }
            // State update using linear innovation
            let s = z[o] - zpdecol[o];
            self.x += &scratch.w * s;
        }
        Ok(rcondmin)
    }
}

/// Prediction Scratch.
///
/// Provides temporary variables for prediction calculation.
pub struct PredictScratch<N: RealField, D: Dim, QD: Dim>
    where
        D: DimAdd<QD>,
        DefaultAllocator: Allocator<N, D, QD> + Allocator<N, DimSum<D, QD>>,
{
    pub G: MatrixMN<N, D, QD>,
    pub d: VectorN<N, DimSum<D, QD>>,
    pub dv: VectorN<N, DimSum<D, QD>>,
    pub v: VectorN<N, DimSum<D, QD>>,
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

impl<N: RealField, D: Dim> UDState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub fn new_predict_scratch<QD: Dim>(&self, qd: QD) -> PredictScratch<N, D, QD>
        where
            D: DimAdd<QD>,
            DefaultAllocator: Allocator<N, D, QD> + Allocator<N, DimSum<D, QD>, U1>
    {
        // x + qd rows
        let xqd_size = self.UD.data.shape().1.add(qd);
        PredictScratch {
            G: MatrixMN::zeros_generic(self.UD.data.shape().0, qd),
            d: VectorN::zeros_generic(xqd_size, U1),
            dv: VectorN::zeros_generic(xqd_size, U1),
            v: VectorN::zeros_generic(xqd_size, U1),
        }
    }

    pub fn new_observe_scratch(&self) -> ObserveScratch<N, D> {
        let x_size = self.x.data.shape().0;
        ObserveScratch {
            w: VectorN::zeros_generic(x_size, U1),
            a: VectorN::zeros_generic(x_size, U1),
            b: VectorN::zeros_generic(x_size, U1),
        }
    }

    pub fn predict_use_scratch<QD: Dim>(
        &mut self,
        scratch: &mut PredictScratch<N, D, QD>,
        x_pred: &VectorN<N, D>,
        fx: &MatrixN<N, D>,
        noise: &CoupledNoise<N, D, QD>,
    ) -> Result<N, &'static str>
        where
            D: DimAdd<QD>,
            DefaultAllocator: Allocator<N, DimSum<D, QD>> + Allocator<N, D, QD> + Allocator<N, QD>,
    {
        self.x = x_pred.clone();

        // Predict UD from model
        let rcond = UDState::predictGq(self, scratch, &fx, &noise.G, &noise.q);
        matrix::check_non_negativ(rcond, "X not PSD")
    }

    pub fn observe_innovation_use_scratch<ZD: Dim>(
        &mut self,
        scratch: &mut ObserveScratch<N, D>,
        s: &VectorN<N, ZD>,
        hx: &MatrixMN<N, ZD, D>,
        noise: &UncorrelatedNoise<N, ZD>,
    ) -> Result<N, &'static str>
        where
            DefaultAllocator: Allocator<N, ZD, D> + Allocator<N, ZD>
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
            hx.row(o).transpose_to(&mut scratch.a);
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

    /// MWG-S prediction from Bierman p.132
    ///
    /// Return: reciprocal condition number, -1 if negative, 0 if semi-definite (including zero)
    fn predictGq<QD: Dim>(
        &mut self,
        scratch: &mut PredictScratch<N, D, QD>,
        Fx: &MatrixN<N, D>,
        G: &MatrixMN<N, D, QD>,
        q: &VectorN<N, QD>,
    ) -> N
        where
            D: DimAdd<QD>,
            DefaultAllocator: Allocator<N, DimSum<D, QD>> + Allocator<N, D, QD> + Allocator<N, QD>,
    {
        let nx = self.x.nrows();
        let nq = q.nrows();
        let nxq = nx + nq;

        // Augment d with q, UD with G
        scratch.d.rows_generic_mut(nx, q.data.shape().0).copy_from(q);
        scratch.G.copy_from(G);

        // U=Fx*U and diagonals retrieved
        for j in (1..nx).rev() { // nx-1..1
            // Prepare d as temporary
            let UDj = self.UD.column(j);
            scratch.d.rows_range_mut(0..j+1).copy_from(&UDj.rows_range(0..j+1));

            // Lower triangle of UD is implicitly empty
            for i in 0..nx {
                let mut t = Fx[(i, j)];
                for k in 0..j {
                    t += Fx[(i, k)] * scratch.d[k];
                }
                self.UD[(i, j)] = t;
            }
        }
        if nx > 0 {
            scratch.d[0] = self.UD[(0, 0)];
        }

        // Complete U = Fx*U
        self.UD.column_mut(0).copy_from(&Fx.column(0));

        // The MWG-S algorithm on UD transpose
        for j in (0..nx).rev() { // n-1..0
            let mut e = self.udu.zero;
            for k in 0..nx {
                scratch.v[k] = self.UD[(j, k)];
                scratch.dv[k] = scratch.d[k] * scratch.v[k];
                e += scratch.v[k] * scratch.dv[k];
            }
            for k in nx..nxq {
                scratch.v[k] = scratch.G[(j, k - nx)];
                scratch.dv[k] = scratch.d[k] * scratch.v[k];
                e += scratch.v[k] * scratch.dv[k];
            }
            // Check diagonal element
            if e > self.udu.zero {
                // Positive definite
                self.UD[(j, j)] = e;

                let diaginv = self.udu.one / e;
                for k in 0..j {
                    e = self.udu.zero;
                    for i in 0..nx {
                        e += self.UD[(k, i)] * scratch.dv[i];
                    }
                    for i in nx..nxq {
                        e += scratch.G[(k, i - nx)] * scratch.dv[i];
                    }
                    e *= diaginv;
                    self.UD[(j, k)] = e;

                    for i in 0..nx {
                        self.UD[(k, i)] -= e * scratch.v[i]
                    }
                    for i in nx..nxq {
                        scratch.G[(k, i - nx)] -= e * scratch.v[i]
                    }
                }
            } else if e == self.udu.zero {
                // Possibly semi-definite, check not negative
                self.UD[(j, j)] = e;

                // 1 / e is infinite
                for k in 0..j {
                    for i in 0..nx {
                        e = self.UD[(k, i)] * scratch.dv[i];
                        if e != self.udu.zero {
                            return self.udu.minus_one;
                        }
                    }
                    for i in nx..nxq {
                        e = scratch.G[(k, i - nx)] * scratch.dv[i];
                        if e != self.udu.zero {
                            return self.udu.minus_one;
                        }
                    }
                    // UD(j,k) unaffected
                }
            } else {
                // Negative
                return self.udu.minus_one;
            }
        } // MWG-S loop

        // Transpose and Zero lower triangle
        self.UD.fill_upper_triangle_with_lower_triangle();
        self.UD.fill_lower_triangle(N::zero(), 1);

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
    fn observeUD(&mut self, scratch: &mut ObserveScratch<N, D>, alpha: &mut N, q: N)
                 -> N {
        let n = self.UD.nrows();
        // a(n) is U'a
        // b(n) is Unweighted Kalman gain

        // Compute b = DU'h, a = U'h
        for j in (1..n).rev() { // n-1..1
            for k in 0..j {
                let t = self.UD[(k, j)] * scratch.a[k];
                scratch.a[j] += t;
            }
            scratch.b[j] = self.UD[(j, j)] * scratch.a[j];
        }
        scratch.b[0] = self.UD[(0, 0)] * scratch.a[0];

        // Update UD(0,0), d(0) modification
        *alpha = q + scratch.b[0] * scratch.a[0];
        if *alpha <= self.udu.zero {
            return self.udu.minus_one;
        }
        let mut gamma = self.udu.one / *alpha;
        self.UD[(0, 0)] *= q * gamma;
        // Update rest of UD and gain b
        for j in 1..n {
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
            for i in 0..j {
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
