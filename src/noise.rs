#![allow(non_snake_case)]

//! Bayesian estimation noise models.
//!
//! Linear Noise models are represented as structs.

use na::{allocator::Allocator, DefaultAllocator, Dim, MatrixMN, MatrixN, VectorN};
use na::SimdRealField;
use na::storage::Storage;
use nalgebra as na;

use crate::matrix;
use nalgebra::RealField;
use crate::matrix::MatrixUDU;

/// Additive noise.
///
/// Noise represented as a the noise variance vector.
pub struct UncorrelatedNoise<N: SimdRealField, QD: Dim>
    where
        DefaultAllocator: Allocator<N, QD>
{
    /// Noise variance
    pub q: VectorN<N, QD>,
}

/// Additive noise.
///
/// Noise represented as a the noise covariance matrix.
pub struct CorrelatedNoise<N: SimdRealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D>
{
    /// Noise covariance
    pub Q: MatrixN<N, D>
}

/// Additive noise.
///
/// Noise represented as a the noise covariance as a factorised UdU' matrix.
pub struct CorrelatedFactorNoise<N: SimdRealField, D: Dim>
    where
        DefaultAllocator: Allocator<N, D, D>
{
    /// Noise covariance
    pub UD: MatrixUDU<N, D>
}

/// Additive noise.
///
/// Noise represented as a the noise variance vector and a noise coupling matrix.
/// The noise covariance is G.q.G'.
pub struct CoupledNoise<N: SimdRealField, D: Dim, QD: Dim>
    where
        DefaultAllocator: Allocator<N, D, QD> + Allocator<N, QD>,
{
    /// Noise variance
    pub q: VectorN<N, QD>,
    /// Noise coupling
    pub G: MatrixMN<N, D, QD>,
}

impl<'a, N: RealField, D: Dim> CorrelatedNoise<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// Creates a CorrelatedNoise from an CoupledNoise.
    pub fn from_coupled<QD: Dim>(coupled: &'a CoupledNoise<N, D, QD>) -> Self
    where
        DefaultAllocator: Allocator<N, QD, QD> + Allocator<N, D, QD> + Allocator<N, QD>
    {
        let mut Q = MatrixMN::zeros_generic(coupled.G.data.shape().0, coupled.G.data.shape().0);
        matrix::quadform_tr(&mut Q, N::one(), &coupled.G, &coupled.q, N::one());
        CorrelatedNoise {
            Q
        }
    }
}

impl<'a, N: SimdRealField, QD: Dim> CorrelatedNoise<N, QD>
where
    DefaultAllocator: Allocator<N, QD, QD> + Allocator<N, QD>
{
    /// Creates a CorrelatedNoise from an UncorrelatedNoise.
    pub fn from_uncorrelated(uncorrelated: &'a UncorrelatedNoise<N, QD>) -> Self {
        let z_size = uncorrelated.q.data.shape().0;
        let mut correlated = CorrelatedNoise {
            Q: MatrixMN::zeros_generic(z_size, z_size)
        };
        for i in 0..uncorrelated.q.nrows() {
            correlated.Q[(i,i)] = uncorrelated.q[i];
        }

        correlated
    }
}
