
use nalgebra as na;

use na::{RealField, Dim, DefaultAllocator, allocator::Allocator, Matrix, Scalar, MatrixN, MatrixMN};
use na::base::storage::{Storage, StorageMut};


pub fn prod_SPD<N: RealField, R: Dim, C: Dim>(X: &MatrixMN<N, R, C>, S: &MatrixN<N, C>) -> MatrixN<N, R>
    where
        DefaultAllocator: Allocator<N, R, C> + Allocator<N, C, R> + Allocator<N, R, R> + Allocator<N, C, C>
{
    X * S * X.transpose()
}

pub fn prod_SPDT<N: RealField, R: Dim, C: Dim>(X: &MatrixMN<N, R, C>, S: &MatrixN<N, R>) -> MatrixN<N, C>
    where
        DefaultAllocator:
        Allocator<N, R, C> + Allocator<N, C, R> + Allocator<N, R, R> + Allocator<N, C, C>
{
    (S * X).transpose() * X
}

pub fn as_zeros<N: RealField, R: Dim, C: Dim>(shape : (R,C)) -> MatrixMN<N,R,C>
    where
        DefaultAllocator: Allocator<N, R, C>
{
    MatrixMN::zeros_generic(shape.0, shape.1)
}

pub fn copy_from<N, R1, C1, SB1, R2, C2, SB2>(this : &mut Matrix<N, R1, C1, SB1>, other: &Matrix<N, R2, C2, SB2>)
    where
        N : Scalar + Copy,
        R1: Dim,
        C1: Dim,
        SB1: StorageMut<N, R1, C1>,
        R2: Dim,
        C2: Dim,
        SB2: Storage<N, R2, C2>
{
    assert!(
        this.shape() == other.shape(),
        "Unable to copy from a matrix with a different shape."
    );

    for j in 0..this.ncols() {
        for i in 0..this.nrows() {
            this[(i, j)] = other[(i, j)];
        }
    }
}

/**
 * Checks a the reciprocal condition number is >= 0
 * IEC 559 NaN values are never true
 */
pub fn check_NN<'a, N: RealField>(rcond : N, message : &'a str) -> Result<N, &'a str>
{
    if rcond >= N::zero() {
        Result::Ok(rcond)
    }
    else {
        Result::Err(message)
    }
}
