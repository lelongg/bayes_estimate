use na::{allocator::Allocator, DefaultAllocator, Dim, Matrix, MatrixMN, RealField, Scalar, SquareMatrix, Vector};
use na::storage::{Storage, StorageMut};
use nalgebra as na;
use nalgebra::constraint::{DimEq, ShapeConstraint};

/// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
///
/// # Examples:
///
/// ```
/// # #[macro_use] extern crate approx;
/// # use nalgebra::{Matrix2, Matrix3, Matrix2x3, Vector2};
/// let mut mat = Matrix2::identity();
/// let lhs = Matrix2x3::new(1.0, 2.0, 3.0,
///                          4.0, 5.0, 6.0);
/// let mid = Vector3::new(0.1, 0.2, 0.3);
/// let expected = lhs * Matrix3::from_diagonal(&mid) * lhs.transpose() * 10.0 + mat * 5.0;
///
/// quadform_tr(&mat, 10.0, &lhs, &mid, 5.0);
/// assert_relative_eq!(mat, expected);
pub fn quadform_tr<N: RealField, D1, S, R3, C3, S3, D4, S4>(
    mat: &mut SquareMatrix<N, D1, S>,
    alpha: N,
    lhs: &Matrix<N, R3, C3, S3>,
    mid: &Vector<N, D4, S4>,
    beta: N,
) where
    D1: Dim,
    S: StorageMut<N, D1, D1>,
    R3: Dim,
    C3: Dim,
    D4: Dim,
    S3: Storage<N, R3, C3>,
    S4: Storage<N, D4>,
    ShapeConstraint: DimEq<D1, R3> + DimEq<C3, D4>
{
    mat.ger(alpha * mid[0], &lhs.column(0), &lhs.column(0), beta);

    for j in 1..mid.nrows() {
        mat.ger(alpha * mid[j], &lhs.column(j), &lhs.column(j), N::one());
    }
}


pub fn as_zeros<N: RealField, R: Dim, C: Dim>(shape: (R, C)) -> MatrixMN<N, R, C>
    where
        DefaultAllocator: Allocator<N, R, C>
{
    MatrixMN::zeros_generic(shape.0, shape.1)
}

pub fn copy_from<N, R1, C1, SB1, R2, C2, SB2>(this: &mut Matrix<N, R1, C1, SB1>, other: &Matrix<N, R2, C2, SB2>)
    where
        N: Scalar + Copy,
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
pub fn check_positive<'a, N: RealField>(rcond: N, message: &'a str) -> Result<N, &'a str>
{
    if rcond >= N::zero() {
        Result::Ok(rcond)
    } else {
        Result::Err(message)
    }
}
