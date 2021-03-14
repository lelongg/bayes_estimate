use na::storage::{Storage, StorageMut};
use na::{Dim, Matrix, RealField, SquareMatrix, Vector};
use nalgebra as na;
use nalgebra::constraint::{DimEq, ShapeConstraint};

/// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
///
/// 'mid' is a diagonal matrix represented by a Vector.
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
    ShapeConstraint: DimEq<D1, R3> + DimEq<C3, D4>,
{
    mat.ger(alpha * mid[0], &lhs.column(0), &lhs.column(0), beta);

    for j in 1..mid.nrows() {
        mat.ger(alpha * mid[j], &lhs.column(j), &lhs.column(j), N::one());
    }
}

/// Computes the quadratic form `self = alpha * lhs * lhs.transpose() + beta * self`.
///
/// there is no 'mid'.
pub fn quadform_tr_x<N: RealField, D1, S, R3, C3, S3>(
    mat: &mut SquareMatrix<N, D1, S>,
    alpha: N,
    lhs: &Matrix<N, R3, C3, S3>,
    beta: N,
) where
    D1: Dim,
    S: StorageMut<N, D1, D1>,
    R3: Dim,
    C3: Dim,
    S3: Storage<N, R3, C3>,
    ShapeConstraint: DimEq<D1, R3>,
{
    mat.ger(alpha, &lhs.column(0), &lhs.column(0), beta);

    for j in 1..lhs.ncols() {
        mat.ger(alpha, &lhs.column(j), &lhs.column(j), N::one());
    }
}

/// Checks a the reciprocal condition number is > 0 .
///
/// IEC 559 NaN values are never true
pub fn check_positive<'a, N: RealField>(rcond: N, message: &'a str) -> Result<N, &'a str> {
    if rcond > N::zero() {
        Ok(rcond)
    } else {
        Err(message)
    }
}

/// Checks a the reciprocal condition number is >= 0 .
///
/// IEC 559 NaN values are never true
pub fn check_non_negativ<'a, N: RealField>(rcond: N, message: &'a str) -> Result<N, &'a str> {
    if rcond >= N::zero() {
        Ok(rcond)
    } else {
        Err(message)
    }
}

