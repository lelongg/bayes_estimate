use nalgebra as na;

use na::{RealField, Dim, DefaultAllocator, allocator::Allocator, Matrix, Vector, Scalar, MatrixN, MatrixMN};
use na::base::storage::{Storage, StorageMut};
use nalgebra::{SquareMatrix, U1};
use nalgebra::base::constraint::{ShapeConstraint, DimEq, SameNumberOfRows};


/// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
///
/// This uses the provided workspace `work` to avoid allocations for intermediate results.
///
/// # Examples:
///
/// ```
/// # #[macro_use] extern crate approx;
/// # use nalgebra::{DMatrix, DVector};
/// // Note that all those would also work with statically-sized matrices.
/// // We use DMatrix/DVector since that's the only case where pre-allocating the
/// // workspace is actually useful (assuming the same workspace is re-used for
/// // several computations) because it avoids repeated dynamic allocations.
/// let mut mat = DMatrix::identity(2, 2);
/// let lhs = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0,
///                                           4.0, 5.0, 6.0]);
/// let mid = DMatrix::from_row_slice(3, 3, &[0.1, 0.2, 0.3,
///                                           0.5, 0.6, 0.7,
///                                           0.9, 1.0, 1.1]);
/// // The random shows that values on the workspace do not
/// // matter as they will be overwritten.
/// let mut workspace = DVector::new_random(2);
/// let expected = &lhs * &mid * lhs.transpose() * 10.0 + &mat * 5.0;
///
/// mat.quadform_tr_with_workspace(&mut workspace, 10.0, &lhs, &mid, 5.0);
/// assert_relative_eq!(mat, expected);
pub fn quadform_tr_with_workspace<N: RealField, D1: Dim, S: StorageMut<N, D1, D1>, D2, S2, R3, C3, S3, D4, S4>(
    mat: &mut SquareMatrix<N, D1, S>,
    work: &mut Vector<N, D2, S2>,
    alpha: N,
    lhs: &Matrix<N, R3, C3, S3>,
    mid: &Vector<N, D4, S4>,
    beta: N,
) where
    D2: Dim,
    R3: Dim,
    C3: Dim,
    D4: Dim,
    S2: StorageMut<N, D2>,
    S3: Storage<N, R3, C3>,
    S4: Storage<N, D4>,
    ShapeConstraint: DimEq<D1, D2> + DimEq<D1, R3> + DimEq<D2, R3> + DimEq<C3, D4>
         + SameNumberOfRows<D2,R3>
{
    work.copy_from::<R3, U1, _>(&lhs.column(0));
    *work *= mid[0];
    mat.ger(alpha, work, &lhs.column(0), beta);

    for j in 1..mid.nrows() {
        work.copy_from(&lhs.column(j));
        *work *= mid[j];
        mat.ger(alpha, work, &lhs.column(j), N::one());
    }
}


/// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
///
/// This allocates a workspace vector of dimension D1 for intermediate results.
/// If `D1` is a type-level integer, then the allocation is performed on the stack.
/// Use `.quadform_tr_with_workspace(...)` instead to avoid allocations.
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
pub fn quadform_tr<N, D1: Dim, S: StorageMut<N, D1, D1>, R3, C3, S3, D4, S4>(
    mat: &mut SquareMatrix<N, D1, S>,
    alpha: N,
    lhs: &Matrix<N, R3, C3, S3>,
    mid: &Vector<N, D4, S4>,
    beta: N,
) where
    N: RealField,
    R3: Dim,
    C3: Dim,
    D4: Dim,
    S3: Storage<N, R3, C3>,
    S4: Storage<N, D4>,
    ShapeConstraint: DimEq<D1, D1> + DimEq<D1, R3> + DimEq<C3, D4>,
    DefaultAllocator: Allocator<N, R3>
{
    let mut work = unsafe { Vector::new_uninitialized_generic(lhs.data.shape().0, U1) };
    quadform_tr_with_workspace(mat, &mut work, alpha, lhs, mid, beta)
}


pub fn prod_spd<N: RealField, R: Dim, C: Dim>(x: &MatrixMN<N, R, C>, s: &MatrixN<N, C>) -> MatrixN<N, R>
    where
        DefaultAllocator: Allocator<N, R, C> + Allocator<N, C, R> + Allocator<N, R, R> + Allocator<N, C, C>
{
    x * s * x.transpose()
}

pub fn prod_spdt<N: RealField, R: Dim, C: Dim>(x: &MatrixMN<N, R, C>, s: &MatrixN<N, R>) -> MatrixN<N, C>
    where
        DefaultAllocator:
        Allocator<N, R, C> + Allocator<N, C, R> + Allocator<N, R, R> + Allocator<N, C, C>
{
    (s * x).transpose() * x
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
