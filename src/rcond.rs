use nalgebra::{RealField, Dim, VectorN, MatrixMN, DefaultAllocator, allocator::Allocator};

/**
 * Numerical comparison of reciprocal condition numbers
 *  Required for all linear algebra in models and filters
 */

/* Estimate the reciprocal condition number of a Diagonal Matrix for inversion.
 * D represents a diagonal matrix, the parameter is actually passed as a vector
 *
 * The Condition Number is defined from a matrix norm.
 *  Choose max element of D as the norm of the original matrix.
 *  Assume this norm for inverse matrix is min element D.
 *  Therefore rcond = min/max
 *
 * Note:
 *  Defined to be 0 for semi-definite and 0 for an empty matrix
 *  Defined to be 0 for max and min infinite
 *  Defined to be <0 for negative matrix (D element a value  < 0)
 *  Defined to be <0 with any NaN element
 *
 *  A negative matrix may be due to errors in the original matrix resulting in
 *   a factorisation producing special values in D (e.g. -infinity,NaN etc)
 *  By definition rcond <= 1 as min<=max
 */
pub fn rcond_vec<N : RealField, R : Dim>(dv: &VectorN<N,R>) -> N
    where DefaultAllocator : Allocator<N,R>
{
    // Special case an empty matrix
    let n = dv.nrows();
    if n == 0 {
        N::zero()
    } else {
        let mut mind = dv[0];
        let mut maxd = mind;

        for i in 0..n {
            let d = dv[i];
            if d != d {
                // NaN
                mind = N::one().neg();
                break;
            }
            if d < mind {
                mind = d;
            }
            if d > maxd {
                maxd = d;
            }
        }

        rcond_min_max(mind, maxd)
    }
}

pub fn rcond_symetric<N : RealField, R : Dim, C : Dim>(sm: &MatrixMN<N,R,C>) -> N
    where DefaultAllocator : Allocator<N,R,C>
{
    // Special case an empty matrix
    let n = sm.nrows();
    if n == 0 {
        N::zero()
    } else {
        let mut mind = sm[(0,0)];
        let mut maxd = mind;

        for i in 0..n {
            let d = sm[(i,i)];
            if d != d {
                // NaN
                mind = N::one().neg();
                break;
            }
            if d < mind {
                mind = d;
            }
            if d > maxd {
                maxd = d;
            }
        }

        rcond_min_max(mind, maxd)
    }
}

/* Estimate the reciprocal condition number of a Diagonal Matrix for inversion.
 * Same as rcond_internal except that elements are infinity are ignored
 * when determining the maximum element.
 */
pub fn rcond_ignore_infinity<N : RealField, R : Dim, C : Dim>(sm: &MatrixMN<N,R,C>) -> N
    where DefaultAllocator : Allocator<N,R,C>
{
    // Special case an empty matrix
    let n = sm.nrows();
    if n == 0 {
        N::zero()
    } else {
        let mut mind = sm[(0,0)];
        let mut maxd = N::zero();

        for i in 0..n {
            let d = sm[(i,i)];
            if d != d {
                // NaN
                mind = N::one().neg();
                break;
            }
            if d < mind {
                mind = d;
            }
            if d > maxd && N::one() / d != N::zero() {
                // ignore infinity for maxd
                maxd = d;
            }
        }

        if mind < N::zero() {
            // matrix is negative
            N::one().neg()
        } else {
            // ISSUE mind may still be -0, this is progated into rcond
            if maxd == N::zero() {
                // singular due to maxd == zero (elements all zero or infinity)
                N::zero()
            } else {
                assert!(mind <= maxd); // check sanity

                let rcond = mind / maxd; // rcond from min/max norm
                if rcond != rcond {
                    // NaN, singular due to (mind == maxd) == infinity
                    N::zero()
                } else {
                    assert!(rcond <= N::one());
                    rcond
                }
            }
        }
    }
}

fn rcond_min_max<N: RealField>(mind: N, maxd: N) -> N {
    if mind < N::zero() {
        // matrix is negative
        mind    // mind < 0 but does not represent a rcond
    } else {
        // ISSUE mind may still be -0, this is progated into rcond
        assert!(mind <= maxd); // check sanity

        let rcond = mind / maxd; // rcond from min/max norm
        if rcond != rcond {
            // NaN, singular due to (mind == maxd) == (zero or infinity)
            N::zero()
        } else {
            assert!(rcond <= N::one());
            rcond
        }
    }
}

