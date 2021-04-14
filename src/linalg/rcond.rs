//! Numerical comparison of reciprocal condition numbers.
//!
//! Required for all linear algebra in models and filters.

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, MatrixMN, RealField};

pub fn rcond_symetric<N: RealField, R: Dim, C: Dim>(sm: &MatrixMN<N, R, C>) -> N
where
    DefaultAllocator: Allocator<N, R, C>,
{
    // Special case an empty matrix
    let n = sm.nrows();
    if n == 0 {
        N::zero()
    } else {
        let mut mind = sm[(0, 0)];
        let mut maxd = mind;

        for i in 0..n {
            let d = sm[(i, i)];
            if is_nan(d) {
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

fn rcond_min_max<N: RealField>(mind: N, maxd: N) -> N {
    if mind < N::zero() {
        // matrix is negative
        mind // mind < 0 but does not represent a rcond
    } else {
        // ISSUE mind may still be -0, this is progated into rcond
        assert!(mind <= maxd); // check sanity

        let rcond = mind / maxd; // rcond from min/max norm
        if is_nan(rcond) {
            // NaN, singular due to (mind == maxd) == (zero or infinity)
            N::zero()
        } else {
            assert!(rcond <= N::one());
            rcond
        }
    }
}

#[inline]
fn is_nan<R: RealField>(x: R) -> bool {
    x.partial_cmp(&R::zero()).is_none()
}
