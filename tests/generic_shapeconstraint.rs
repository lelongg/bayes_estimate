use nalgebra as na;
use na::{RealField, MatrixMN};
use na::{DefaultAllocator, allocator::Allocator, U1, U2};
use na::{Dim, Matrix2};
use na::base::constraint::{ShapeConstraint, DimEq, SameNumberOfRows, SameNumberOfColumns};


trait GenricDim<D: Dim>
{
    fn dim(&self) -> D;
}

struct UseU1 {
}
struct UseU2 {
}

impl GenricDim<U1> for UseU1 {

    fn dim(&self) -> U1 {
        U1
    }
}

impl GenricDim<U2> for UseU2 {

    fn dim(&self) -> U2 {
        U2
    }
}

pub fn test_dim_u1() {
    // test_dim_constraint(UseU1{}); // NOTE should not compile
}

pub fn test_dim_u2() {
    test_dim_constraint(UseU2{});
}

fn test_dim_constraint<D>(flt: impl GenricDim<D>)
    where
        DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
        D : Dim,
        ShapeConstraint: SameNumberOfRows<D, U2> + SameNumberOfColumns<D, U2>, // implies DimEq
        ShapeConstraint: SameNumberOfRows<U2, U2> + SameNumberOfColumns<U2, U2>, // WHY is this required!!
        // ShapeConstraint: DimEq<D, U2>,
        // ShapeConstraint: DimEq<U2, U2>,
{
    let d : D = flt.dim();
    let matrix2 = Matrix2::new(1., 2., 3., 4.);
    let _ = zero_from(U2, U2, &matrix2);
    let _ = zero_from(d, d, &matrix2);
}


fn zero_from<N: RealField, R: Dim, C: Dim, R1: Dim, C1: Dim>(_r : R, _c : C, _m : &MatrixMN<N, R1, C1>)
    where
        DefaultAllocator: Allocator<N, R1, C1> + Allocator<N, R, C>,
        // ShapeConstraint: SameNumberOfRows<R, R1> + SameNumberOfColumns<C, C1>,
        ShapeConstraint: DimEq<R, R1> + DimEq<C, C1>
{
}
