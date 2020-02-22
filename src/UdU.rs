/*
 * Bayes++ the Bayesian Filtering Library
 * Copyright (c) 2020 Michael Stevens
 * See accompanying Bayes++.htm for terms and conditions of use.
 */

#![allow(non_snake_case)]

use crate::rcond;
use nalgebra as na;
use na::{RealField, Dim, MatrixMN, VectorN};
use na::{DefaultAllocator, allocator::Allocator};

pub struct UDU<N : RealField> {
	pub zero : N,
	pub one : N,
	pub minus_one: N
}

impl<N : RealField> UDU<N> {

	pub fn new() -> UDU<N> {
		UDU {
			zero : N::zero(),
			one : N::one(),
			minus_one: N::one().neg()
		}
	}
	
	/*
	 * Linear algebra support functions for filter classes
	 * Cholesky and Modified Cholesky factorisations
	 *
	 * UdU' and LdL' factorisations of Positive semi-definite matrices. Where
	 *  U is unit upper triangular
	 *  d is diagonal
	 *  L is unit lower triangular
	 * Storage
	 *  UD(RowMatrix) format of UdU' factor
	 *   strict_upper_triangle(UD) = strict_upper_triangle(U), diagonal(UD) = d, strict_lower_triangle(UD) ignored or zeroed
	 *  LD(LTriMatrix) format of LdL' factor
	 *   strict_lower_triangle(LD) = strict_lower_triangle(L), diagonal(LD) = d, strict_upper_triangle(LD) ignored or zeroed
	 */


	/* Estimate the reciprocal condition number for inversion of the original PSD
	 * matrix for which d is the factor UdU' or LdL'.
	 * The original matrix must therefore be diagonal
	 */
	pub fn UdUrcond_vec<R: Dim>(d: &VectorN<N,R>) -> N
		where DefaultAllocator : Allocator<N,R>
	{
		rcond::rcond_vec(d)
	}

	/* Estimate the reciprocal condition number for inversion of the original PSD
	 * matrix for which UD is the factor UdU' or LdL'
	 * Additional columns are ignored.
	 * The rcond of the original matrix is simply the rcond of its d factor
	 * Using the d factor is fast and simple, and avoids computing any squares.
	 */
	pub fn UdUrcond<R: Dim, C: Dim>(UD: &MatrixMN<N,R,C>) -> N
		where DefaultAllocator : Allocator<N,R,C>	{
		rcond::rcond_symetric(&UD)
	}

	/* Estimate the reciprocal condition number for inversion of the original PSD
	 * matrix for which U is the factor UU'
	 *
	 * The rcond of the original matrix is simply the square of the rcond of diagonal(UC)
	 */
	pub fn UCrcond<R: Dim, C: Dim>(&self, UC: &MatrixMN<N,R,C>) -> N
		where DefaultAllocator : Allocator<N,R,C>
	{
		assert_eq!(UC.nrows(), UC.nrows());
		let rcond = rcond::rcond_symetric(&UC);
		// Square to get rcond of original matrix, take care to propogate rcond's sign!
		if rcond < self.zero {
			-(rcond * rcond)
		} else {
			rcond * rcond
		}
	}

	/* Compute the determinant of the original PSD
	 * matrix for which UD is the factor UdU' or LdL'
	 * Result comes directly from determinant of diagonal in triangular matrices
	 *  Defined to be 1 for 0 size UD
	 */
	pub fn UdUdet<R: Dim, C: Dim>(&self, UD: &MatrixMN<N,R,C>) -> N
		where DefaultAllocator : Allocator<N,R,C>
	{
		let n = UD.nrows();
		assert_eq!(n, UD.ncols());
		let mut det = self.one;
		for i in 0..n {
			det *= UD[(i, i)];
		}
		det
	}

	/* In place Modified upper triangular Cholesky factor of a
	 *  Positive definite or semi-definite matrix M
	 * Reference: A+G p.218 Upper Cholesky algorithm modified for UdU'
	 *  Numerical stability may not be as good as M(k,i) is updated from previous results
	 *  Algorithm has poor locality of reference and avoided for large matrices
	 *  Infinity values on the diagonal can be factorised
	 *
	 * Strict lower triangle of M is ignored in computation
	 *
	 * Input: M, n=last std::size_t to be included in factorisation
	 * Output: M as UdU' factor
	 *    strict_upper_triangle(M) = strict_upper_triangle(U)
	 *    diagonal(M) = d
	 *    strict_lower_triangle(M) is unmodified
	 * Return:
	 *    reciprocal condition number, -1 if negative, 0 if semi-definite (including zero)
	 */
	pub fn UdUfactor_variant1<R: Dim, C: Dim>(&self, M: &mut MatrixMN<N,R,C>, n: usize) -> N
		where DefaultAllocator : Allocator<N,R,C>
	{
		for j in (0..n).rev() {
			let mut d = M[(j, j)];

			// Diagonal element
			if d > self.zero {
				// Positive definite
				d = self.one / d;

				for i in 0..j {
					let e = M[(i, j)];
					M[(i, j)] = d * e;
					for k in 0..=i {
						let mut Mk = M.row_mut(k);
						let t = e * Mk[j];
						Mk[i] -= t;
					}
				}
			} else if d == self.zero {
				// Possibly semi-definite, check not negative
				for i in 0..j {
					if M[(i, j)] != self.zero {
						return self.minus_one
					}
				}
			} else {
				// Negative
				return self.minus_one
			}
		}

		// Estimate the reciprocal condition number
		UDU::UdUrcond(M)
	}

	/* In place modified upper triangular Cholesky factor of a
	 *  Positive definite or semi-definite matrix M
	 * Reference: A+G p.219 right side of table
	 *  Algorithm has good locality of reference and preferable for large matrices
	 *  Infinity values on the diagonal cannot be factorised
	 *
	 * Strict lower triangle of M is ignored in computation
	 *
	 * Input: M, n=last std::size_t to be included in factorisation
	 * Output: M as UdU' factor
	 *    strict_upper_triangle(M) = strict_upper_triangle(U)
	 *    diagonal(M) = d
	 *    strict_lower_triangle(M) is unmodified
	 * Return:
	 *    reciprocal condition number, -1 if negative, 0 if semi-definite (including zero)
	 */
	pub fn UdUfactor_variant2<R: Dim, C: Dim>(&self, M: &mut MatrixMN<N,R,C>, n: usize) -> N
		where DefaultAllocator : Allocator<N,R,C>
	{
		for j in (0..n).rev() {
			let mut d = M[(j, j)];

			// Diagonal element
			if d > self.zero {
				// Positive definite
				for i in (0..=j).rev() {
					let mut e = M[(i, j)];
					for k in j + 1..n {
						e -= M[(i, k)] * M[(k, k)] * M[(j, k)];
					}
					if i == j {
						d = e;
						M[(i, j)] = e; // Diagonal element
					} else {
						M[(i, j)] = e / d;
					}
				}
			} else if d == self.zero {
				// Possibly semi-definite, check not negative, whole row must be identically zero
				for k in j + 1..n {
					if M[(j, k)] != self.zero {
						return self.minus_one
					}
				}
			} else {
				// Negative
				return self.minus_one
			}
		}

		// Estimate the reciprocal condition number
		UDU::UdUrcond(M)
	}

	/* In place modified lower triangular Cholesky factor of a
	 *  Positive definite or semi-definite matrix M
	 * Reference: A+G p.218 Lower Cholesky algorithm modified for LdL'
	 *
	 * Input: M, n=last std::size_t to be included in factorisation
	 * Output: M as LdL' factor
	 *    strict_lower_triangle(M) = strict_lower_triangle(L)
	 *    diagonal(M) = d
	 * Return:
	 *    reciprocal condition number, -1 if negative, 0 if semi-definite (including zero)
	 * ISSUE: This could change to be equivalent to UdUfactor_varient2
	 */
	pub fn LdLfactor_n<R: Dim, C: Dim>(&self, M: &mut MatrixMN<N,R,C>, n: usize) -> N
		where DefaultAllocator : Allocator<N,R,C>
	{
		for j in 0..n {
			let mut d = M[(j, j)];

			// Diagonal element
			if d > self.zero {
				// Positive definite
				d = self.one / d;

				for i in j + 1..n {
					let e = M[(i, j)];
					M[(i, j)] = d * e;
					for k in i..n {
						let t = e * M[(k, j)];
						M[(k, i)] -= t;
					}
				}
			} else if d == self.zero {
				// Possibly semi-definite, check not negative
				for i in j + 1..n {
					if M[(i, j)] != self.zero {
						return self.minus_one
					}
				}
			} else {
				// Negative
				return self.minus_one
			}
		}

		// Estimate the reciprocal condition number
		UDU::UdUrcond(M)
	}


	/* In place upper triangular Cholesky factor of a
	 *  Positive definite or semi-definite matrix M
	 * Reference: A+G p.218
	 * Strict lower triangle of M is ignored in computation
	 *
	 * Input: M, n=last std::size_t to be included in factorisation
	 * Output: M as UC*UC' factor
	 *    upper_triangle(M) = UC
	 * Return:
	 *    reciprocal condition number, -1 if negative, 0 if semi-definite (including zero)
	 */
	pub fn UCfactor_n<R: Dim, C: Dim>(&self, M: &mut MatrixMN<N,R,C>, n: usize) -> N
		where DefaultAllocator : Allocator<N,R,C>
	{
		for j in (0..n).rev() {
			let mut d = M[(j, j)];

			// Diagonal element
			if d > self.zero {
				// Positive definite
				d = N::sqrt(d);
				M[(j, j)] = d;
				d = self.one / d;

				for i in 0..j {
					let e = d * M[(i, j)];
					M[(i, j)] = e;
					for k in 0..=i {
						let t = e * M[(k, j)];
						M[(k, i)] -= t;
					}
				}
			} else if d == self.zero {
				// Possibly semi-definite, check not negative
				for i in 0..j {
					if M[(i, j)] != self.zero {
						return self.one
					}
				}
			} else {
				// Negative
				return self.minus_one
			}
		}

		// Estimate the reciprocal condition number
		self.UCrcond(M)
	}

	/* In-place (destructive) inversion of diagonal and unit upper triangular matrices in UD
	 * BE VERY CAREFUL THIS IS NOT THE INVERSE OF UD
	 *  Inversion on d and U is separate: inv(U)*inv(d)*inv(U') = inv(U'dU) NOT EQUAL inv(UdU')
	 * Lower triangle of UD is ignored and unmodified
	 * Only diagonal part d can be singular (zero elements), inverse is computed of all elements other then singular
	 * Reference: A+G p.223
	 *
	 * Output:
	 *    UD: inv(U), inv(d)
	 * Return:
	 *    singularity (of d), true iff d has a zero element
	 */
	pub fn UdUinverse<R: Dim, C: Dim>(&self, UD: &mut MatrixMN<N,R,C>) -> bool
		where DefaultAllocator : Allocator<N,R,C>
	{
		let n = UD.nrows();
		assert_eq!(n, UD.ncols());

		// Invert U in place
		if n > 1 {
			for i in (0..n - 1).rev() {
				for j in (i + 1..n).rev() {
					let mut UDij = -UD[(i, j)];
					for k in i + 1..j {
						UDij -= UD[(i, k)] * UD[(k, j)];
					}
					UD[(i, j)] = UDij;
				}
			}
		}

		// Invert d in place
		let mut singular = false;
		for i in 0..n {
			// Detect singular element
			let UDii = UD[(i, i)];
			if UDii != self.zero {
				UD[(i, i)] = self.one / UDii;
			} else {
				singular = true;
			}
		}

		singular
	}


	/* In-place (destructive) inversion of upper triangular matrix in U
	 *
	 * Output:
	 *    U: inv(U)
	 * Return:
	 *    singularity (of U), true iff diagonal of U has a zero element
	 */
	pub fn UTinverse<R: Dim, C: Dim>(&self, U: &mut MatrixMN<N,R,C>) -> bool
		where DefaultAllocator : Allocator<N,R,C>
	{
		let n = U.nrows();
		assert_eq!(n, U.ncols());

		let mut singular = false;
		// Invert U in place
		for i in (0..n).rev() {
			let mut d = U[(i, i)];
			if d == self.zero {
				singular = true;
				break;
			}
			d = self.one / d;
			U[(i, i)] = d;

			for j in (i + 1..n).rev() {
				let mut e = self.zero;
				for k in i + 1..=j {
					e -= U[(i, k)] * U[(k, j)];
				}
				U[(i, j)] = e * d;
			}
		}

		singular
	}


	/* In-place recomposition of Symmetric matrix from U'dU factor store in UD format
	 *  Generally used for recomposing result of UdUinverse
	 * Note definiteness of result depends purely on diagonal(M)
	 *  i.e. if d is positive definite (>0) then result is positive definite
	 * Reference: A+G p.223
	 * In place computation uses simple structure of solution due to triangular zero elements
	 *  Defn: R = (U' d) row i , C = U column j   -> M(i,j) = R dot C;
	 *  However M(i,j) only dependent R(k<=i), C(k<=j) due to zeros
	 *  Therefore in place multiple sequences such k < i <= j
	 * Input:
	 *    M - U'dU factorisation (UD format)
	 * Output:
	 *    M - U'dU recomposition (symmetric)
	 */
	pub fn UdUrecompose_transpose<R: Dim, C: Dim>(M: &mut MatrixMN<N,R,C>)
		where DefaultAllocator : Allocator<N,R,C>
	{
		let n = M.nrows();
		assert_eq!(n, M.ncols());

		// Recompose M = (U'dU) in place
		for i in (0..n).rev() {
			// (U' d) row i of lower triangle from upper triangle
			for j in 0..i {
				M[(i, j)] = M[(j, i)] * M[(j, j)];
			}
			// (U' d) U in place
			for j in (i..n).rev() {
				// Compute matrix product (U'd) row i * U col j
				if j > i {                    // Optimised handling of 1 in U
					let mii = M[(i, i)];
					M[(i, j)] *= mii;
				}
				for k in 0..i {    // Inner loop k < i <=j, only strict triangular elements
					let t = M[(i, k)] * M[(k, j)];
					M[(i, j)] += t;        // M(i,k) element of U'd, M(k,j) element of U
				}
				M[(j, i)] = M[(i, j)];
			}
		}
	}


	/* In-place recomposition of Symmetric matrix from UdU' factor store in UD format
	 *  See UdUrecompose_transpose()
	 * Input:
	 *    M - UdU' factorisation (UD format)
	 * Output:
	 *    M - UdU' recomposition (symmetric)
	 */
	pub fn UdUrecompose<R: Dim, C: Dim>(M: &mut MatrixMN<N,R,C>)
		where DefaultAllocator : Allocator<N,R,C>
	{
		let n = M.nrows();
		assert_eq!(n, M.ncols());

		// Recompose M = (UdU') in place
		for i in 0..n {
			// (d U') col i of lower triangle from upper trinagle
			for j in i + 1..n {
				M[(j, i)] = M[(i, j)] * M[(j, j)];
			}
			// U (d U') in place
			for j in 0..=i {    // j<=i
				// Compute matrix product (U'd) row i * U col j
				if j > i {                // Optimised handling of 1 in U
					let mii = M[(i, i)];
					M[(i, j)] *= mii;
				}
				for k in i + 1..n {        // Inner loop k > i >=j, only strict triangular elements
					let t = M[(i, k)] * M[(k, j)];
					M[(i, j)] += t;        // M(i,k) element of U'd, M(k,j) element of U
				}
				M[(j, i)] = M[(i, j)];
			}
		}
	}


	/*
	 * Zero strict lower triangle of Matrix
     */
	pub fn Lzero<R: Dim, C: Dim>(&self, M: &mut MatrixMN<N,R,C>)
		where DefaultAllocator : Allocator<N,R,C>
	{
		let n = M.nrows();
		assert_eq!(n, M.ncols());
		for i in 1..n {
			for j in 0..i {
				M[(i, j)] = self.zero;
			}
		}
	}

	/* Zero strict upper triangle of Matrix
	 */
	pub fn Uzero<R: Dim, C: Dim>(&self, M: &mut MatrixMN<N,R,C>)
		where DefaultAllocator : Allocator<N,R,C>
	{
		let n = M.nrows();
		assert_eq!(n, M.ncols());
		for i in 0..n {
			for j in i + 1..n {
				M[(i, j)] = self.zero;
			}
		}
	}


	/* Convert a normal upper triangular Cholesky factor into
	 * a Modified Cholesky factor.
	 * Lower triangle of UD is ignored and unmodified
	 * Ignores Columns with zero diagonal element
	 *  Correct for zero columns i.e. UD is Cholesky factor of a PSD Matrix
	 * Note: There is no inverse to this function toCholesky as square losses the sign
	 *
	 * Input:
	 *    U Normal Cholesky factor (Upper triangular)
	 * Output:
	 *    U Modified Cholesky factor (UD format)
	 */
	pub fn UdUfromUCholesky<R: Dim, C: Dim>(&self, U: &mut MatrixMN<N,R,C>)
		where DefaultAllocator : Allocator<N,R,C>
	{
		let n = U.nrows();
		assert_eq!(n, U.ncols());
		for j in 0..n {
			let sd = U[(j, j)];
			U[(j, j)] = sd * sd;
			// Devide columns by square of non zero diagonal
			if sd != self.zero {
				for i in 0..j {
					U[(i, j)] /= sd;
				}
			}
		}
	}

	/* Extract the separate U and d parts of the UD factorisation
	 * Output:
	 *    U and d parts of UD
	 */
	pub fn UdUseperate<R: Dim, C: Dim>(&self, U: &mut MatrixMN<N,R,C>, d: &mut VectorN<N,R>, UD: &MatrixMN<N,R,C>)
		where DefaultAllocator : Allocator<N,R,C> + Allocator<N,R>
	{
		let n = UD.nrows();
		assert_eq!(n, UD.ncols());

		for j in 0..n {
			// Extract d and set diagonal to 1
			d[j] = UD[(j, j)];
			U[(j, j)] = self.one;
			for i in 0..j {
				U[(i, j)] = UD[(i, j)];
				// Zero lower triangle of U
				U[(j, i)] = self.zero;
			}
		}
	}


	/*
	 * Positive Definite matrix inversions built using UdU factorisation
	 */

	/* Inverse of Positive Definite matrix
	 * The inverse is able to deal with infinities on the leading diagonal
	 * Input:
	 *     M is a symmetric matrix
	 * Output:
	 *     M inverse of M, only updated if return value >0
	 * Return:
	 *     reciprocal condition number, -1 if negative, 0 if semi-definite (including zero)
	 */
	pub fn UdUinversePDignoreInfinity<R: Dim, C: Dim>(&self, M: &mut MatrixMN<N,R,C>) -> N
		where DefaultAllocator : Allocator<N,R,C>
	{
		// Must use variant1, variant2 cannot deal with infinity
		let orig_rcond = UDU::UdUfactor_variant1(self, M, M.nrows());
		// Ignore the normal rcond and recompute ignoring infinites
		let rcond = rcond::rcond_ignore_infinity(&M);
		assert!(rcond == orig_rcond || orig_rcond == self.zero);

		// Only invert and recompose if PD
		if rcond > self.zero {
			let singular = self.UdUinverse(M);
			assert!(!singular);
			UDU::UdUrecompose_transpose(M);
		}
		rcond
	}

	/* Inverse of Positive Definite matrix
	 * Input:
	 *     M is a symmetric matrix
	 * Output:
	 *     M inverse of M, only updated if return value >0
	 * Return:
	 *     reciprocal condition number, -1 if negative, 0 if semi-definite (including zero)
	 */
	pub fn UdUinversePD<R: Dim, C: Dim>(&self, M: &mut MatrixMN<N,R,C>) -> N
		where DefaultAllocator : Allocator<N,R,C>
	{
		let rcond = UDU::UdUfactor_variant2(self, M, M.nrows());
		// Only invert and recompose if PD
		if rcond > self.zero {
			let singular = self.UdUinverse(M);
			assert!(!singular);
			UDU::UdUrecompose_transpose(M);
		}
		rcond
	}

	/* Inverse of Positive Definite matrix
	 * Input:
	 *     M is a symmetric matrix
	 * Output:
	 *     M inverse of M, only updated if return value >0
	       detM determinant of original M if M is PSD
	 * Return:
	 *     reciprocal condition number, -1 if negative, 0 if semi-definite (including zero)
	 */
	pub fn UdUinversePD_det<R: Dim, C: Dim>(&self, M: &mut MatrixMN<N,R,C>, detM: &mut N) -> N
		where DefaultAllocator : Allocator<N,R,C>
	{
		let rcond = UDU::UdUfactor_variant2(self, M, M.nrows());
		// Only invert and recompose if PD
		if rcond > self.zero {
			*detM = self.UdUdet(M);
			let singular = self.UdUinverse(M);
			assert!(!singular);
			UDU::UdUrecompose_transpose(M);
		}
		rcond
	}

}
