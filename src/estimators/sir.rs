//! SIR state estimation.
//!


use nalgebra as na;
use na::{allocator::Allocator, DefaultAllocator, storage::Storage, Dim, U1, MatrixMN, MatrixN, RealField, VectorN};
use nalgebra::DVector;

use rand::distributions::Standard;
use rand::Rng;
use std::convert::TryInto;

type Resamples = Vec<u32>;

pub fn standard_resampler<N: RealField, R: Rng>(presamples: &Resamples, uresamples: &mut u64, w: &mut DVector<N>, rng: &mut R) -> Result<N, &'static str>
/* Standard resampler from [1]
 * Algorithm:
 *	A particle is chosen once for each time its cumulative weight intersects with a uniform random draw.
 *	Complexity O(n*log(n))
 *  This complexity is required to sort the uniform random draws made,
 *	this allows comparing of the two ordered lists w(cumulative) and ur (the sort random draws).
 * Output:
 *  presamples number of times this particle should be resampled
 *  uresamples number of unqiue particles (number of non zeros in Presamples)
 *  w becomes a normalised cumulative sum
 * Side effects:
 *  A draw is made from 'r' for each particle
 */
{
    assert!(presamples.len() == w.nrows());
    // Normalised cumulative sum of likelihood weights (Kahan algorithm), and find smallest weight
    let mut wmin = N::max_value();
    let mut wcum = N::zero();
    {
        let mut c = N::zero();
        for wi in w.iter_mut() {
            if *wi < wmin {
                wmin = *wi;
            }
            let y = *wi - c;
            let t = wcum + y;
            c = t - wcum - y;
            wcum = t;
            *wi = t;
        }
    }
    if wmin < N::zero() { // bad weights
        return Err("negative weight");
    }
    if wcum <= N::zero() { // bad cumulative weights (previous check should actually prevent -ve
        return Err("zero cumulative weight sum");
    }
    // Any numerical failure should cascade into cumulative sum
    if wcum != wcum { // inequality due to NaN
        return Err("NaN cumulative weight sum");
    }

    // Sorted uniform random distribution [0..1) for each resample
    let range = (0..w.nrows());
    let uniform01 = range.map(|i| {rng.sample(Standard)});
    let mut ur = DVector::<f32>::from_iterator(w.nrows(), uniform01);

    unsafe {
        ur.data.as_vec_mut().sort_by(|a, b| a.total_cmp(&b));
    }
    // ur.into_iter().sorted().collect();
    assert!(ur[0] >= 0. && ur[ur.nrows()-1] < 1.);	// very bad if random is incorrect
    // Scale ur to cumulative sum
    let scale: f32 = wcum.try_into().unwrap();
    ur *= scale;//wcum.try_into();
    // Resamples based on cumulative weights from sorted resample random values
    // Resamples_t::iterator pri = presamples.begin();
    // wi = w.begin();
    // DenseVec::iterator ui = ur.begin(), ui_end = ur.end();
    // std::size_t unique = 0;
    //
    // while (wi != wi_end)
    // {
    // std::size_t Pres = 0;		// assume P not resampled until find out otherwise
    // if (ui != ui_end && *ui < *wi)
    // {
    // ++unique;
    // do						// count resamples
    // {
    // ++Pres;
    // ++ui;
    // } while (ui != ui_end && *ui < *wi);
    // }
    // ++wi;
    // *pri++ = Pres;
    // }
    // assert (pri==presamples.end());	// must traverse all of P
    //
    // if (ui != ui_end)				// resample failed due no non numeric weights
    // error (Numeric_exception("weights are not numeric and cannot be resampled"));
    //
    // uresamples = unique;
    return Ok(wmin / wcum);
}
