//! SIR state estimation.
//!


use num_traits::{Float, Pow, ToPrimitive};
use rand::{Rng, RngCore};
use rand_distr::{Distribution, StandardNormal, Uniform};

use na::{DefaultAllocator, Dim, DVector, MatrixN, RealField, VectorN};
use na::allocator::Allocator;
use na::storage::Storage;
use nalgebra as na;
use nalgebra::{Dynamic, U1};

use crate::cholesky::UDU;
use crate::matrix::{check_non_negativ};
use crate::models::{Estimator, KalmanEstimator, KalmanState};
use crate::noise::CorrelatedNoise;

pub struct SampleState<N: RealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D>,
{
    pub s: Samples<N, D>,
    pub w: Weights,
    pub rng: Box<dyn RngCore>
}

pub type Samples<N, D> = Vec<VectorN<N, D>>;

pub type Weights = DVector<f32>;

pub type Resamples = Vec<u32>;

pub type Resampler = dyn FnMut(&mut Resamples, &mut DVector<f32>, &mut dyn RngCore) -> Result<(u64, f32), &'static str>;

pub type Roughener<N, D> = dyn FnMut(&mut Vec<VectorN<N, D>>, &mut dyn RngCore);


impl<N: RealField + ToPrimitive, D: Dim> SampleState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub fn new_equal_weigth(s: Vec<VectorN<N, D>>, rng: Box<dyn RngCore>) -> SampleState<N, D> {
        let samples = s.len();
        SampleState {
            s,
            w: DVector::<f32>::repeat(samples, 1f32),
            rng
        }
    }

    pub fn predict(&mut self, f: fn(&VectorN<N, D>) -> VectorN<N, D>)
    /* Predict state posterior with sampled noise model
     *  Pre : S represent the prior distribution
     *  Post: S represent the predicted distribution, stochastic_samples := samples in S
     */
    {
        // Predict particles S using supplied predict model
        self.s.iter_mut().for_each(|r|{
            r.copy_from(&f(r))
        });
    }

    pub fn sample_likelihood(&mut self, l: Vec<f32>) {
        assert!(self.w.len() == l.len());
        let mut li = l.iter();
        for wi in self.w.iter_mut() {
            *wi *= li.next().unwrap();
        }
    }

    pub fn update_resample(&mut self, resampler: &mut Resampler, roughener: &mut Roughener<N, D>) -> Result<f32, &'static str>
    /* Resample particles using weights and roughen
     * Pre : S represent the predicted distribution
     * Post: S represent the fused distribution, n_resampled from weighted_resample
     * Exceptions:
     *  Bayes_filter_exception from resampler
     *    unchanged: S, stochastic_samples
     * Return
     *  lcond, Smallest normalised weight, represents conditioning of resampling solution
     *  lcond == 1 if no resampling performed
     *  This should by multiplied by the number of samples to get the Likelihood function conditioning
     */
    {
        // Resample based on likelihood weights
        let mut resamples = Resamples::with_capacity(self.w.nrows());
        let (_unqiue_samples, lcond) = resampler(&mut resamples, &mut self.w, self.rng.as_mut())?;

        // update s
        SampleState::live_samples(&mut self.s, &resamples);

        roughener(&mut self.s, self.rng.as_mut()); // Roughen samples

        self.w.fill(1.);        // Resampling results in uniform weights

        Ok(lcond)
    }

    fn live_samples(p: &mut Vec<VectorN<N, D>>, resamples: &Resamples)
    /* Update ps by selectively copying resamples
     * Uses a in-place copying algorithm
     * Algorithm: In-place copying
     *  First copy the live samples (those resampled) to end of P
     *  Replicate live sample in-place
     */
    {
        // reverse_copy_if live
        let mut si = p.len();
        let mut livei = si;
        for pr in resamples.iter().rev() {
            si -= 1;
            if *pr > 0 {
                livei -= 1;
                p[livei] = p[si].clone();
            }
        }
        assert!(si == 0);
        // Replicate live samples
        si = 0;
        for pr in resamples {
            let mut res = *pr;
            if res > 0 {
                loop {
                    p[si] = p[livei].clone();
                    si += 1;
                    res -= 1;
                    if res == 0 { break; }
                }
                livei += 1;
            }
        }
        assert!(si == p.len());
        assert!(livei == p.len());
    }

    pub fn roughen_minmax(ps: &mut Vec<VectorN<N, D>>, k: f32, rng: &mut dyn RngCore)
    where
        DefaultAllocator: Allocator<f32, D> + Allocator<N, D, U1>,
    /* Roughening
     *  Uses algorithm from Ref[1] using max-min in each state of P
     *  K is scaling factor for roughening noise
     *  unique_samples is unchanged as roughening is used to postprocess observe resamples
     * Numerical collapse of P
     *  P with very small or zero range result in minimal roughening
     * Exceptions:
     *  none
     *		unchanged: P
     */
    {
        let x_dim = ps[0].data.shape().0;
        let x_size = x_dim.value();
        // Scale Sigma by constant and state dimensions
        let sigma_scale = k * f32::pow(ps.len() as f32, -1f32/(x_size as f32));

        // Find min and max states in all P, precond P not empty
        let mut xmin = ps[0].clone();
        let mut xmax = xmin.clone();
        for p in ps.iter_mut() {		// Loop includes 0 to simplify code
            let mut mini = xmin.iter_mut();
            let mut maxi = xmax.iter_mut();

            for xp in p.iter() {
                let min = mini.next().unwrap();
                let max = maxi.next().unwrap();

                if *xp < *min {*min = *xp;}
                if *xp > *max {*max = *xp;}
            }
        }
        // Roughening st.dev max-min
        let mut rootq = xmax - xmin;
        rootq *= N::from_f32(sigma_scale).unwrap();
        // Apply roughening predict based on scaled variance
        for p in ps.iter_mut() {
            let rnormal = StandardNormal.sample_iter(&mut *rng).map(|n| {N::from_f32(n).unwrap()}).take(x_size);
            let mut n = VectorN::<N, D>::from_iterator_generic(x_dim, U1, rnormal);

            // multiply elements by std dev
            for (ni, nr) in n.iter_mut().enumerate() {
                *nr *= rootq[ni];
            }
            *p += n;
        }
    }
}

pub fn test() {
    let rng = rand::thread_rng();
    let mut sir = SampleState::new_equal_weigth(Vec::new(), Box::new(rng));

    let resampler : &mut Resampler = &mut |presamples: &mut Resamples, w: &mut DVector<f32>, rng: &mut dyn RngCore| {standard_resampler(presamples, w, rng)};
    let mut roughener= |ps: &mut Vec<VectorN<f32, Dynamic>>, rng: &mut dyn RngCore| {SampleState::roughen_minmax(ps, 1f32, rng)};
    sir.update_resample(resampler, &mut roughener).unwrap();
}


pub fn standard_resampler(resamples: &mut Resamples, w: &mut DVector<f32>, rng: &mut dyn RngCore) -> Result<(u64, f32), &'static str>
/* Standard resampler from [1]
 * Algorithm:
 *	A particle is chosen once for each time its cumulative weight intersects with a uniform random draw.
 *	Complexity O(n*log(n))
 *  This complexity is required to sort the uniform random draws made,
 *	this allows comparing of the two ordered lists w(cumulative) and ur (the sort random draws).
 * Output:
 *  resamples number of times this particle should be resampled
 *  number of unqiue particles (number of non zeros in resamples)
 *  w becomes a normalised cumulative sum
 * Return:
 *  conditioning of the weigths (min weight / sum weigths)
 * Side effects:
 *  A draw is made from 'r' for each particle
 */
{
    let uniform01: Uniform<f32> = Uniform::new(0f32, 1f32);

    assert!(resamples.len() == 0);

    // Normalised cumulative sum of likelihood weights (Kahan algorithm), and find smallest weight
    let mut wmin = f32::max_value();
    let mut wcum = 0.;
    {
        let mut c = 0.;
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
    if wmin < 0.{ // bad weights
        return Err("negative weight");
    }
    if wcum <= 0. { // bad cumulative weights (previous check should actually prevent -ve
        return Err("zero cumulative weight sum");
    }
    // Any numerical failure should cascade into cumulative sum
    if wcum != wcum { // inequality due to NaN
        return Err("NaN cumulative weight sum");
    }

    // Sorted uniform random distribution [0..1) for each resample
    let urng = rng.sample_iter(uniform01).take(w.nrows());
    let mut ur = DVector::<f32>::from_iterator(w.nrows(), urng);

    unsafe {
        ur.data.as_vec_mut().sort_by(|a, b| a.total_cmp(&b));
    }
    // ur.into_iter().sorted().collect();
    assert!(ur[0] >= 0. && ur[ur.nrows()-1] < 1.);	// very bad if random is incorrect
    // Scale ur to cumulative sum
    ur *= wcum;

    // Resamples based on cumulative weights from sorted resample random values
    let mut uri = ur.iter().cloned();
    let mut urn = uri.next();
    let mut unique : u64 = 0;

    for wi in w.iter().cloned() {
        let mut res: u32 = 0;		// assume not resampled until find out otherwise
        if (urn.is_some()) && urn.unwrap() < wi {
            unique += 1;
            loop {                        // count resamples
                res += 1;
                urn = uri.next();
                if urn.is_none() { break; }
                if !(urn.unwrap() < wi) { break;}
            }
        }
        resamples.push(res);
    }

    if uri.peekable().peek().is_some() {                // resample failed due no non numeric weights
        return Err("weights are not numeric and cannot be resampled");
    }

    return Ok((unique, wmin / wcum));
}

impl<N: RealField, D: Dim> Estimator<N, D>  for SampleState<N, D>
    where
    DefaultAllocator: Allocator<N, D>,
{
    fn state(&self) -> Result<VectorN<N, D>, &'static str> {
        // Mean of distribution: mean of particles
        let s_shape = self.s[0].data.shape();
        let mut x = VectorN::zeros_generic(s_shape.0, s_shape.1);
        for s in self.s.iter() {
            x += s;
        }
        x /= N::from_usize(self.s.len()).unwrap();

        Ok(x)
    }
}

impl<N: RealField, D: Dim> CorrelatedNoise<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, U1, D> + Allocator<N, D>,
{
    pub fn add_sample_noise(&self, state: &mut SampleState<N,D>) -> Result<(), &'static str> {
        // Decorrelate state noise
        let mut uc = self.Q.clone();
        let udu = UDU::new();
        let rcond = udu.UCfactor_n(&mut uc, self.Q.nrows());
        check_non_negativ(rcond, "Init X not PSD")?;
        udu.Lzero(&mut uc);

        // Sample from noise variance
        for s in state.s.iter_mut() {
            let rnormal = StandardNormal.sample_iter(&mut *state.rng).map(|n| {N::from_f32(n).unwrap()}).take(self.Q.nrows());
            let n = VectorN::<N, D>::from_iterator_generic(self.Q.data.shape().0, U1, rnormal);
            *s += &uc * n;
        }
        Ok(())
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for SampleState<N, D>
    where
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, U1, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<N, &'static str> {
        for s in self.s.iter_mut() {
            s.copy_from(&state.x);
        }
        CorrelatedNoise{Q: state.X.clone()}.add_sample_noise(self)?;

        Ok(N::one())
    }

    fn kalman_state(&self) -> Result<KalmanState<N, D>, &'static str> {
        let x = self.state()?;
        // Covariance of distribution: covariance of particles
        let s_shape = self.s[0].data.shape();
        let mut xx = MatrixN::zeros_generic(s_shape.0, s_shape.0);

        for s in self.s.iter() {
            let sx = s - &x;
            let sxt = sx.transpose();
            xx += sx * sxt;
        }
        xx /= N::from_usize(self.s.len()).unwrap();

        Ok(KalmanState { x, X: xx })
    }
}

