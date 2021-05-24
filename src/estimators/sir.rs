//!
//! Sampling Importance Resampleing estimation.
//! Also know as a weighted Booststrap.
//!
//! References
//!  [1] "Novel approach to nonlinear-non-Guassian Bayesian state estimation"
//!   NJ Gordon, DJ Salmond, AFM Smith IEE Proceeding-F Vol.140 No.2 April 1993
//!  [2] Building Robust Simulation-based Filter for Evolving Data Sets"
//!   J Carpenter, P Clifford, P Fearnhead Technical Report Unversity of Oxford
//!
//! A variety of resampling algorithms can be used for SIR.
//! There are implementations for two algorithms:
//!   standard_resample: Standard resample algorithm from [1]
//!   systematic_resample: A Simple stratified resampler from [2]
//!
//! NOTES:
//!  SIR algorithms is sensitive to the PRNG properties.
//!  In particular we require that the uniform random number range be [0..1) NOT [0..1].
//!  Quantisation generated random number must not approach the sample size.  This will result in quantisation
//!  of the resampling. For example if random identically equal to 0 becomes highly probable due to quantisation
//!  this will result in the first sample being selectively draw whatever its likelihood.
//!
//! Numerics:
//!   Resampling requires comparisons of normalised likelihoods. These may become insignificant if
//!   likelihoods have a large range. Resampling becomes ill conditioned for these samples.

use na::allocator::Allocator;
use na::storage::Storage;
use na::{DefaultAllocator, Dim, MatrixN, RealField, VectorN, U1};
use nalgebra as na;
use nalgebra::MatrixMN;
use num_traits::real::Real;
use num_traits::{Pow, ToPrimitive};
use rand_core::RngCore;
use rand_distr::{Distribution, StandardNormal, Uniform};

use crate::models::{Estimator, KalmanEstimator, KalmanState};
use crate::noise::{CorrelatedNoise, CoupledNoise};

/// Sample state.
///
/// State distribution is represented as state samples and their likelihood.
pub struct SampleState<N: RealField, D: Dim>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// State samples
    pub s: Samples<N, D>,
    /// and their likelihoods (bootstrap weights)
    pub w: Likelihoods,
    /// A PRNG use to draw random samples
    pub rng: Box<dyn RngCore>,
}

/// State samples.
pub type Samples<N, D> = Vec<VectorN<N, D>>;

/// likelihoods.
pub type Likelihoods = Vec<f32>;

/// Resample count.
pub type Resamples = Vec<u32>;

/// A resampling function.
pub type Resampler =
    dyn FnMut(&mut Likelihoods, &mut dyn RngCore) -> Result<(Resamples, u32, f32), &'static str>;

/// A roughening function.
pub type Roughener<N, D> = dyn FnMut(&mut Samples<N, D>, &mut dyn RngCore);

impl<N: RealField, D: Dim> SampleState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    /// Creates a [SampleState] with equal likelihood weights.
    pub fn new_equal_likelihood(s: Samples<N, D>, rng: Box<dyn RngCore>) -> SampleState<N, D> {
        let samples = s.len();
        SampleState {
            s,
            w: vec![1f32; samples],
            rng,
        }
    }

    /// Predict sample state using a state prediction function 'f'.
    pub fn predict(&mut self, f: impl Fn(&VectorN<N, D>) -> VectorN<N, D>) {
        self.s.iter_mut().for_each(|el| el.copy_from(&f(el)));
    }

    /// Predict sample state using a sampled state prediction function 'f'.
    /// The sampling function should predict the state and sample any noise.
    pub fn predict_sampled(
        &mut self,
        f: impl Fn(&VectorN<N, D>, &mut dyn RngCore) -> VectorN<N, D>,
    ) {
        // Predict particles s using supplied prediction function
        for si in 0..self.s.len() {
            let ps = f(&self.s[si], &mut self.rng);
            self.s[si].copy_from(&ps)
        }
    }

    /// Observe sample likehoods using a likelihood function 'l'.
    /// The sample likelihoods are multiplied by the observed likelihoods.
    pub fn observe<LikelihoodFn>(&mut self, l: LikelihoodFn)
    where
        LikelihoodFn: Fn(&VectorN<N, D>) -> f32,
    {
        let mut wi = self.w.iter_mut();
        for si in self.s.iter() {
            let w = wi.next().unwrap();
            *w *= l(si);
        }
    }

    /// Observe sample likehoods directly.
    /// The sample likelihoods are multiplied by these likelihoods.
    pub fn observe_likelihood(&mut self, l: Likelihoods) {
        assert_eq!(self.w.len(), l.len());
        let mut li = l.iter();
        for wi in self.w.iter_mut() {
            *wi *= li.next().unwrap();
        }
    }

    /// Resample using likelihoods and roughen the sample state.
    /// Error returns:
    ///   When the resampler fails due to numeric problems with the likelihoods
    /// Returns:
    ///  number of unique samples,
    ///  smallest normalised likelohood, to determine numerical conditioning of likehoods
    pub fn update_resample(
        &mut self,
        resampler: &mut Resampler,
        roughener: &mut Roughener<N, D>,
    ) -> Result<(u32, f32), &'static str> {
        // Resample based on likelihoods
        let (resamples, unqiue_samples, lcond) = resampler(&mut self.w, self.rng.as_mut())?;

        // Select live sample
        live_samples(&mut self.s, &resamples);
        // Resampling results in uniform likelihoods
        self.w.fill(1.);

        // Roughen samples
        roughener(&mut self.s, self.rng.as_mut());

        Ok((unqiue_samples, lcond))
    }
}

impl<N: RealField, D: Dim> Estimator<N, D> for SampleState<N, D>
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

/// Standard resampler from [1].
/// Algorithm:
/// A sample is chosen once for each time its cumulative likelihood intersects with a uniform random draw.
/// Complexity is that of Vec::sort, O(n * log(n)) worst-case.
/// This complexity is required to sort the uniform random draws made,
/// this allows comparing of the two ordered lists w(cumulative) and ur (the sorted random draws).
///
/// Returns:
/// number of times this particle should be resampled,
/// number of unqiue particles (number of non zeros in resamples),
/// conditioning of the likelihoods (min likelihood / sum likelihoods)
///
/// Side effects:
/// 'l' becomes a normalised cumulative sum,
/// Draws are made from 'rng' for each likelihood
pub fn standard_resampler(
    w: &mut Likelihoods,
    rng: &mut dyn RngCore,
) -> Result<(Resamples, u32, f32), &'static str> {
    let (lmin, lcum) = cumaltive_likelihood(w)?;

    // Sorted uniform random distribution [0..1) for each resample
    let uniform01: Uniform<f32> = Uniform::new(0f32, 1f32);
    let mut ur: Vec<f32> = uniform01.sample_iter(rng).take(w.len()).collect();
    ur.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(*ur.first().unwrap() >= 0. && *ur.last().unwrap() < 1.); // very bad if random is incorrect

    // Scale ur to cumulative sum
    ur.iter_mut().for_each(|el| *el *= lcum);

    // Resamples based on cumulative likelihood from sorted resample random values
    let mut uri = ur.iter();
    let mut urn = uri.next();
    let mut unique: u32 = 0;
    let mut resamples = Resamples::with_capacity(w.len());

    for wi in w.iter() {
        let mut res: u32 = 0; // assume not resampled until find out otherwise
        if (urn.is_some()) && *urn.unwrap() < *wi {
            unique += 1;
            loop {
                // count resamples
                res += 1;
                urn = uri.next();
                if urn.is_none() {
                    break;
                }
                if *urn.unwrap() >= *wi {
                    break;
                }
            }
        }
        resamples.push(res);
    }

    if uri.peekable().peek().is_some() {
        // resample failed due no non numeric likelhoods
        return Err("likelihoods are not numeric and cannot be resampled");
    }

    Ok((resamples, unique, lmin / lcum))
}

/// Systematic resample algorithm from [2].
/// Algorithm:
/// A particle is chosen once for each time its cumulative likelihood intersects with an equidistant grid.
/// A uniform random draw is chosen to position the grid within the cumulative likelihoods.
/// Complexity O(n)
///
/// Returns:
/// number of times this particle should be resampled,
/// number of unqiue particles (number of non zeros in resamples),
/// conditioning of the likelihoods (min likelihood / sum likelihoods)
///
/// Side effects:
/// 'l' becomes a normalised cumulative sum,
/// Draws are made from 'rng' for each likelihood
pub fn systematic_resampler(
    l: &mut Likelihoods,
    rng: &mut dyn RngCore,
) -> Result<(Resamples, u32, f32), &'static str> {
    let (lmin, lcum) = cumaltive_likelihood(l)?;

    let uniform01: Uniform<f32> = Uniform::new(0f32, 1f32);

    // Setup grid
    let glen = l.len();
    let gstep = lcum / glen as f32;
    let goffset = uniform01.sample(rng) * gstep; // random offset

    // Resamples based on cumulative likelihoods
    let mut resamples = Resamples::with_capacity(glen);
    let mut unique: u32 = 0;

    let mut gi: u32 = 0;
    for li in l.iter() {
        let mut res: u32 = 0; // assume not resampled until find out otherwise
        if (goffset + lcum * gi as f32 / glen as f32) < *li {
            unique += 1;
            loop {
                // count resamples
                res += 1;
                gi += 1;
                if (goffset + lcum * gi as f32 / glen as f32) >= *li {
                    break;
                }
            }
        }
        resamples.push(res);
    }

    Ok((resamples, unique, lmin / lcum))
}

/// Normalised cumulative sum of likelihoods (Kahan algorithm), and find smallest likelihood.
fn cumaltive_likelihood(l: &mut Likelihoods) -> Result<(f32, f32), &'static str> {
    let mut lmin = f32::max_value();
    let mut lcum = 0.;
    {
        let mut c = 0.;
        for li in l.iter_mut() {
            if *li < lmin {
                lmin = *li;
            }
            let y = *li - c;
            let t = lcum + y;
            c = t - lcum - y;
            lcum = t;
            *li = t;
        }
    }
    if lmin < 0. {
        // bad likelihoods
        return Err("negative likelihood");
    }
    if lcum <= 0. {
        // bad cumulative likelihood (previous check should actually prevent -ve
        return Err("zero cumulative likelihood sum");
    }
    // Any numerical failure should cascade into cumulative sum
    if lcum.is_nan() {
        // inequality due to NaN
        return Err("NaN cumulative likelihood sum");
    }
    Ok((lmin, lcum))
}

/// Update 's' by selectively copying resamples.
/// Uses a in-place copying algorithm:
/// First copy the live samples (those resampled) to end of s.
/// Replicate live sample in-place start an the begining of s.
pub fn live_samples<N: RealField, D: Dim>(s: &mut Samples<N, D>, resamples: &[u32])
where
    DefaultAllocator: Allocator<N, D>,
{
    // reverse_copy_if live
    let mut si = s.len();
    let mut livei = si;
    for pr in resamples.iter().rev() {
        si -= 1;
        if *pr > 0 {
            livei -= 1;
            s[livei] = s[si].clone();
        }
    }
    assert_eq!(si, 0);

    // Replicate live samples
    si = 0;
    for pr in resamples {
        let mut res = *pr;
        if res > 0 {
            loop {
                s[si] = s[livei].clone();
                si += 1;
                res -= 1;
                if res == 0 {
                    break;
                }
            }
            livei += 1;
        }
    }
    assert_eq!(si, s.len());
    assert_eq!(livei, s.len());
}

/// Roughen sample state using min max roughening of the samples.
///
/// Uses algorithm from Ref[1] which applies then the states have a singe locality but the distribution is unknown.
/// max-min in each state dimension in the samples determines the amount of normally distributed noise added to that
/// dimension for each sample.
///
/// 'k' is scaling factor for normally distributed noise
///
/// Numerics:
///  If the are very few unique samples the roughening will colapse as it is not representative of the true state distribution.
pub fn roughen_minmax<N: RealField, D: Dim>(s: &mut Samples<N, D>, k: f32, rng: &mut dyn RngCore)
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    let x_dim = s[0].data.shape().0;
    let x_size = x_dim.value();

    // Find min and max state dimension in all states
    let mut xmin = s[0].clone();
    let mut xmax = xmin.clone();
    for si in s.iter() {
        // Loop includes 0 to simplify code
        let mut mini = xmin.iter_mut();
        let mut maxi = xmax.iter_mut();

        for xd in si.iter() {
            let minx = mini.next().unwrap();
            let maxx = maxi.next().unwrap();

            if *xd < *minx {
                *minx = *xd;
            }
            if *xd > *maxx {
                *maxx = *xd;
            }
        }
    }
    // Roughening st.dev from scaled max-min and state dimensions
    let sigma_scale = k * f32::pow(s.len() as f32, -1f32 / (x_size as f32));
    let sigma = (xmax - xmin) * N::from_f32(sigma_scale).unwrap();
    let noise = normal_noise_sampler(sigma);

    for si in s.iter_mut() {
        *si += noise(&mut *rng);
    }
}

/// Couple noise roughening.
///
/// The roughening is scaled by the deviation of the noise.
///
/// 'k' is scaling factor for normally distributed noise
pub fn roughen_noise<N: RealField, D: Dim, ND: Dim>(
    s: &mut Samples<N, D>,
    noise: &CoupledNoise<N, D, ND>,
    k: f32,
    rng: &mut dyn RngCore,
) where
    DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, D, ND>
        + Allocator<N, ND, ND>
        + Allocator<N, D>
        + Allocator<N, ND>,
{
    // Roughening st.dev from scaled  and state dimensions
    let sigma_scale =
        N::from_f32(k * f32::pow(s.len() as f32, -1f32 / (noise.G.nrows() as f32))).unwrap();
    let mut coupling = noise.G.clone();
    for (i, mut c) in coupling.column_iter_mut().enumerate() {
        c *= sigma_scale * noise.q[i];
    }
    let noise = normal_noise_sampler_coupled(coupling);

    for si in s.iter_mut() {
        *si += noise(&mut *rng);
    }
}

impl<N: RealField, D: Dim> KalmanEstimator<N, D> for SampleState<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, U1, D> + Allocator<N, D>,
{
    fn init(&mut self, state: &KalmanState<N, D>) -> Result<(), &'static str> {
        for s in self.s.iter_mut() {
            s.copy_from(&state.x);
        }
        let coupled_noise = CoupledNoise::from_correlated(&CorrelatedNoise { Q: state.X.clone() })?;
        let sampler = normal_noise_sampler_coupled(coupled_noise.G);
        self.predict_sampled(
            move |x: &VectorN<N, D>, rng: &mut dyn RngCore| -> VectorN<N, D> { x + sampler(rng) },
        );

        Ok(())
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

/// Generate as sampling function for normally distributed noise.
///
/// 'std_dev' standard deviation of normally distributed noise.
pub fn normal_noise_sampler<N: RealField, D: Dim>(
    std_dev: VectorN<N, D>,
) -> impl Fn(&mut dyn RngCore) -> VectorN<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    // Sample with the coupled noise deviation
    move |rng: &mut dyn RngCore| -> VectorN<N, D> {
        let rnormal = StandardNormal
            .sample_iter(rng)
            .enumerate()
            .map(|(i, n): (usize, f32)| N::from_f32(n).unwrap() * std_dev[i])
            .take(std_dev.nrows());
        VectorN::from_iterator_generic(std_dev.data.shape().0, U1, rnormal)
    }
}

/// Generate as sampling function for normally distributed coupled noise.
pub fn normal_noise_sampler_coupled<N: RealField, D: Dim, ND: Dim>(
    coupleing: MatrixMN<N, D, ND>,
) -> impl Fn(&mut dyn RngCore) -> VectorN<N, D>
where
    DefaultAllocator: Allocator<N, D, ND> + Allocator<N, D> + Allocator<N, ND>,
{
    // Sample with the coupled noise deviation
    move |rng: &mut dyn RngCore| -> VectorN<N, D> {
        let rnormal = StandardNormal
            .sample_iter(rng)
            .map(|n: f32| N::from_f32(n).unwrap())
            .take(coupleing.ncols());
        let n = VectorN::from_iterator_generic(coupleing.data.shape().1, U1, rnormal);
        &coupleing * n
    }
}

pub fn gaussian_observation_likelihood<'r, N: RealField + ToPrimitive, D: Dim, ZD: Dim>(
    z: &'r VectorN<N, ZD>,
    h: impl Fn(&VectorN<N, D>) -> VectorN<N, ZD> + 'r,
    noise: &CorrelatedNoise<N, ZD>,
) -> impl Fn(&VectorN<N, D>) -> f32 + 'r
where
    DefaultAllocator:
        Allocator<N, D, D> + Allocator<N, ZD, ZD> + Allocator<N, D> + Allocator<N, ZD>,
{
    // Observation Likihood for correlated Gaussian noise
    let cholesky = noise.Q.clone().cholesky().unwrap();
    let zinv = cholesky.inverse();
    let zinv_diagonal = cholesky.l_dirty().iter().step_by(noise.Q.nrows() + 1);
    let determinate_zinv = zinv_diagonal
        .fold(N::one(), |prod: N, n: &N| prod * *n)
        .to_f32()
        .unwrap();

    move |x: &VectorN<N, D>| -> f32 {
        let innov = z - h(x);
        let logl = innov.dot(&(&zinv * &innov)).to_f32().unwrap();
        (-0.5 * (logl + determinate_zinv.ln())).exp()
    }
}
