//! Operation of a Bayesian sample state estimator in a simple example.
//!
//! A Sampling Importance Resampling position estimator.

use na::{U2, Vector2};
use nalgebra as na;
use num_traits::Pow;
use rand::{Rng, RngCore};

use bayes_estimate::estimators::sir;
use bayes_estimate::models::Estimator;
use sir::SampleState;

fn main() {
    // We need random numbers
    let mut rng= rand::thread_rng();
    // And a numeric type
    type N = f64;

    // Setup the initial position to be in a box 1000 by 1000
    let range1000 = rand_distr::Uniform::new(0., 1000.);
    let mut samples: sir::Samples<N, U2> = vec!();
    for _i in 0..10000 {
        samples.push(Vector2::new(rng.sample(range1000), rng.sample(range1000)))
    }
    // Assume any position in the box is initialy equally likely
    let mut estimate = SampleState::new_equal_likelihood(samples, Box::new(rng));

    // We make an observation that we appear to be in a circle 50 wide, positioned at 100,100
    let circle_at_100 = |state: &Vector2<N>| {
        let dist2: N = (state[0] - 100.).pow(2) + (state[1] - 100.).pow(2);
        if dist2.sqrt() <= 50. { 1. } else { 0. }
    };
    estimate.observe(circle_at_100);

    // Update our position estimate, for this we need a resampler and a roughener
    let mut roughener= |s: &mut sir::Samples<N, U2>, rng: &mut dyn RngCore| {
        sir::roughen_minmax(s, 1., rng)
    };
    estimate.update_resample(&mut sir::standard_resampler, &mut roughener).unwrap();
    println!("{}", estimate.state().unwrap());

    // Now we have moved, with some uncertainty
    let range20 = rand_distr::Uniform::new(-10., 10.);
    estimate.predict_sampled(|s: &Vector2<N>, rng: &mut dyn RngCore|
        s + Vector2::new(100. + rng.sample(range20), 50. + rng.sample(range20)));
    println!("{}", estimate.state().unwrap());
}