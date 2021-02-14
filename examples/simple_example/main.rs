//! Example of using Bayesian State Estimator to solve a simple problem.
//! A linear filter with one state and constant noises.

use nalgebra as na;
use na::{Matrix1, Vector1};

use bayes_estimate::models::{
    KalmanState, LinearObserveModel, LinearPredictModel, LinearObserver, CorrelatedNoise, LinearPredictor
};

fn main() {
    // Construct simple Prediction and Observation models
    let my_predict_model = LinearPredictModel {
        Fx: Matrix1::new(1.)
    };
    let my_predict_noise = CorrelatedNoise {
        Q: Matrix1::new(1.),
    };
    let my_observe_model = LinearObserveModel {
        Hx: Matrix1::new(1.),
    };
    let my_observe_noise = CorrelatedNoise {
        Q: Matrix1::new(1.),
    };

    // Setup the initial state and covariance
    let mut my_filter = KalmanState {
        x: Vector1::new(10.),
        X: Matrix1::new(0.)
    };
    println!("Initial {:.1}{:.2}", my_filter.x, my_filter.X);

    // Predict the filter forward
    my_filter.predict (&my_predict_model, my_predict_model.Fx * my_filter.x, &my_predict_noise).unwrap();
    println!("Predict {:.1}{:.2}", my_filter.x, my_filter.X);

    // Make an observation that we should be at 11
    let z = Vector1::new(11.);
    let innovation = z - &my_observe_model.Hx * &my_filter.x;
    my_filter.observe_innovation (&my_observe_model, &my_observe_noise, &innovation).unwrap();
    println!("Filtered {:.1}{:.2}", my_filter.x, my_filter.X);
}