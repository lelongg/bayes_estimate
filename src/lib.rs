///
/// Bayes+Estimate the Bayesian estimation library.
/// Copyright (c) 2020 Michael Stevens
///
/// Bayesian Filtering is a probabilistic technique for data fusion. The technique combines a concise mathematical formulation of a system with observations of that system.
/// Probabilities are used to represent the state of a system, likelihood functions to represent their relationships.
/// In this form Bayesian inference can be applied and further related probabilities deduced. See Wikipedia for information on Probability theory, Bayes theorem, Bayesian Inference.
///
/// For discrete systems the Bayesian formulation results in a naturally iterative data fusion solution. For dynamic systems there is a class of solutions, discrete filters,
/// that combine observed outputs of the system with the system's dynamic model. An estimator computes a estimate of the systems state with each observation of the system.
/// Linear estimators such as the Kalman Filter are commonly applied.
///
/// Bayes+Estimate is an open source library for Rust. The library implements a wide variety of numerical algorithms for Bayesian estimation of discrete systems.
///
/// Prediction and observation models are represented by a hierarchy of traits that define the estimation operations for different models.
/// State represention are definied by structs.
/// Estimators implement the models for their state representation and provide numerical implementation of the operations.
//
/// # Licensing
///
/// All Bayes++ source code files are copyright with the license conditions as given here. The copyright notice is that of the MIT license.
///  This in no way restricts any commercial use you may wish to make using our source code.
///  As long as you respect the copyright and license conditions, Michael Stevens is happy to for you to use it in any way you wish.
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction,
/// including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
/// and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
/// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

pub mod models;
pub mod covariance_filter;
pub mod ud_filter;
pub mod information_filter;
pub mod linalg;
mod mine;