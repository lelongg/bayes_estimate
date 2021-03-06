# Bayes+Estimate the Bayesian estimation library.

Bayesian estimation is a probabilistic technique for data fusion. The technique combines a concise mathematical formulation of a system with observations of that system.
Probabilities are used to represent the state of a system, likelihood functions to represent their relationships.
In this form Bayesian inference can be applied and further related probabilities deduced. See Wikipedia for information on probability theory, Bayes theorem, Bayesian inference.

For discrete systems the Bayesian formulation results in a naturally iterative data fusion solution. For dynamic systems there is a class of solutions, discrete filters,
that combine observed outputs of the system with the system's dynamic model. An estimator computes a estimate of the systems state with each observation of the system.
Linear estimators such as the Kalman Filter are commonly applied.

Bayes+Estimate is an open source library for Rust. The library implements a wide variety of numerical algorithms for Bayesian estimation of discrete systems.

The following linear estimator are implemented for linear or linearised models:
* **covariance** the classic extended Kalman filter.
* **information** representing the inverse form of state allows the additive propperties of information to be used.
* **information_root** square root factorised form of information for better numerics.
* **ud** UdU' factorised form of covariance for better numerics.
* **unscented** used 'unscented' transform to better deal with non linear models.

A Sampling Importance Resampleing estimator is implemeted for use where linearised models are not appropriate:
* **sir** a Sampling Importance Resampleing (or weighted bootstrap) estimator.

State and noise models have been logically seperated and are defined by their structures.
Prediction and observation are represented by traits that define the estimation operations for different models.
Estimators implement the models for their state representation and provide a numerical implementation of the operations.

The estimators operations provide consistent implementations which allow numerically stable estimators to be implemented.
For linear estimators the conditioning of the estimate as a reciprocal condition number can be calculated.
For the *sir* estimator the sample likelihood conditioning can be calculated.

The library supports no_std operation.

This work is based on my Bayes++ C++ Bayesian estimation library. See http://bayesclasses.sourceforge.net/Bayes++.html

Copyright (c) 2020 Michael Stevens

# Licensing

All Bayes+Estimate source code files are copyright with the license conditions as given here. The copyright notice is that of the MIT license.
This in no way restricts any commercial use you may wish to make using our source code.
As long as you respect the copyright and license conditions, Michael Stevens is happy to for you to use it in any way you wish.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
