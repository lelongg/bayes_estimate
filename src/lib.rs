#![feature(total_cmp)]

//! Bayes+Estimate the Bayesian estimation library.
//! Copyright (c) 2020 Michael Stevens
//!
//! # Licensing
//!
//! All Bayes+Estimate source code files are copyright with the license conditions as given here. The copyright notice is that of the MIT license.
//!  This in no way restricts any commercial use you may wish to make using our source code.
//!  As long as you respect the copyright and license conditions, Michael Stevens is happy to for you to use it in any way you wish.
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction,
//! including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
//! and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
//! WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod models;
pub mod noise;
pub mod estimators;
mod linalg;
mod matrix;

pub use linalg::cholesky;
