//! A collection of Bayesian state estimators.

pub mod covariance;
pub mod ud;
pub mod unscented;
pub mod information;
pub mod information_root;
#[cfg(feature = "std")]
pub mod sir;
