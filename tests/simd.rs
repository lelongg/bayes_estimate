use simba::simd::WideF32x4;

use bayes_filter as bf;
use bf::models::{
    KalmanState
};
use nalgebra::{Matrix2, Vector2};

#[test]
fn simd() {
    let w = WideF32x4::from([1.0, 2.0, 3.0, 4.0]);
    let x = Vector2::new(w, w);
    let xx = Matrix2::new(w, w, w, w);
    let _ks = KalmanState{ x: x, X: xx};
}
