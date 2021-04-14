use nalgebra::{MatrixN, U2, Vector2};

use bayes_estimate::estimators::information_root::InformationRootState;
use bayes_estimate::estimators::ud::UdState;
use bayes_estimate::models::{InformationState, KalmanState};
use bayes_estimate::models::KalmanEstimator;

#[test]
fn test_init_matches_state() {
    let mut udstate = UdState::new(
        MatrixN::<f64, U2>::new(1., 3., 0., 2.),
        Vector2::new(4., 5.)
    );
    println!("UD {:} {:}", udstate.x, udstate.UD);

    let kalman_state = udstate.kalman_state().unwrap();
    println!("Kalman {:} {:}", kalman_state.x, kalman_state.X);

    udstate.init(&kalman_state).unwrap();
    assert_kalman_eq(&kalman_state, &udstate.kalman_state().unwrap());

    let mut information_state = InformationState::new_zero(U2);
    information_state.init(&kalman_state).unwrap();
    println!("Information {:} {:}", information_state.i, information_state.I);
    assert_kalman_eq(&kalman_state, &information_state.kalman_state().unwrap());

    let mut information_root_state = InformationRootState::new_zero(U2);
    information_root_state.init(&kalman_state).unwrap();
    println!("InformationRoot from Kalman {:} {:}", information_root_state.r, information_root_state.R);
    assert_kalman_eq(&kalman_state, &information_root_state.kalman_state().unwrap());
    assert_information_eq(&information_state, &information_root_state.information_state().unwrap());

    information_root_state.init_information(&information_state).unwrap();
    println!("InformationRoot from Information {:} {:}", information_root_state.r, information_root_state.R);
    assert_kalman_eq(&kalman_state, &information_root_state.kalman_state().unwrap());
    assert_information_eq(&information_state, &information_root_state.information_state().unwrap());
}

fn assert_kalman_eq(expect: &KalmanState<f64, U2>, actual: &KalmanState<f64, U2>, )
{
    approx::assert_relative_eq!(actual.x[0], expect.x[0], max_relative = 0.00000001);
    approx::assert_relative_eq!(actual.x[1], expect.x[1], max_relative = 0.01);

    approx::assert_abs_diff_eq!(actual.X[(0,0)], expect.X[(0,0)], epsilon = 0.000001);
    approx::assert_abs_diff_eq!(actual.X[(0,1)], expect.X[(0,1)], epsilon = 0.000001);
    approx::assert_abs_diff_eq!(actual.X[(0,1)], expect.X[(1,0)], epsilon = 0.0000001);
    approx::assert_abs_diff_eq!(actual.X[(1,1)], expect.X[(1,1)], epsilon = 0.000001);
}

fn assert_information_eq(expect: &InformationState<f64, U2>, actual: &InformationState<f64, U2>, )
{
    approx::assert_relative_eq!(actual.i[0], expect.i[0], max_relative = 0.00000001);
    approx::assert_relative_eq!(actual.i[1], expect.i[1], max_relative = 0.01);

    approx::assert_abs_diff_eq!(actual.I[(0,0)], expect.I[(0,0)], epsilon = 0.000001);
    approx::assert_abs_diff_eq!(actual.I[(0,1)], expect.I[(0,1)], epsilon = 0.000001);
    approx::assert_abs_diff_eq!(actual.I[(0,1)], expect.I[(1,0)], epsilon = 0.0000001);
    approx::assert_abs_diff_eq!(actual.I[(1,1)], expect.I[(1,1)], epsilon = 0.000001);
}
