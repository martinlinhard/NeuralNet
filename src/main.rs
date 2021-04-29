#![allow(dead_code)]
mod matrix;

pub use crate::matrix::Matrix;

fn main() {
    let m: Matrix<3, 3> = get_matrix_1();
    let r: Matrix<3, 3> = get_matrix_1();

    println!("{:?}", m * r);
    //let r: Matrix<2, 1> = Matrix::new_zeroed();
}

fn get_matrix_1() -> Matrix<3, 3> {
    let mut out = Matrix::new_zeroed();
    out.set_at_position(0, 0, 2.0);
    out.set_at_position(0, 1, 4.0);
    out.set_at_position(0, 2, 7.0);

    out.set_at_position(1, 0, 8.0);
    out.set_at_position(1, 1, -1.0);
    out.set_at_position(1, 2, 2.0);

    out.set_at_position(2, 0, -1.0);
    out.set_at_position(2, 1, 4.0);
    out.set_at_position(2, 2, 3.0);
    out
}

fn get_matrix_2() -> Matrix<2, 1> {
    let mut out = Matrix::new_zeroed();
    out.set_at_position(0, 0, 1.0);
    out.set_at_position(1, 0, 0.5);
    out
}
