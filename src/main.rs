mod matrix;
mod neural_net;

use neural_net::NeuralNet;

pub use crate::matrix::Matrix;
use std::f64::consts::E;

fn main() {
    let input = Matrix::new([[0.9], [0.1], [0.8]]);

    let wih = Matrix::new([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]]);

    let who = Matrix::new([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]]);

    let sigmoid = |input: &f64| (1.0 / (1.0 + E.powf(*input * -1.0)));

    let mut neural_net = NeuralNet::new(sigmoid);
    neural_net.add_layer(wih);
    neural_net.add_layer(who);

    let result = neural_net.calculate(input);

    println!("{:?}", result);
}
