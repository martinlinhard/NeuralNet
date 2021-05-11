mod gpu_neural_net;
mod matrix;
mod neural_net;

use arrayfire::*;
use gpu_neural_net::GpuNeuralNet;
use neural_net::NeuralNet;

pub use crate::matrix::Matrix;
use std::f64::consts::E;

fn main() {
    test_normal();
    test_gpu();
}

fn test_normal() {
    let input = Matrix::new([[0.9], [0.1], [0.8]]);

    let wih = Matrix::new([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]]);

    let who = Matrix::new([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]]);

    let sigmoid = |input: &f64| (1.0 / (1.0 + E.powf(*input * -1.0)));

    let mut neural_net = NeuralNet::new(sigmoid);
    neural_net.add_layer(wih);
    neural_net.add_layer(who);

    let result = neural_net.forward_propagation(input);
    println!("{:?}", result);

    //let expected_result = Matrix::new([[1.0], [1.0], [1.0]]);

    //neural_net.backward_propagation(&result, &expected_result);
}

fn test_gpu() {
    set_device(0);
    let input = Array::new(&[0.9, 0.1, 0.8], Dim4::new(&[3, 1, 1, 1]));
    let wih = Array::new(
        &[0.9, 0.2, 0.1, 0.3, 0.8, 0.5, 0.4, 0.2, 0.6],
        Dim4::new(&[3, 3, 1, 1]),
    );
    let who = Array::new(
        &[0.3, 0.6, 0.8, 0.7, 0.5, 0.1, 0.5, 0.2, 0.9],
        Dim4::new(&[3, 3, 1, 1]),
    );

    let mut neural_net = GpuNeuralNet::new(sigmoid);
    neural_net.add_layer(wih);
    neural_net.add_layer(who);

    let result = neural_net.forward_propagation(input);
    print(&result);
}
