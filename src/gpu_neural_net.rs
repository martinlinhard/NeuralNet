use arrayfire::{matmul, print, sigmoid, Array};

use crate::Matrix;

pub struct GpuNeuralNet<F> {
    activation_function: F,
    layers: Vec<Array<f64>>,
}

impl<F> GpuNeuralNet<F> {
    pub fn new(activation_function: F) -> Self {
        Self {
            activation_function,
            layers: vec![],
        }
    }

    pub fn add_layer(&mut self, layer: Array<f64>) {
        self.layers.push(layer);
    }

    pub fn forward_propagation(&self, input: Array<f64>) -> Array<f64>
    where
        F: Fn(&Array<f64>) -> Array<f64>,
    {
        self.layers.iter().fold(input, |previous, current_layer| {
            let result = matmul(
                &current_layer,
                &previous,
                arrayfire::MatProp::NONE,
                arrayfire::MatProp::NONE,
            );
            (self.activation_function)(&result)
        })
    }

    pub fn backward_propagation(&self, actual_result: &Array<f64>, expected_result: &Array<f64>) {
        let error = expected_result - actual_result;

        // visit layers in reverse order; multiply them by the previous error
        self.layers
            .iter()
            .rev()
            .fold(error, |previous_error, current_layer| {
                matmul(
                    &current_layer,
                    &previous_error,
                    arrayfire::MatProp::NONE,
                    arrayfire::MatProp::NONE,
                )
            });
    }
}
