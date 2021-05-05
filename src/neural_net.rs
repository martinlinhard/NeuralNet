use crate::Matrix;

pub struct NeuralNet<F> {
    activation_function: F,
    layers: Vec<Matrix>,
}

impl<F> NeuralNet<F> {
    pub fn new(activation_function: F) -> Self {
        Self {
            activation_function,
            layers: vec![],
        }
    }

    pub fn add_layer(&mut self, layer: Matrix) {
        self.layers.push(layer);
    }

    pub fn calculate(&self, input: Matrix) -> Matrix
    where
        F: Fn(&f64) -> f64,
    {
        self.layers.iter().fold(input, |previous, current_layer| {
            let mut result = current_layer * &previous;
            result.apply_to_each(&self.activation_function);
            result
        })
    }
}
