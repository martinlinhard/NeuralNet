use crate::Matrix;

pub struct NeuralNet<F> {
    activation_function: F,
    input: Matrix,
    layers: Vec<Matrix>,
}

impl<F> NeuralNet<F> {
    pub fn new(input: Matrix, activation_function: F) -> Self {
        Self {
            input,
            activation_function,
            layers: vec![],
        }
    }

    pub fn add_layer(&mut self, layer: Matrix) {
        self.layers.push(layer);
    }

    pub fn calculate(self) -> Matrix
    where
        F: Fn(&f64) -> f64,
    {
        let NeuralNet {
            activation_function,
            input,
            layers,
        } = self;

        layers
            .into_iter()
            .fold(input, move |previous, current_layer| {
                let mut result = current_layer * previous;
                result.apply_to_each(&activation_function);
                result
            })
    }
}
