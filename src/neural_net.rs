use crate::Matrix;

pub struct NeuralNet<F, const N: usize> {
    activation_function: F,
    input: Matrix<N, 1>,
    layers: Vec<Matrix<N, N>>,
}

impl<F, const N: usize> NeuralNet<F, N> {
    pub fn new(input: Matrix<N, 1>, activation_function: F) -> Self {
        Self {
            input,
            activation_function,
            layers: vec![],
        }
    }

    pub fn add_layer(&mut self, layer: Matrix<N, N>) {
        self.layers.push(layer);
    }

    pub fn calculate(self) -> Matrix<N, 1>
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
