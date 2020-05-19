use super::{sigmoid, sigmoid_derivative, Input, MLP};
use itertools::izip;
use nalgebra::{DMatrix, DVector};

impl MLP {
    /// Computes the cost of the network for one input
    ///
    /// Outputs a tuple containing:
    /// - the total cost of the network for the input
    /// - a `DVector` of the costs of each neuron for that input
    pub fn cost_single_case(&self, input: &Input, expected_class: usize) -> (f64, DVector<f64>) {
        let output = self.classify(input);
        assert!(expected_class < output.len());

        let costs = DVector::from_row_slice(
            &output
                .iter()
                .enumerate()
                .map(|(class, activation)| {
                    let expected = if class == expected_class { 1. } else { 0. };
                    (activation - expected).powf(2.)
                })
                .collect::<Vec<f64>>(),
        );

        (costs.sum(), costs)
    }

    /// Computes the average cost of the network for the dataset
    ///
    /// Outputs a tuple containing:
    /// - the total cost of the network for the input
    /// - a `DVector` of the costs of each neuron for that input
    pub fn cost_dataset(&self, dataset: &[Input], expected: &[usize]) -> (f64, DVector<f64>) {
        assert!(dataset.len() != 0);
        assert_eq!(dataset.len(), expected.len());

        let costs: DVector<f64> = dataset
            .iter()
            .zip(expected.iter())
            .map(|(input, expected_class)| self.cost_single_case(input, *expected_class).1)
            .sum();

        (
            costs.sum() / (dataset.len() as f64),
            costs / (dataset.len() as f64),
        )
    }

    /// Tries to classify the input and backpropagates the error once
    pub fn train_case(&mut self, input: &Input, expected_class: usize) {
        // ---- Feed forward the input ----
        let mut activations = vec![input.clone()];
        let mut outputs = vec![]; // each layer's z vector (output before the activation function)
        for (weights, biases) in self.weights.iter().zip(&self.biases) {
            // Compute the output of a layer
            outputs.push(weights * activations.last().unwrap() + biases);
            // Activate the neurons and save their activations
            activations.push(outputs.last().unwrap().map(sigmoid));
        }
        let output = activations.pop().unwrap();

        // ---- Compute the output error ----
        // adjustments to apply to weights, in reverse order of layer
        let mut nabla_weights: Vec<DMatrix<f64>> = vec![];
        // adjustments to apply to biases, in reverse order of layer
        let mut nabla_biases: Vec<DVector<f64>> = vec![];

        // the correct network output would be every output neuron at 0 except the one
        // corresponding to the input's class, which would be 1
        let mut expected = Input::zeros(output.nrows());
        expected[expected_class] = 1.;

        // compute the output layer's delta and adjustments
        // `delta` contains numbers representing the magnitude of the impact a nudge to the
        // corresponding neuron would have on the network's cost
        let last_output = outputs.pop().unwrap();
        let mut delta = (expected - output).component_mul(&last_output.map(sigmoid_derivative));
        nabla_biases.push(delta.clone());
        nabla_weights.push(&delta * activations.pop().unwrap().transpose());

        // ---- Backpropagate the error ----
        // We already computed the adjustments to make to the last layer.
        // To compute the adjustments to make to some layer, we need the weights of the following
        // one.
        // We'll iterate over the weights that we need to compute those adjustments: from the last
        // layer to the second one.
        for weights in self.weights.iter().skip(1).rev() {
            let output = outputs.pop().unwrap();
            let sp = output.map(sigmoid_derivative);
            delta = (weights.transpose() * delta).component_mul(&sp);
            nabla_biases.push(delta.clone());
            nabla_weights.push(&delta * activations.pop().unwrap().transpose());
        }

        // Apply the nablas to each layer
        let nabla_iterator = izip!(
            self.weights.iter_mut(),
            self.biases.iter_mut(),
            nabla_weights.iter().rev(),
            nabla_biases.iter().rev()
        );
        for (layer_w, layer_b, layer_nw, layer_nb) in nabla_iterator {
            *layer_w += layer_nw;
            *layer_b += layer_nb;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::mlp;
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    #[should_panic]
    fn cost_single_case_wrong_input_size() {
        let network = mlp![5; 6];
        network.cost_single_case(&Input::zeros(6), 0);
    }

    #[test]
    #[should_panic]
    fn cost_single_case_output_too_high() {
        let network = mlp![5; 6];
        network.cost_single_case(&Input::zeros(5), 6);
    }

    fn get_good_xor_network() -> MLP {
        // Built using a slightly modified version of the MLP training from
        // https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d
        MLP {
            weights: vec![
                DMatrix::from_row_slice(2, 2, &[6.7, 6.7, 4.6, 4.6]),
                DMatrix::from_row_slice(2, 2, &[-7., 7.5, 7., -7.5]),
            ],
            biases: vec![
                DVector::from_row_slice(&[-3., -7.]),
                DVector::from_row_slice(&[3.2, -3.2]),
            ],
        }
    }

    fn get_bad_xor_network() -> MLP {
        // Built using a slightly modified version of the MLP training from
        // https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d
        // Gives the opposite answer for each of the 4 possible inputs
        MLP {
            weights: vec![
                DMatrix::from_row_slice(2, 2, &[6.5, 6.6, 4.3, 4.4]),
                DMatrix::from_row_slice(2, 2, &[6.7, -7.2, -6.7, 7.2]),
            ],
            biases: vec![
                DVector::from_row_slice(&[-2.9, -6.7]),
                DVector::from_row_slice(&[-3., 3.]),
            ],
        }
    }

    #[test]
    fn cost_single_case_valid_good_network() {
        // This test computes the cost of a zeroed MLP against the different cases of the XOR operation
        let network = get_good_xor_network();

        let dataset = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let expected_set: [usize; 4] = [0, 1, 1, 0];
        for (input, expected) in dataset.iter().zip(expected_set.iter()) {
            let (cost, _) = network.cost_single_case(&Input::from_row_slice(input), *expected);
            assert!(cost < 0.1, "cost is {}, should be close to 0", cost);
        }
    }

    #[test]
    fn cost_single_case_valid_bad_network() {
        // This test computes the cost of a zeroed MLP against the different cases of the XOR operation
        let network = get_bad_xor_network();

        let dataset = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let expected_set: [usize; 4] = [0, 1, 1, 0];
        for (input, expected) in dataset.iter().zip(expected_set.iter()) {
            let (cost, _) = network.cost_single_case(&Input::from_row_slice(input), *expected);
            assert!(cost > 1., "cost is {}, should be greater than 1", cost);
        }
    }

    #[test]
    #[should_panic]
    fn cost_dataset_empty_input_set() {
        let network = mlp![5; 6];
        network.cost_dataset(&[], &[0]);
    }

    #[test]
    #[should_panic]
    fn cost_dataset_empty_output_set() {
        let network = mlp![5; 6];
        network.cost_dataset(&[Input::from_row_slice(&[0.])], &[]);
    }

    #[test]
    #[should_panic]
    fn cost_dataset_input_output_sets_len_mismatch() {
        let network = mlp![5; 6];
        network.cost_dataset(&[Input::zeros(5), Input::zeros(5)], &[0, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn cost_dataset_some_wrong_input() {
        let network = mlp![5; 6];
        network.cost_dataset(&[Input::zeros(5), Input::zeros(6)], &[0, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn cost_dataset_some_wrong_output() {
        let network = mlp![5; 6];
        network.cost_dataset(&[Input::zeros(5), Input::zeros(5)], &[0, 6, 0]);
    }

    #[test]
    fn cost_dataset_valid_good_network() {
        let dataset = [
            Input::from_row_slice(&[0., 0.]),
            Input::from_row_slice(&[0., 1.]),
            Input::from_row_slice(&[1., 0.]),
            Input::from_row_slice(&[1., 1.]),
        ];
        let network = get_good_xor_network();
        let (cost, _) = network.cost_dataset(&dataset, &[0, 1, 1, 0]);
        assert!(cost < 0.1, "cost is {}, should be close to 0", cost)
    }

    #[test]
    fn cost_dataset_valid_bad_network() {
        let dataset = [
            Input::from_row_slice(&[0., 0.]),
            Input::from_row_slice(&[0., 1.]),
            Input::from_row_slice(&[1., 0.]),
            Input::from_row_slice(&[1., 1.]),
        ];
        let network = get_bad_xor_network();
        let (cost, _) = network.cost_dataset(&dataset, &[0, 1, 1, 0]);
        assert!(cost > 1., "cost is {}, should be greater than 1", cost)
    }

    #[test]
    fn train_case_valid() {
        // just verify that it doesn't panic
        let mut network = mlp![50; 10];
        network.train_case(&Input::zeros(50), 0);
        network.train_case(&Input::zeros(50), 5);
        network.train_case(&Input::zeros(50), 9);

        let mut network = mlp![10, 5, 6, 8, 5, 5];
        network.train_case(&Input::zeros(10), 0);
        network.train_case(&Input::zeros(10), 2);
        network.train_case(&Input::zeros(10), 4);

        let mut network = mlp![50, 10]; // no hidden layer
        network.train_case(&Input::zeros(50), 5);
        network.train_case(&Input::zeros(50), 0);
        network.train_case(&Input::zeros(50), 9);
    }
}
