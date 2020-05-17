use super::{Input, MLP};
use nalgebra::DVector;

impl MLP {
    /// Computes the cost of the network for one input
    ///
    /// Outputs a tuple containing:
    /// - the total cost of the network for the input
    /// - a `DVector` of the costs of each neuron for that input
    fn cost_single_case(&self, input: &Input, expected_class: usize) -> (f64, DVector<f64>) {
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
    fn cost_dataset(&self, dataset: &[Input], expected: &[usize]) -> (f64, DVector<f64>) {
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
}
