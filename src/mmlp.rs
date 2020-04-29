//! A modular implementation for Multilayer Perceptrons with efficient matrix representations and
//! manipulations.
//!
//! This crate provides the structure and functions for common manipulations of [Multilayer
//! Perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) which are a kind of
//! artificial neural network.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::error::Error;

/// The main structure representing an MLP.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct MLP {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl MLP {
    /// Returns a new MLP given an input size and a number of classes (output size).
    ///
    /// The returned MLP contains exactly one hidden layer of size between the input's and the
    /// output's one.
    ///
    /// # Panics
    /// Panics if either the input size or the output size is below 1.
    ///
    /// # Examples
    /// ```
    /// use ocrust::mmlp::MLP;
    ///
    /// let network = MLP::new(15, 10);
    /// ```
    pub fn new(input_size: usize, output_size: usize) -> Self {
        todo!();
    }

    /// Returns a new MLP following the given topology
    ///
    /// The topology is a slice where the first value is the network's input size, the following
    /// values are the number of neurons in each remaining layers, the last one being also the
    /// output size.
    ///
    /// # Panics
    /// Panics if there is less than two elements in the topology or if an element is below 1.
    ///
    /// # Examples
    /// ```
    /// use ocrust::mmlp::MLP;
    ///
    /// let network = MLP::from_topology(&[10, 4, 6, 15, 20]);
    /// ```
    pub fn from_topology(topology: &[usize]) -> Self {
        todo!();
    }

    /// Asks an MLP to classify a given input, returns a probability vector.
    ///
    /// Each number in the output vector represents the probability that the input data belongs the
    /// corresponding class.
    ///
    /// # Panics
    /// Panics if the input is the wrong size
    ///
    /// # Examples
    /// ```
    /// use ocrust::mmlp::MLP;
    ///
    /// let mut network = MLP::new(10, 5);
    /// let output: Vec<f64> = network.classify(&[0.; 10]);
    /// ```
    pub fn classify(&self, input: &[f64]) -> Vec<f64> {
        todo!();
    }

    /// Returns a new MLP that was saved with `save_to_file`.
    ///
    /// # Panics
    /// Panics if the file isn't a valid saved MLP
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn Error>> {
        todo!();
    }

    /// Save a plain text representation of the MLP to a file.
    ///
    /// # Panics
    /// Panics if the given path isn't accessible for writing.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn Error>> {
        todo!();
    }

    /// Randomizes the weights and the bias of every neurons of an MLP.
    pub fn randomize(&mut self) {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::izip;

    #[test]
    fn new_valid() {
        let input_size = 15;
        let output_size = 10;
        let network = MLP::new(input_size, output_size);

        // check number and shape of layers (should be one hidden and one output)
        assert_eq!(network.biases.len(), 2);
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.weights[0].ndim(), 2);
        assert_eq!(network.weights[1].ndim(), 2);

        // check if weights number is compatible with the input
        assert_eq!(network.weights[0].dim().0, input_size);

        // Hidden layer
        let hidden = &network.weights[0];
        let hidden_neurons = hidden.dim().1;
        assert_eq!(network.biases[0].dim(), hidden_neurons);
        assert!(
            input_size <= hidden_neurons && hidden_neurons <= output_size
                || input_size >= hidden_neurons && hidden_neurons >= output_size,
            "input - hidden - output: {} - {} - {}",
            input_size,
            hidden_neurons,
            output_size
        );

        // check if output's weights number is compatible with the hidden layer
        assert_eq!(network.weights[1].dim().0, hidden_neurons);

        // check if the output layer is compatible with the number of classes required
        assert_eq!(network.biases[1].dim(), output_size);
        assert_eq!(network.weights[1].dim().1, output_size);
    }

    #[test]
    #[should_panic]
    fn new_wrong_input() {
        MLP::new(0, 2);
    }

    #[test]
    #[should_panic]
    fn new_wrong_output() {
        MLP::new(2, 0);
    }

    #[test]
    fn from_topology_valid() {
        let topology = [10, 4, 6, 15, 20];
        let network = MLP::from_topology(&topology);

        // check number and shape of layers (should be one hidden and one output)
        assert_eq!(network.biases.len(), topology.len() - 1);
        assert_eq!(network.weights.len(), topology.len() - 1);
        for layer in &network.weights {
            assert_eq!(layer.ndim(), 2);
        }

        // check layer's compatibility with the previous one
        let mut input_size = topology[0];
        let layer_cases = izip!(&network.weights, network.biases, topology.iter().skip(1));
        for (weights, biases, wanted_size) in layer_cases {
            assert_eq!(weights.dim().0, input_size);
            assert_eq!(weights.dim().1, *wanted_size);
            assert_eq!(biases.dim(), *wanted_size);
            input_size = *wanted_size;
        }
    }

    #[test]
    #[should_panic]
    fn from_topology_too_short() {
        MLP::from_topology(&[2]);
    }

    #[test]
    #[should_panic]
    fn from_topology_empty_layer() {
        MLP::from_topology(&[5, 5, 0, 5, 5]);
    }

    #[test]
    fn classify_valid() {
        let input_size = 10;
        let output_size = 5;

        let network = MLP::new(input_size, output_size);
        let output = network.classify(&vec![0.; input_size]);

        assert_eq!(output.len(), output_size);
    }

    #[test]
    #[should_panic]
    fn classify_input_too_short() {
        let network = MLP::new(10, 5);
        network.classify(&[0.; 4]);
    }

    #[test]
    #[should_panic]
    fn classify_input_too_long() {
        let network = MLP::new(10, 5);
        network.classify(&[0.; 6]);
    }

    #[test]
    fn randomize_valid() {
        // just verify it doesn't panic
        let mut network = MLP::new(10, 5);
        network.randomize();
    }
}
