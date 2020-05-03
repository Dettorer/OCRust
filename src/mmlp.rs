//! A modular implementation for Multilayer Perceptrons with efficient matrix representations and
//! manipulations.
//!
//! This crate provides the structure and functions for common manipulations of [Multilayer
//! Perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) which are a kind of
//! artificial neural network.

use ndarray::{arr1, Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// The main structure representing an MLP.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct MLP {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

/// A common trait for the ability to generate one and two dimensions ndarrays of `f64`.
///
/// This is used in this module to help having a common interface for generating either zeroed
/// arrays or randomized arrays.
trait ArrayBuilder {
    fn array1(dim: usize, rnd: Uniform<f64>) -> Array1<f64>;
    fn array2(dim1: usize, dim2: usize, rnd: Uniform<f64>) -> Array2<f64>;
}

/// An `ArrayBuilder` that generates arrays with elements initialized to 0.
enum ZeroBuilder {}
impl ArrayBuilder for ZeroBuilder {
    fn array1(dim: usize, _rnd: Uniform<f64>) -> Array1<f64> {
        Array1::zeros(dim)
    }
    fn array2(dim1: usize, dim2: usize, _rnd: Uniform<f64>) -> Array2<f64> {
        Array2::zeros((dim1, dim2))
    }
}

/// An `ArrayBuilder` that generates arrays with randomized elements.
enum RandomBuilder {}
impl ArrayBuilder for RandomBuilder {
    fn array1(dim: usize, rnd: Uniform<f64>) -> Array1<f64> {
        Array1::random(dim, rnd)
    }
    fn array2(dim1: usize, dim2: usize, rnd: Uniform<f64>) -> Array2<f64> {
        Array2::random((dim1, dim2), rnd)
    }
}

impl MLP {
    /// Returns a new MLP following the given topology, with weights and biases generated with the
    /// given `ArrayBuilder`.
    ///
    /// The topology is a slice where the first value is the network's input size, the following
    /// values are the number of neurons in each remaining layers, the last one being also the
    /// output size.
    ///
    /// Each weight matrix and bias array is an ndarray generated using the given `ArrayBuilder`.
    ///
    /// # Panics
    /// Panics if there is less than two elements in the topology or if an element is below 1.
    fn from_topology<Builder: ArrayBuilder>(topology: &[usize]) -> Self {
        assert!(
            topology.len() >= 2,
            "Trying to create an MLP with less than two layers"
        );
        topology
            .iter()
            .for_each(|&size| assert!(size > 0, "Trying to create an MLP with an empty layer"));

        let rnd = ndarray_rand::rand_distr::Uniform::new(-1_f64, 1_f64);

        MLP {
            weights: topology
                .windows(2)
                .map(|win| Builder::array2(win[0], win[1], rnd))
                .collect(),
            biases: topology
                .iter()
                .skip(1)
                .map(|&size| Builder::array1(size, rnd))
                .collect(),
        }
    }

    /// Returns a new MLP following the given topology, with weights initialized with 0.
    ///
    /// The topology is a slice where the first value is the network's input size, the following
    /// values are the number of neurons in each remaining layers, the last one being also the
    /// output size.
    ///
    /// # Panics
    /// Panics if there is less than two elements in the topology or if an element is below 1.
    pub fn from_topology_zeros(topology: &[usize]) -> Self {
        MLP::from_topology::<ZeroBuilder>(topology)
    }

    /// Returns a new MLP following the given topology, with weights initialized with a random
    /// value between -1 and 1.
    ///
    /// The topology is a slice where the first value is the network's input size, the following
    /// values are the number of neurons in each remaining layers, the last one being also the
    /// output size.
    ///
    /// # Panics
    /// Panics if there is less than two elements in the topology or if an element is below 1.
    pub fn from_topology_randomized(topology: &[usize]) -> Self {
        MLP::from_topology::<RandomBuilder>(topology)
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
    /// let mut network = ocrust::mlp![10; 5];
    /// let output: Vec<f64> = network.classify(&[0.; 10]);
    /// ```
    pub fn classify(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(&self.biases)
            .fold(arr1(input), |input, (weights, biases)| {
                (input.dot(weights) + biases).map(sigmoid)
            })
            .to_vec()
    }

    /// Returns a new MLP that was saved with `save_to_file`.
    ///
    /// # Panics
    /// Panics if the file isn't a valid saved MLP
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mlp = serde_json::from_reader(reader)?;

        Ok(mlp)
    }

    /// Save a plain text representation of the MLP to a file.
    ///
    /// # Panics
    /// Panics if the given path isn't accessible for writing.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, self)?;

        Ok(())
    }
}

/// Creates a zero-initialized [`MLP`] with the given topology.
///
/// There are two forms of this macro:
///
/// - Create an [`MLP`] with a detailed topology:
///
/// ```
/// use ocrust::mlp;
///
/// let mut network = mlp![50, 30, 45, 40, 26];
/// let output = network.classify(&[0.5; 50]);
/// assert_eq!(26, output.len());
/// ```
///
/// - Create a [`MLP`] with an input and output size:
///
/// ```
/// use ocrust::mlp;
///
/// let input_size = 15;
/// let output_size = 10;
/// let mut network = mlp![input_size; output_size];
/// let output = network.classify(&[0.5; 15]);
/// assert_eq!(output_size, output.len());
/// ```
///
/// This last variant will create an MLP with exactly one hidden layer which size is between the
/// input's and the output's.
///
/// [`MLP`]: ./mmlp/struct.MLP.html
#[macro_export]
macro_rules! mlp {
    ($input:expr ; $output:expr) => {
        $crate::mmlp::MLP::from_topology_zeros(&[$input, ($input + $output) / 2, $output])
    };
    ($($layers:expr),+) => {
        $crate::mmlp::MLP::from_topology_zeros(&[$($layers),*])
    };
}

/// Creates a [`MLP`] with the given topology and randomized weights and biases.
///
/// There are two forms of this macro:
///
/// - Create an [`MLP`] with a detailed topology:
///
/// ```
/// use ocrust::randomized_mlp;
///
/// let mut network = randomized_mlp![50, 30, 45, 40, 26];
/// let output = network.classify(&[0.5; 50]);
/// assert_eq!(26, output.len());
/// ```
///
/// - Create a [`MLP`] with an input and output size:
///
/// ```
/// use ocrust::randomized_mlp;
///
/// let input_size = 15;
/// let output_size = 10;
/// let mut network = randomized_mlp![input_size; output_size];
/// let output = network.classify(&[0.5; 15]);
/// assert_eq!(output_size, output.len());
/// ```
///
/// This last variant will create an MLP with exactly one hidden layer which size is between the
/// input's and the output's.
///
/// [`MLP`]: ./mmlp/struct.MLP.html
#[macro_export]
macro_rules! randomized_mlp {
    ($input:expr ; $output:expr) => {
        $crate::mmlp::MLP::from_topology_randomized(&[$input, ($input + $output) / 2, $output])
    };
    ($($layers:expr),+) => {
        $crate::mmlp::MLP::from_topology_randomized(&[$($layers),*])
    };
}

fn sigmoid(x: &f64) -> f64 {
    let e = std::f64::consts::E;
    1.0_f64 / (1_f64 + e.powf(-x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::izip;

    fn valid_with_topology(network: &MLP, topology: &[usize]) {
        // check number and shape of layers (should be one hidden and one output)
        assert_eq!(network.biases.len(), topology.len() - 1);
        assert_eq!(network.weights.len(), topology.len() - 1);
        for layer in &network.weights {
            assert_eq!(layer.ndim(), 2);
        }

        // check each layer's correctness and compatibility with the previous one
        let mut input_size = topology[0];
        let layer_cases = izip!(&network.weights, &network.biases, topology.iter().skip(1));
        for (weights, biases, wanted_size) in layer_cases {
            assert_eq!(weights.dim().0, input_size);
            assert_eq!(weights.dim().1, *wanted_size);
            assert_eq!(biases.dim(), *wanted_size);
            input_size = *wanted_size;
        }
    }

    #[test]
    fn macro_mlp_short_valid() {
        let input_size = 15;
        let output_size = 10;
        let network = mlp![input_size; output_size];
        valid_with_topology(&network, &[15, 12, 10]);
    }

    #[test]
    fn macro_mlp_long_valid() {
        let network = mlp![10, 4, 6, 15, 20];
        valid_with_topology(&network, &[10, 4, 6, 15, 20]);
    }

    #[test]
    fn macro_randomized_short_valid() {
        let input_size = 15;
        let output_size = 10;
        let network = randomized_mlp![input_size; output_size];
        valid_with_topology(&network, &[15, 12, 10]);
    }

    #[test]
    fn macro_randomized_long_valid() {
        let network = randomized_mlp![10, 4, 6, 15, 20];
        valid_with_topology(&network, &[10, 4, 6, 15, 20]);
    }

    #[test]
    #[should_panic]
    fn macro_mlp_short_wrong_input() {
        mlp![0; 2];
    }

    #[test]
    #[should_panic]
    fn macro_mlp_short_wrong_output() {
        mlp![2; 0];
    }

    #[test]
    fn from_topology_valid() {
        let topology = [10, 4, 6, 15, 20];
        let network = MLP::from_topology::<ZeroBuilder>(&topology);
        valid_with_topology(&network, &topology);
    }

    #[test]
    #[should_panic]
    fn from_topology_too_short() {
        MLP::from_topology::<ZeroBuilder>(&[2]);
    }

    #[test]
    #[should_panic]
    fn from_topology_empty_layer() {
        MLP::from_topology::<ZeroBuilder>(&[5, 5, 0, 5, 5]);
    }

    #[test]
    fn classify_valid() {
        let input_size = 10;
        let output_size = 5;

        let network = mlp![input_size; output_size];
        let output = network.classify(&vec![0.; input_size]);

        assert_eq!(output.len(), output_size);
    }

    #[test]
    #[should_panic]
    fn classify_input_too_short() {
        let network = mlp![10; 5];
        network.classify(&[0.; 4]);
    }

    #[test]
    #[should_panic]
    fn classify_input_too_long() {
        let network = mlp![10; 5];
        network.classify(&[0.; 6]);
    }
}
