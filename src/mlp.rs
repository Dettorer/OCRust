//! A modular implementation for Multilayer Perceptrons.
//!
//! This crate provides the structure and functions for common manipulations of [Multilayer
//! Perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) which are a kind of
//! artificial neural network.

mod layer;
mod neuron;

use layer::Layer;
use neuron::Neuron;

/// The main structure representing an MLP, contains [`Layer`](struct.Layer.html)s.
pub struct MLP {
    input_size: usize,
    layers: Vec<Layer>,
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
    /// use ocrust::mlp::MLP;
    ///
    /// let network = MLP::new(15, 10);
    /// ```
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut layers = vec![];
        let hidden_size = (input_size + output_size) / 2;
        layers.push(Layer::new(hidden_size, input_size));
        layers.push(Layer::new(output_size, hidden_size));

        MLP {
            input_size: input_size,
            layers: layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        let input_size = 15;
        let output_size = 10;
        let network = MLP::new(input_size, output_size);
        assert_eq!(network.input_size, input_size);

        assert_eq!(network.layers.len(), 2);
        // Hidden layer
        let hidden = &network.layers[0];
        let hidden_size = hidden.neurons.len();
        assert!(
            input_size <= hidden_size && hidden_size <= output_size
                || input_size >= hidden_size && hidden_size >= output_size,
            "input - hidden - output: {} - {} - {}",
            input_size,
            hidden_size,
            output_size
        );
        for neuron in &hidden.neurons {
            assert_eq!(neuron.weights.len(), input_size);
            assert!(neuron.activation.is_none());
        }

        // Output layer
        assert_eq!(network.layers[1].neurons.len(), output_size);
    }
}
