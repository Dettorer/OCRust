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

    /// Returns a new MLP following the given topology
    ///
    /// The topology is a slice where the first value is the network's input size, the following
    /// values are the sizes of remaining layers, the last one being also the output size.
    ///
    /// # Panics
    /// Panics if there is less than two elements in the topology or if an element is below 1.
    ///
    /// # Examples
    /// ```
    /// use ocrust::mlp::MLP;
    ///
    /// let network = MLP::from_topology(&[10, 4, 6, 15, 20]);
    /// ```
    pub fn from_topology(topology: &[usize]) -> Self {
        assert!(
            topology.len() >= 2,
            "Trying to create an MLP with an invalid topology"
        );

        let input_size = topology[0];
        let mut layers = vec![];
        let mut previous_size = input_size;
        for size in topology.iter().skip(1) {
            layers.push(Layer::new(*size, previous_size));
            previous_size = *size;
        }

        MLP {
            input_size: topology[0],
            layers: layers,
        }
    }

    /// Ask an MLP to classify a given input, updating the activations of it's output layer
    ///
    /// # Panics
    /// Panics if the input is the wrong size
    ///
    /// # Examples
    /// TODO
    pub fn classify(&mut self, input: &[f64]) {
        assert_eq!(
            input.len(),
            self.input_size,
            "Trying to classify a wrong-sized input"
        );

        self.layers[0].activate(input);
        let mut last_output = self.layers[0].output();
        for layer in &mut self.layers.iter_mut().skip(1) {
            layer.activate(&last_output);
            last_output = layer.output();
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
        assert_eq!(network.input_size, 10);

        assert_eq!(network.layers.len(), topology.len() - 1);
        let mut last_size = topology[0];
        let layer_cases = network.layers.iter().zip(topology.iter().skip(1));
        for (layer, wanted_size) in layer_cases {
            assert_eq!(layer.neurons.len(), *wanted_size);
            for neuron in &layer.neurons {
                assert_eq!(neuron.weights.len(), last_size);
            }
            last_size = *wanted_size;
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
        let mut network = MLP::new(10, 5);
        network.classify(&[0.; 10]);
        for neuron in &network.layers[1].neurons {
            assert!(neuron.activation.is_some());
        }
    }

    #[test]
    #[should_panic]
    fn classify_input_too_short() {
        let mut network = MLP::new(10, 5);
        network.classify(&[0.; 4]);
    }

    #[test]
    #[should_panic]
    fn classify_input_too_long() {
        let mut network = MLP::new(10, 5);
        network.classify(&[0.; 6]);
    }
}
