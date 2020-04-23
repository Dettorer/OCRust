//! A modular implementation for Multilayer Perceptrons.
//!
//! This crate provides the structure and functions for common manipulations of [Multilayer
//! Perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) which are a kind of
//! artificial neural network.

mod layer;
mod neuron;

pub use layer::Layer;
pub use neuron::Neuron;

/// The main structure representing an MLP, contains [`Layer`](struct.Layer.html)s.
pub struct MLP {
    layers: Vec<Layer>,
}
