use super::Neuron;

/// A layer in an MLP, contains [`Neuron`](struct.Neuron.html)s
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates a new MLP layer with `size` neurons of `input_size` weights.
    ///
    /// # Panics
    /// Panics if `size == 0`. Neuron initialization will also panic if `input_size == 0`.
    ///
    /// # Examples
    /// ```
    /// use ocrust::mlp;
    ///
    /// let layer = mlp::Layer::new(30, 15);
    /// ```
    pub fn new(size: usize, input_size: usize) -> Layer {
        if size == 0 {
            panic!("Tried to create a new layer with 0 neurons.");
        }

        Layer {
            neurons: vec![Neuron::new(input_size); size],
        }
    }

    /// Returns the number of neurons in a layer.
    ///
    /// # Examples
    /// ```
    /// use ocrust::mlp::Layer;
    ///
    /// let l = Layer::new(3, 10);
    /// assert_eq!(l.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.neurons.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        let layer = Layer::new(1, 10);
        assert_eq!(layer.neurons.len(), 1);
        for neuron in layer.neurons {
            assert_eq!(neuron.len(), 10);
        }

        let layer = Layer::new(100, 2);
        assert_eq!(layer.neurons.len(), 100);
        for neuron in layer.neurons {
            assert_eq!(neuron.len(), 2);
        }
    }

    #[test]
    #[should_panic]
    fn new_panic_zero_neurons() {
        let _l = Layer::new(0, 12);
    }

    #[test]
    fn len() {
        let layer = Layer { neurons: vec![] };
        assert_eq!(layer.len(), 0);
    }
}
