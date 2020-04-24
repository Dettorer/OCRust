use super::Neuron;

/// A layer in an MLP, contains [`Neuron`](struct.Neuron.html)s
pub struct Layer {
    pub neurons: Vec<Neuron>,
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

    /// Updates every neurons' activation of a layer given an other layer as input
    ///
    /// # Panics
    /// Neuron activation will panic if the input layer has inactivated neurons.
    ///
    /// # Examples
    /// ```
    /// use ocrust::mlp::Layer;
    ///
    /// let mut prev_layer = Layer::new(15, 1);
    /// for neuron in prev_layer.neurons.iter_mut() {
    ///     neuron.activation = Some(1.);
    /// }
    ///
    /// let mut layer = Layer::new(10, 15);
    /// layer.activate(&prev_layer);
    /// for neuron in layer.neurons {
    ///     assert!(neuron.activation.is_some(), "{:?}", neuron.activation);
    /// }
    /// ```
    pub fn activate(&mut self, input: &Layer) {
        for neuron in &mut self.neurons {
            neuron.activate(input);
        }
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
    fn len_valid() {
        let layer = Layer {
            neurons: vec![Neuron::new(1); 15],
        };
        assert_eq!(layer.len(), 15);

        let layer = Layer {
            neurons: vec![Neuron::new(1); 2],
        };
        assert_eq!(layer.len(), 2);
    }

    #[test]
    fn activate_valid() {
        let mut prev_layer = Layer::new(15, 1);
        for neuron in prev_layer.neurons.iter_mut() {
            neuron.activation = Some(1.);
        }

        let mut layer = Layer {
            neurons: vec![Neuron::new(15); 10],
        };
        layer.activate(&prev_layer);
        for neuron in layer.neurons {
            assert!(neuron.activation.is_some(), "{:?}", neuron.activation);
        }
    }
}
