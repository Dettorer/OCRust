/// A neuron in an MLP, contains some weights and a bias
#[derive(Clone)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    /// Creates a new MLP neuron with `input_size` weights.
    ///
    /// # Panics
    /// Panics if `input_size == 0`
    ///
    /// # Examples
    /// ```
    /// use ocrust::mlp;
    ///
    /// let neuron = mlp::Neuron::new(15);
    /// ```
    pub fn new(input_size: usize) -> Neuron {
        if input_size == 0 {
            panic!("Tried to create a new neuron with 0 weights.");
        }

        Neuron {
            weights: vec![0.; input_size],
            bias: 0.,
        }
    }

    /// Returns the number of weights in a neuron.
    ///
    /// # Examples
    /// ```
    /// use ocrust::mlp::Neuron;
    ///
    /// let n = Neuron::new(30);
    /// assert_eq!(n.len(), 30);
    /// ```
    pub fn len(&self) -> usize {
        self.weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        let n = Neuron::new(15);
        assert_eq!(n.weights.len(), 15);
        assert_eq!(n.bias, 0.);

        let n = Neuron::new(1);
        assert_eq!(n.weights.len(), 1);
        assert_eq!(n.bias, 0.);
    }

    #[test]
    #[should_panic]
    fn new_panic_zero_weights() {
        let _n = Neuron::new(0);
    }

    #[test]
    fn len() {
        let neuron = Neuron {
            weights: vec![],
            bias: 0.,
        };
        assert_eq!(neuron.len(), 0);

        let neuron = Neuron {
            weights: vec![0.; 150],
            bias: 0.,
        };
        assert_eq!(neuron.len(), 150);
    }
}
