use super::Layer;

/// A neuron in an MLP, contains some weights and a bias
#[derive(Clone, Default)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    pub activation: Option<f64>,
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
            activation: None,
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

    /// Updates a neuron's activation given an input layer
    ///
    /// The computed activation is always between 0 and 1 inclusive.
    ///
    /// # Panics
    /// Panics if the input layer isn't the same size as the neuron's weights vector, or if a
    /// neuron in the input layer isn't activated.
    ///
    /// # Examples
    /// ```
    /// use ocrust::mlp::{Neuron, Layer};
    ///
    /// let mut prev_layer = Layer::new(15, 1);
    /// for neuron in prev_layer.neurons.iter_mut() {
    ///     neuron.activation = Some(1.);
    /// }
    ///
    /// let mut neuron = Neuron::new(15);
    /// neuron.activate(&prev_layer);
    /// assert!(0. <= neuron.activation.unwrap() && neuron.activation.unwrap() <= 1.);
    /// ```
    pub fn activate(&mut self, input: &Layer) {
        if input.len() != self.weights.len() {
            panic!("Trying to activate a neuron with wrong sized input");
        }

        self.activation = Some(sigmoid(
            input
                .neurons
                .iter()
                .map(|neuron| {
                    // extract each input neuron's activation
                    neuron
                        .activation
                        .expect("Trying to activate a neuron with a not fully activated input")
                })
                .zip(&self.weights) // combine each input activation with its weight
                .map(|(activation, weight)| activation * weight)
                .sum::<f64>()
                + self.bias,
        ));
    }
}

fn sigmoid(x: f64) -> f64 {
    let e = std::f64::consts::E;
    1.0_f64 / (1_f64 + e.powf(-x))
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
    fn len_valid() {
        let neuron = Neuron {
            weights: vec![],
            bias: 0.,
            activation: None,
        };
        assert_eq!(neuron.len(), 0);

        let neuron = Neuron {
            weights: vec![0.; 150],
            bias: 0.,
            activation: None,
        };
        assert_eq!(neuron.len(), 150);
    }

    #[test]
    fn activate_valid() {
        let prev_neuron = Neuron {
            activation: Some(1.),
            ..Default::default()
        };
        let prev_layer = Layer {
            neurons: vec![prev_neuron; 15],
        };
        let mut neuron = Neuron {
            weights: vec![0.; 15], // weight of 0 for each prev_neuron
            bias: 0.,
            activation: None,
        };
        neuron.activate(&prev_layer);
        assert!(neuron.activation.is_some());
        assert!(
            0. <= neuron.activation.unwrap() && neuron.activation.unwrap() <= 1.,
            "neuron.activation = {}",
            neuron.activation.unwrap()
        );

        let prev_neuron = Neuron {
            activation: Some(1.),
            ..Default::default()
        };
        let prev_layer = Layer {
            neurons: vec![prev_neuron; 15],
        };
        let mut neuron = Neuron {
            weights: vec![1.; 15], // max weight for each prev_neuron
            bias: 0.,
            activation: None,
        };
        neuron.activate(&prev_layer);
        assert!(neuron.activation.is_some());
        assert!(
            0. <= neuron.activation.unwrap() && neuron.activation.unwrap() <= 1.,
            "neuron.activation = {}",
            neuron.activation.unwrap()
        );

        // make a layer with two differently-activated kind of neurons
        let mut prev_neurons = vec![
            Neuron {
                activation: Some(0.2),
                ..Default::default()
            };
            10
        ];
        prev_neurons.extend(vec![
            Neuron {
                activation: Some(0.6),
                ..Default::default()
            };
            10
        ]);
        let prev_layer = Layer {
            neurons: prev_neurons,
        };
        let mut neuron = Neuron {
            weights: vec![0.3; 20],
            bias: 1.,
            activation: None,
        };
        neuron.activate(&prev_layer);
        assert!(neuron.activation.is_some());
        assert!(
            0. <= neuron.activation.unwrap() && neuron.activation.unwrap() <= 1.,
            "neuron.activation = {}",
            neuron.activation.unwrap()
        );

        // make a layer with three differently-activated kind of neurons
        let mut prev_neurons = vec![
            Neuron {
                activation: Some(0.),
                ..Default::default()
            };
            10
        ];
        prev_neurons.extend(vec![
            Neuron {
                activation: Some(1.),
                ..Default::default()
            };
            10
        ]);
        prev_neurons.extend(vec![
            Neuron {
                activation: Some(0.5),
                ..Default::default()
            };
            10
        ]);
        let prev_layer = Layer {
            neurons: prev_neurons,
        };
        let mut neuron = Neuron {
            weights: vec![0.3; 30], // max weight for each prev_neuron
            bias: 0.3,
            activation: None,
        };
        neuron.activate(&prev_layer);
        assert!(neuron.activation.is_some());
        assert!(
            0. <= neuron.activation.unwrap() && neuron.activation.unwrap() <= 1.,
            "neuron.activation = {}",
            neuron.activation.unwrap()
        );
    }

    #[test]
    #[should_panic]
    fn activate_previous_layer_not_activated() {
        let prev_neuron = Neuron {
            activation: None,
            ..Default::default()
        };
        let prev_layer = Layer {
            neurons: vec![prev_neuron; 10],
        };
        let mut neuron: Neuron = Default::default();
        neuron.activate(&prev_layer);
    }

    #[test]
    #[should_panic]
    fn activate_previous_layer_too_short() {
        let prev_neuron = Neuron {
            activation: Some(1.),
            ..Default::default()
        };
        let prev_layer = Layer {
            neurons: vec![prev_neuron; 14],
        };
        let mut neuron = Neuron {
            weights: vec![0.; 15], // weight of 0 for each prev_neuron
            bias: 0.,
            activation: None,
        };
        neuron.activate(&prev_layer);
    }

    #[test]
    #[should_panic]
    fn activate_previous_layer_too_long() {
        let prev_neuron = Neuron {
            activation: Some(1.),
            ..Default::default()
        };
        let prev_layer = Layer {
            neurons: vec![prev_neuron; 16],
        };
        let mut neuron = Neuron {
            weights: vec![0.; 15], // weight of 0 for each prev_neuron
            bias: 0.,
            activation: None,
        };
        neuron.activate(&prev_layer);
    }

    #[test]
    fn sigmoid_valid() {
        let res = sigmoid(0.);
        assert!(0. <= res && res <= 1., "res = {}", res);

        let res = sigmoid(0.3);
        assert!(0. <= res && res <= 1., "res = {}", res);

        let res = sigmoid(-0.3);
        assert!(0. <= res && res <= 1., "res = {}", res);

        let res = sigmoid(1.);
        assert!(0. <= res && res <= 1., "res = {}", res);

        let res = sigmoid(-1.);
        assert!(0. <= res && res <= 1., "res = {}", res);

        let res = sigmoid(-42.);
        assert!(0. <= res && res <= 1., "res = {}", res);

        let res = sigmoid(42.);
        assert!(0. <= res && res <= 1., "res = {}", res);
    }
}
