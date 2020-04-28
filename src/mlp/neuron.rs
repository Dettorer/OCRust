use serde::{Deserialize, Serialize};

/// A neuron in an MLP, contains some weights and a bias
#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Debug)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub activation: Option<f64>,
}

impl Neuron {
    /// Creates a new MLP neuron with `input_size` weights.
    ///
    /// # Panics
    /// Panics if `input_size == 0`
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
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Updates a neuron's activation given a slice of floats as input.
    ///
    /// The slice typically comes from the activations of the neurons of a previous layer, or the
    /// MLP's own input.
    ///
    /// The computed activation is always between 0 and 1 inclusive.
    ///
    /// # Panics
    /// Panics if the input slice isn't the same size as the neuron's weights vector.
    pub fn activate(&mut self, input: &[f64]) {
        if input.len() != self.weights.len() {
            panic!("Trying to activate a neuron with wrong sized input");
        }

        self.activation = Some(sigmoid(
            input
                .iter()
                .zip(&self.weights) // combine each input activation with its weight
                .map(|(signal, weight)| signal * weight)
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
        let input = [1_f64; 15];
        let mut neuron = Neuron {
            weights: vec![0.; 15], // weight of 0 for each prev_neuron
            bias: 0.,
            activation: None,
        };
        neuron.activate(&input);
        assert!(neuron.activation.is_some());
        assert!(
            0. <= neuron.activation.unwrap() && neuron.activation.unwrap() <= 1.,
            "neuron.activation = {}",
            neuron.activation.unwrap()
        );

        let input = [1_f64; 15];
        let mut neuron = Neuron {
            weights: vec![1.; 15], // max weight for each prev_neuron
            bias: 0.,
            activation: None,
        };
        neuron.activate(&input);
        assert!(neuron.activation.is_some());
        assert!(
            0. <= neuron.activation.unwrap() && neuron.activation.unwrap() <= 1.,
            "neuron.activation = {}",
            neuron.activation.unwrap()
        );

        // make a layer with two differently-activated kind of neurons
        let mut input = vec![0.2_f64; 10];
        input.extend(vec![0.6_f64; 10]);
        let mut neuron = Neuron {
            weights: vec![0.3; 20],
            bias: 1.,
            activation: None,
        };
        neuron.activate(&input);
        assert!(neuron.activation.is_some());
        assert!(
            0. <= neuron.activation.unwrap() && neuron.activation.unwrap() <= 1.,
            "neuron.activation = {}",
            neuron.activation.unwrap()
        );

        let mut input = vec![0_f64; 10];
        input.extend(vec![1_f64; 10]);
        input.extend(vec![0.5_f64; 10]);
        let mut neuron = Neuron {
            weights: vec![0.3; 30], // max weight for each prev_neuron
            bias: 0.3,
            activation: None,
        };
        neuron.activate(&input);
        assert!(neuron.activation.is_some());
        assert!(
            0. <= neuron.activation.unwrap() && neuron.activation.unwrap() <= 1.,
            "neuron.activation = {}",
            neuron.activation.unwrap()
        );
    }

    #[test]
    #[should_panic]
    fn activate_input_too_short() {
        let input = [1_f64; 14];
        let mut neuron = Neuron {
            weights: vec![0.; 15], // weight of 0 for each prev_neuron
            bias: 0.,
            activation: None,
        };
        neuron.activate(&input);
    }

    #[test]
    #[should_panic]
    fn activate_input_too_long() {
        let input = [1_f64; 16];
        let mut neuron = Neuron {
            weights: vec![0.; 15], // weight of 0 for each prev_neuron
            bias: 0.,
            activation: None,
        };
        neuron.activate(&input);
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
