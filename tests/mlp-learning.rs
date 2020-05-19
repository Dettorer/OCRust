extern crate ocrust;
use ocrust::mlp;

const TRAINING_DATA: [((f64, f64), usize); 4] =
    [((0., 0.), 0), ((0., 1.), 1), ((1., 0.), 1), ((0., 0.), 0)];

#[test]
fn mlp_learn_xor_case_by_case() {
    let mut network = mlp![2; 2];

    // Train the network over the whole dataset 100 times
    for _ in 0..100 {
        for ((a, b), expected) in TRAINING_DATA.iter() {
            network.train_case(mlp::Input::from_row_slice(&[*a, *b]), *expected);
        }
    }

    for ((in_a, in_b), expected) in TRAINING_DATA.iter() {
        let input = mlp::Input::from_row_slice(&[*in_a, *in_b]);
        let output = network.classify(&input);

        // get the most probable class (index of the most activated neuron)
        let mut best = 0;
        let mut best_activation = f64::NEG_INFINITY;
        for (class, &activation) in output.iter().enumerate() {
            if activation > best_activation {
                best = class;
                best_activation = activation;
            }
        }

        assert_eq!(
            best, *expected,
            "For input ({}, {}), the network answered {} instead of the correct answer ({})",
            in_a, in_b, best, expected
        );
    }
}
