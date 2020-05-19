//! This is a proof-of-concept tool for our neural network, the goal is to create, train and use a
//! neural network that can output the result of a xor operation given two input bits.

use ocrust::{mlp, randomized_mlp};

const TRAINING_DATA: [((f64, f64), usize); 4] =
    [((0., 0.), 0), ((0., 1.), 1), ((1., 0.), 1), ((0., 0.), 0)];

fn best_class(output: mlp::Output) -> Option<usize> {
    let mut best = None;
    let mut best_activation = f64::NEG_INFINITY;
    for (class, &activation) in output.iter().enumerate() {
        if activation > best_activation {
            best = Some(class);
            best_activation = activation;
        }
    }

    best
}

fn display_network_behavior(network: &mlp::MLP) {
    for ((a, b), expected) in TRAINING_DATA.iter() {
        let input = mlp::Input::from_row_slice(&[*a, *b]);
        let res = best_class(network.classify(&input));
        println!(
            "{} xor {}: got {}, expected {}",
            a,
            b,
            res.unwrap(),
            expected
        );
    }
}

fn main() {
    let mut network = randomized_mlp![2; 2];
    network.display_weights();
    display_network_behavior(&network);

    print!("\nTraining over the whole dataset 100 times... ");
    for _ in 0..100 {
        for ((a, b), expected) in TRAINING_DATA.iter() {
            let input = mlp::Input::from_row_slice(&[*a, *b]);
            network.train_case(&input, *expected);
        }
    }
    println!("ok");
    network.display_weights();
    display_network_behavior(&network);
}
