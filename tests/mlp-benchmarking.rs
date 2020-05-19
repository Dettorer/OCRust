#![feature(test)]

#[macro_use]
extern crate ocrust;
extern crate test;

use nalgebra::DVector;
use test::Bencher;

#[bench]
fn bench_classify_mlp(bencher: &mut Bencher) {
    let network = mlp![20, 15, 15, 10];
    let input = DVector::repeat(20, 10.);
    bencher.iter(|| network.classify(&input));
}

#[bench]
fn bench_randomize_mlp(bencher: &mut Bencher) {
    bencher.iter(|| randomized_mlp![20, 15, 15, 13, 17, 10]);
}

#[bench]
fn bench_learning_mlp(bencher: &mut Bencher) {
    let mut network = mlp![2;2];
    bencher.iter(|| {
        let input = DVector::from_row_slice(&[0_f64, 1_f64]);
        network.train_case(input, 1)
    });
}
