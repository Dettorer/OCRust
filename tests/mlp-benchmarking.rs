#![feature(test)]

#[macro_use]
extern crate ocrust;
extern crate test;

use nalgebra::DVector;
use test::Bencher;

#[bench]
fn bench_mlp_classify(bencher: &mut Bencher) {
    let network = mlp![20, 15, 15, 10];
    let input = DVector::repeat(20, 10.);
    bencher.iter(|| network.classify(&input));
}

#[bench]
fn bench_mlp_randomize(bencher: &mut Bencher) {
    bencher.iter(|| randomized_mlp![20, 15, 15, 13, 17, 10]);
}

#[bench]
fn bench_mlp_learning(bencher: &mut Bencher) {
    let mut network = mlp![20, 15, 15, 10];
    let input = DVector::repeat(20, 10.);
    bencher.iter(|| network.train_case(&input, 5));
}
