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
