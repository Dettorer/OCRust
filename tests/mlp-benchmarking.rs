#![feature(test)]

extern crate ocrust;
extern crate test;

use nalgebra::DVector;
use ocrust::mlp;
use test::Bencher;

#[bench]
fn bench_classify_mlp(bencher: &mut Bencher) {
    let mut network = mlp::MLP::from_topology(&[20, 15, 15, 10]);
    bencher.iter(|| network.classify(&[10.; 20]));
}

#[bench]
fn bench_classify_mmlp(bencher: &mut Bencher) {
    let network = ocrust::mlp![20, 15, 15, 10];
    let input = DVector::repeat(20, 10.);
    bencher.iter(|| network.classify(&input));
}

#[bench]
fn bench_randomize_mmlp(bencher: &mut Bencher) {
    bencher.iter(|| ocrust::randomized_mlp![20, 15, 15, 13, 17, 10]);
}
