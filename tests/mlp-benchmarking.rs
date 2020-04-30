#![feature(test)]

extern crate test;

use ocrust::mlp;
use ocrust::mmlp;
use test::Bencher;

#[bench]
fn bench_classify_mlp(bencher: &mut Bencher) {
    let mut network = mlp::MLP::from_topology(&[20, 15, 15, 10]);
    bencher.iter(|| network.classify(&[10.; 20]));
}

#[bench]
fn bench_classify_mmlp(bencher: &mut Bencher) {
    let network = mmlp::MLP::from_topology(&[20, 15, 15, 10]);
    bencher.iter(|| network.classify(&[10.; 20]));
}
