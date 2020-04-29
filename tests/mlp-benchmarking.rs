#![feature(test)]

extern crate test;

use ocrust::mlp::MLP;
use test::Bencher;

#[bench]
fn bench_classify(bencher: &mut Bencher) {
    let mut network = MLP::from_topology(&[20, 15, 15, 10]);
    bencher.iter(|| network.classify(&[10.; 20]));
}
