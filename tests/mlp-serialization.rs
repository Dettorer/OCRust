extern crate ocrust;
use ocrust::mlp;
use std::env;

#[test]
fn mlp_serialization() {
    // Find a temporary file
    let mut tmp_path = env::temp_dir();
    tmp_path.push(format!(
        "ocrust-test-serialization-{:x}.mlp",
        rand::random::<u64>()
    ));
    assert!(
        !tmp_path.exists(),
        "Could not prepare serialization test, {} file exists.",
        tmp_path.display()
    );

    // Prepare the network to be saved
    let network = mlp![10; 15];

    // Save to file
    network.save_to_file(&tmp_path).unwrap();
    assert!(
        tmp_path.exists(),
        "MLP::save_to_file did not create the {} file.",
        tmp_path.display()
    );

    // Recover from file
    let recovered = mlp::MLP::from_file(&tmp_path).unwrap();
    assert_eq!(network, recovered);

    // Delete temporary file
    std::fs::remove_file(&tmp_path).expect("Could not clean up after serialization test");
}
