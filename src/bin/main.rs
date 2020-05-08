use nalgebra::{DMatrix, DVector};

fn main() {
    let v = DVector::from_row_slice(&[0, 1, 2, 3, 4]);
    let m = DMatrix::from_row_slice(2, 5, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    println!("{} * {} = {}", v, m, &m * &v);
}
