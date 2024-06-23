use neurons::feedforward;
use neurons::activation::Activation::{ReLU, Softmax};

fn main() {
    let mut network = feedforward::Feedforward::new(None);

    network.add_dense(1, 4, ReLU, false, None);
    network.add_dense(4, 3, ReLU, true, None);
    network.add_dense(3, 1, Softmax, false, None);

    println!("{}", network);
}