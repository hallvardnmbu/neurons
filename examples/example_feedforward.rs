use neurons::feedforward;
use neurons::activation::Activation::{ReLU, Softmax};

fn main() {
    let mut network = feedforward::Feedforward::new();

    network.add_layer(1, 4, ReLU, false, None);
    network.add_layer(4, 3, ReLU, true, None);
    network.add_layer(3, 1, Softmax, false, None);

    println!("{}", network);
}