use neurons::network;
use neurons::activation::Activation::{ReLU, Softmax};

fn main() {
    let mut network = network::Network::new();

    network.add_layer(1, 4, ReLU, false);
    network.add_layer(4, 3, ReLU, true);
    network.add_layer(3, 1, Softmax, false);

    println!("{}", network);
}