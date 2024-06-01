use neurons::network;
use neurons::activation::Activation::{ReLU, Softmax};
use neurons::random;

fn main() {
    let mut network = network::Network::new();

    network.add_layer(1, 4, ReLU, false);
    network.add_layer(4, 3, ReLU, true);
    network.add_layer(3, 1, Softmax, false);

    println!("{}", network);

    let mut rand = random::Generator::create(1);
    for _ in 0..10 {
        println!("{}", rand.next(0.0, 1.0));
    }
}