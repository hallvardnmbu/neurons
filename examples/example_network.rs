use neurons::tensor;
use neurons::feedforward;
use neurons::activation::Activation::{ReLU, Softmax};

fn main() {
    let mut network = feedforward::Feedforward::new(tensor::Shape::Vector(2));

    network.dense(100, ReLU, false, None);
    network.convolution(5, (5, 5), (1, 1), (1, 1), ReLU, false, Some(0.1));
    network.dense(1, Softmax, false, None);

    println!("{}", network);
}