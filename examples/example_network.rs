use neurons::tensor;
use neurons::feedforward;
use neurons::activation::Activation::{ReLU, Softmax};

fn main() {
    let mut network = feedforward::Feedforward::new(tensor::Shape::Dense(2));

    network.add_dense(100, ReLU, false, None);
    network.add_convolution(5, (5, 5), (1, 1), (1, 1), ReLU, false, Some(0.1));
    network.add_dense(1, Softmax, false, None);

    println!("{}", network);
}