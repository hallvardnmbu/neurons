// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, tensor};

fn main() {
    let mut network = network::Network::new(tensor::Shape::Single(2));

    network.dense(100, activation::Activation::ReLU, false, None);
    network.convolution(
        5,
        (5, 5),
        (1, 1),
        (1, 1),
        activation::Activation::ReLU,
        Some(0.1),
    );
    network.dense(1, activation::Activation::Softmax, false, None);

    println!("{}", network);
}
