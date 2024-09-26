// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, feedback, network, tensor};

fn main() {
    let mut network = network::Network::new(tensor::Shape::Single(32 * 32));

    network.dense(256, activation::Activation::Linear, false, None);

    network.feedback(
        vec![
            feedback::Layer::Dense(124, activation::Activation::ReLU, false, None),
            feedback::Layer::Dense(86, activation::Activation::ReLU, false, None),
            feedback::Layer::Dense(256, activation::Activation::ReLU, false, None),
        ],
        2,
        true,
    );

    network.dense(10, activation::Activation::Sigmoid, false, None);

    println!("{}", network);
}
