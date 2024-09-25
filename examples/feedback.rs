// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, dense, network, tensor};

fn main() {
    let mut network = network::Network::new(tensor::Shape::Single(32 * 32));

    network.dense(256, activation::Activation::Linear, false, None);

    network.feedback(
        vec![
            network::Layer::Dense(dense::Dense::create(
                tensor::Shape::Single(256),
                tensor::Shape::Single(124),
                &activation::Activation::ReLU,
                false,
                None,
            )),
            network::Layer::Dense(dense::Dense::create(
                tensor::Shape::Single(124),
                tensor::Shape::Single(86),
                &activation::Activation::LeakyReLU,
                false,
                None,
            )),
            network::Layer::Dense(dense::Dense::create(
                tensor::Shape::Single(86),
                tensor::Shape::Single(256),
                &activation::Activation::Tanh,
                false,
                None,
            )),
        ],
        2,
    );

    network.dense(10, activation::Activation::Sigmoid, false, None);

    println!("{}", network);
}
