// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, dense, network, tensor, feedback};

fn main() {
    let inputs = tensor::Shape::Single(12);

    let mut network = network::Network::new(inputs.clone(), feedback::Accumulation::Add);

    network.dense(12, activation::Activation::Linear, false, None);

    network.feedback(
        vec![
            network::Layer::Dense(dense::Dense::create(
                inputs.clone(),
                tensor::Shape::Single(24),
                &activation::Activation::ReLU,
                false,
                None,
            )),
            network::Layer::Dense(dense::Dense::create(
                tensor::Shape::Single(24),
                tensor::Shape::Single(2),
                &activation::Activation::LeakyReLU,
                false,
                None,
            )),
            network::Layer::Dense(dense::Dense::create(
                tensor::Shape::Single(2),
                inputs.clone(),
                &activation::Activation::Tanh,
                false,
                None,
            )),
        ],
        2,
    );

    network.dense(5, activation::Activation::Sigmoid, false, None);

    println!("{}", network);
}
