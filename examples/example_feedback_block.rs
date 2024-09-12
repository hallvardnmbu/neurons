// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, dense, network, tensor};

fn main() {
    let inputs = tensor::Shape::Single(12);

    let mut network = network::Network::new(inputs.clone());

    network.dense(12, activation::Activation::Linear, false, None);

    network.coupled(
        vec![
            network::Layer::Dense(dense::Dense::create(
                inputs.clone(),
                inputs.clone(),
                &activation::Activation::ReLU,
                false,
                None,
            )),
            network::Layer::Dense(dense::Dense::create(
                inputs.clone(),
                inputs.clone(),
                &activation::Activation::LeakyReLU,
                false,
                None,
            )),
            network::Layer::Dense(dense::Dense::create(
                inputs.clone(),
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
