// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, objective, optimizer, tensor};

fn main() {
    // New feedforward network with input shape (1, 28, 28)
    let mut network = network::Network::new(tensor::Shape::Triple(1, 28, 28));

    // Convolution(filters, kernel, stride, padding, activation, Some(dropout))
    network.convolution(
        5,
        (3, 3),
        (1, 1),
        (1, 1),
        activation::Activation::ReLU,
        None,
    );

    // Maxpool(kernel, stride)
    network.maxpool((2, 2), (2, 2));

    // Dense(outputs, activation, bias, Some(dropout))
    network.dense(100, activation::Activation::ReLU, false, None);

    // Dense(outputs, activation, bias, Some(dropout))
    network.dense(10, activation::Activation::Softmax, false, None);

    network.set_optimizer(optimizer::Optimizer::AdamW(optimizer::AdamW {
        learning_rate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        decay: 0.01,

        // To be filled by the network:
        momentum: vec![],
        velocity: vec![],
    }));
    network.set_objective(
        objective::Objective::MSE, // Objective function
        Some((-1f32, 1f32)),       // Gradient clipping
    );

    println!("{}", network); // Display the network
}
