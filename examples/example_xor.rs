// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, objective, optimizer, tensor, feedback};

fn main() {
    // Create the training data for the binary AND operation
    let x: Vec<tensor::Tensor> = vec![
        tensor::Tensor::single(vec![0.0, 0.0]),
        tensor::Tensor::single(vec![0.0, 1.0]),
        tensor::Tensor::single(vec![1.0, 0.0]),
        tensor::Tensor::single(vec![1.0, 1.0]),
    ];
    let y: Vec<tensor::Tensor> = vec![
        tensor::Tensor::single(vec![0.0]),
        tensor::Tensor::single(vec![0.0]),
        tensor::Tensor::single(vec![0.0]),
        tensor::Tensor::single(vec![1.0]),
    ];

    let inputs: Vec<&tensor::Tensor> = x.iter().collect();
    let targets: Vec<&tensor::Tensor> = y.iter().collect();

    // Create the network
    let mut network = network::Network::new(tensor::Shape::Single(2), feedback::Accumulation::Add);

    network.dense(10, activation::Activation::ReLU, true, None);
    network.dense(1, activation::Activation::Sigmoid, false, None);

    network.set_optimizer(optimizer::SGD::create(0.1, Some(0.01)));
    network.set_objective(objective::Objective::BinaryCrossEntropy, None);

    // Train the network
    let _epoch_loss = network.learn(&inputs, &targets, None, 4, 500, Some(50));

    // Validate the network
    let (val_loss, val_acc) = network.validate(&inputs, &targets, 1e-1);
    println!(
        "Final validation accuracy: {:.2} % and loss: {:.5}",
        val_acc * 100.0,
        val_loss
    );

    // Use the network
    let prediction = network.predict(inputs.get(0).unwrap());
    println!(
        "Prediction on input: {} Target: {} Output: {}",
        inputs[0].data, targets[0].data, prediction.data
    );
}
