// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, objective, optimizer, tensor};

fn main() {
    // Create the training data for the binary AND operation
    let x: Vec<tensor::Tensor> = vec![
        tensor::Tensor::vector(vec![0.0, 0.0]),
        tensor::Tensor::vector(vec![0.0, 1.0]),
        tensor::Tensor::vector(vec![1.0, 0.0]),
        tensor::Tensor::vector(vec![1.0, 1.0]),
    ];
    let y: Vec<tensor::Tensor> = vec![
        tensor::Tensor::vector(vec![0.0]),
        tensor::Tensor::vector(vec![0.0]),
        tensor::Tensor::vector(vec![0.0]),
        tensor::Tensor::vector(vec![1.0]),
    ];

    let inputs: Vec<&tensor::Tensor> = x.iter().collect();
    let targets: Vec<&tensor::Tensor> = y.iter().collect();

    // Create the network
    let mut network = network::Network::new(tensor::Shape::Vector(2));

    network.dense(10, activation::Activation::Linear, false, None);
    network.dense(1, activation::Activation::Sigmoid, false, None);

    network.set_optimizer(optimizer::Optimizer::SGD(optimizer::SGD {
        learning_rate: 0.1,
        decay: Some(0.01),
    }));
    network.set_objective(objective::Objective::BinaryCrossEntropy, None);

    // Train the network
    let _epoch_loss = network.learn(&inputs, &targets, None, 4, 500, Some(50));

    // Validate the network
    let (val_loss, val_acc) = network.validate(&inputs, &targets, 0.1);
    println!("1. Validation acc: {} loss: {}", val_acc, val_loss);

    // Use the network
    let prediction = network.predict(inputs.get(0).unwrap());
    println!(
        "2. Input: {}, Target: {}, Output: {}",
        inputs[0].data, targets[0].data, prediction.data
    );

    // Use the network on batch
    let predictions = network.predict_batch(&inputs);
    println!(
        "3. Input: {},\n   Target: {},\n   Output: {}",
        inputs[0].data, targets[0].data, predictions[0].data
    );
}
