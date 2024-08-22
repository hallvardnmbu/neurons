// Copyright (C) 2024 Hallvard Høyland Lavik

extern crate csv;

use neurons::{activation, network, objective, optimizer, random, tensor};

fn data(
    path: &str,
) -> (
    Vec<tensor::Tensor>,
    Vec<tensor::Tensor>,
    Vec<tensor::Tensor>,
    Vec<tensor::Tensor>,
) {
    let mut reader = csv::Reader::from_path(path).unwrap();

    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<Vec<f32>> = Vec::new();

    reader.records().for_each(|record| {
        let record = record.unwrap();
        x.push(vec![
            record.get(1).unwrap().parse::<f32>().unwrap(),
            record.get(2).unwrap().parse::<f32>().unwrap(),
            record.get(3).unwrap().parse::<f32>().unwrap(),
            record.get(4).unwrap().parse::<f32>().unwrap(),
        ]);
        y.push(match record.get(5).unwrap() {
            "Iris-setosa" => vec![1.0, 0.0, 0.0],
            "Iris-versicolor" => vec![0.0, 1.0, 0.0],
            "Iris-virginica" => vec![0.0, 0.0, 1.0],
            // "Iris-setosa" => vec![0.0],
            // "Iris-versicolor" => vec![1.0],
            // "Iris-virginica" => vec![2.0],
            _ => panic!("Unknown class"),
        });
    });

    let mut generator = random::Generator::create(12345);
    let mut indices: Vec<usize> = (0..x.len()).collect();
    generator.shuffle(&mut indices);

    let x: Vec<tensor::Tensor> = indices
        .iter()
        .map(|&i| tensor::Tensor::from_single(x[i].clone()))
        .collect();
    let y: Vec<tensor::Tensor> = indices
        .iter()
        .map(|&i| tensor::Tensor::from_single(y[i].clone()))
        .collect();

    let split = (x.len() as f32 * 0.8) as usize;
    let x = x.split_at(split);
    let y = y.split_at(split);

    let x_train = x.0.to_vec();
    let y_train = y.0.to_vec();
    let x_test = x.1.to_vec();
    let y_test = y.1.to_vec();

    (x_train, y_train, x_test, y_test)
}

fn main() {
    // Load the iris dataset
    let (x_train, y_train, x_test, y_test) = data("./datasets/iris.csv");
    println!(
        "Train data {}x{}: {} => {}",
        x_train.len(),
        x_train[0].shape,
        x_train[0].data,
        y_train[0].data
    );
    println!(
        "Test data {}x{}: {} => {}",
        x_test.len(),
        x_test[0].shape,
        x_test[0].data,
        y_test[0].data
    );

    // Create the network
    let mut network = network::Network::new(tensor::Shape::Vector(4));

    network.dense(50, activation::Activation::ReLU, false, Some(0.1));
    network.dense(50, activation::Activation::ReLU, false, Some(0.1));
    network.dense(3, activation::Activation::Softmax, false, Some(0.1));

    network.set_optimizer(optimizer::Optimizer::RMSprop(optimizer::RMSprop {
        learning_rate: 0.001,
        alpha: 0.0,
        epsilon: 1e-8,

        decay: Some(0.01),
        momentum: Some(0.01),
        centered: Some(true),

        // To be filled by the network:
        velocity: vec![],
        gradient: vec![],
        buffer: vec![],
    }));
    network.set_objective(
        objective::Objective::CrossEntropy, // Objective function
        Some((-1f32, 1f32)),                // Gradient clipping
    );

    // Train the network
    let _epoch_loss = network.learn(&x_train, &y_train, 500);

    // Validate the network
    let (val_acc, val_loss) = network.validate(&x_test, &y_test, 0.1);
    println!("1. Validation acc: {}, loss: {}", val_acc, val_loss);

    // Use the network
    let prediction = network.predict(x_test.get(0).unwrap());
    println!(
        "2. Input: {}, Target: {}, Output: {}",
        x_test[0].data, y_test[0].data, prediction
    );
}
