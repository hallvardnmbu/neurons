// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, objective, optimizer, random, tensor};

use std::time;
extern crate csv;

fn data(path: &str) -> (Vec<tensor::Tensor>, Vec<tensor::Tensor>) {
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

    (x, y)
}

fn main() {
    // Load the iris dataset
    let (x, y) = data("./datasets/iris.csv");

    let split = (x.len() as f32 * 0.8) as usize;
    let x = x.split_at(split);
    let y = y.split_at(split);

    let x_train: Vec<&tensor::Tensor> = x.0.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y.0.iter().collect();
    let x_test: Vec<&tensor::Tensor> = x.1.iter().collect();
    let y_test: Vec<&tensor::Tensor> = y.1.iter().collect();

    let (x_train, y_train, x_test, y_test) = (
        x_train.to_vec(),
        y_train.to_vec(),
        x_test.to_vec(),
        y_test.to_vec(),
    );
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

    let mut times: Vec<time::Duration> = Vec::new();

    for _ in 0..10 {
        let start = time::Instant::now();

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
        let _epoch_loss = network.learn(&x_train, &y_train, Some(25), 500);

        let duration = start.elapsed();
        times.push(duration);
    }

    let sum: time::Duration = times.iter().sum();
    let avg = sum / times.len() as u32;
    println!("Average time: {:?}", avg);
}
