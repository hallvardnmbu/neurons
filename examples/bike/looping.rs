// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, objective, optimizer, plot, random, tensor};

use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::Arc,
};

fn data(path: &str) -> (Vec<tensor::Tensor>, Vec<tensor::Tensor>) {
    let reader = BufReader::new(File::open(&path).unwrap());

    let mut x: Vec<tensor::Tensor> = Vec::new();
    let mut y: Vec<tensor::Tensor> = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let record: Vec<&str> = line.split(',').collect();

        let mut data: Vec<f32> = Vec::new();
        for i in 2..14 {
            data.push(record.get(i).unwrap().parse::<f32>().unwrap());
        }
        x.push(tensor::Tensor::single(data));

        y.push(tensor::Tensor::single(vec![record
            .get(16)
            .unwrap()
            .parse::<f32>()
            .unwrap()]));
    }

    let mut generator = random::Generator::create(12345);
    let mut indices: Vec<usize> = (0..x.len()).collect();
    generator.shuffle(&mut indices);

    let x: Vec<tensor::Tensor> = indices.iter().map(|i| x[*i].clone()).collect();
    let y: Vec<tensor::Tensor> = indices.iter().map(|i| y[*i].clone()).collect();

    (x, y)
}

fn main() {
    // Load the ftir dataset
    let (x, y) = data("./examples/datasets/bike/hour.csv");

    let split = (x.len() as f32 * 0.8) as usize;
    let x = x.split_at(split);
    let y = y.split_at(split);

    let x_train: Vec<&tensor::Tensor> = x.0.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y.0.iter().collect();
    let x_test: Vec<&tensor::Tensor> = x.1.iter().collect();
    let y_test: Vec<&tensor::Tensor> = y.1.iter().collect();

    // Create the network
    let mut network = network::Network::new(tensor::Shape::Single(12));

    network.dense(24, activation::Activation::ReLU, false, None);
    network.dense(24, activation::Activation::ReLU, false, None);
    network.dense(24, activation::Activation::ReLU, false, None);

    network.dense(1, activation::Activation::Linear, false, None);
    network.set_objective(objective::Objective::RMSE, None);

    network.loopback(2, 1, 2, Arc::new(|_loops| 1.0));

    network.set_optimizer(optimizer::Adam::create(0.01, 0.9, 0.999, 1e-4, None));

    println!("{}", network);

    // Train the network

    let (train_loss, val_loss, val_acc) = network.learn(
        &x_train,
        &y_train,
        Some((&x_test, &y_test, 25)),
        64,
        600,
        Some(100),
    );
    plot::loss(
        &train_loss,
        &val_loss,
        &val_acc,
        &"LOOP : BIKE",
        &"./output/bike/loop.png",
    );

    // Use the network
    let prediction = network.predict(x_test.get(0).unwrap());
    println!(
        "Prediction. Target: {}. Output: {}.",
        y_test[0].data, prediction.data
    );
}
