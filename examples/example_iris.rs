// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, objective, optimizer, random, tensor};

use std::{
    fs::File,
    io::{BufRead, BufReader},
};

fn data(path: &str) -> (Vec<tensor::Tensor>, Vec<tensor::Tensor>) {
    let reader = BufReader::new(File::open(&path).unwrap());

    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<usize> = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let record: Vec<&str> = line.split(',').collect();
        x.push(vec![
            record.get(1).unwrap().parse::<f32>().unwrap(),
            record.get(2).unwrap().parse::<f32>().unwrap(),
            record.get(3).unwrap().parse::<f32>().unwrap(),
            record.get(4).unwrap().parse::<f32>().unwrap(),
        ]);
        y.push(match record.get(5).unwrap() {
            &"Iris-setosa" => 0,
            &"Iris-versicolor" => 1,
            &"Iris-virginica" => 2,
            _ => panic!("> Unknown class."),
        });
    }

    let mut generator = random::Generator::create(12345);
    let mut indices: Vec<usize> = (0..x.len()).collect();
    generator.shuffle(&mut indices);

    let x: Vec<tensor::Tensor> = indices
        .iter()
        .map(|&i| tensor::Tensor::single(x[i].clone()))
        .collect();
    let y: Vec<tensor::Tensor> = indices
        .iter()
        .map(|&i| tensor::Tensor::one_hot(y[i], 3))
        .collect();

    (x, y)
}

fn main() {
    // Load the iris dataset
    let (x, y) = data("./examples/datasets/iris.csv");

    let split = (x.len() as f32 * 0.8) as usize;
    let x = x.split_at(split);
    let y = y.split_at(split);

    let x_train: Vec<&tensor::Tensor> = x.0.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y.0.iter().collect();
    let x_test: Vec<&tensor::Tensor> = x.1.iter().collect();
    let y_test: Vec<&tensor::Tensor> = y.1.iter().collect();

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
    let mut network = network::Network::new(tensor::Shape::Single(4));

    network.dense(50, activation::Activation::ReLU, false, None);
    network.dense(50, activation::Activation::ReLU, false, None);
    network.dense(3, activation::Activation::Softmax, false, None);

    network.set_optimizer(optimizer::RMSprop::create(
        0.001,      // Learning rate
        0.0,        // Alpha
        1e-8,       // Epsilon
        Some(0.01), // Decay
        Some(0.01), // Momentum
        true,       // Centered
    ));
    network.set_objective(
        objective::Objective::CrossEntropy, // Objective function
        Some((-1f32, 1f32)),                // Gradient clipping
    );

    // Train the network
    let (_train_loss, _val_loss) = network.learn(
        &x_train,
        &y_train,
        Some((&x_test, &y_test, 5)),
        25,
        5,
        Some(1),
    );

    // Validate the network
    let (val_loss, val_acc) = network.validate(&x_test, &y_test, 1e-6);
    println!(
        "1. Validation acc: {:.2}, loss: {:.5}",
        val_acc * 100.0,
        val_loss
    );

    // Use the network
    let prediction = network.predict(x_test.get(0).unwrap());
    println!(
        "2. Input: {}, Target: {}, Output: {}",
        x_test[0].data,
        y_test[0].argmax(),
        prediction.argmax()
    );
}
