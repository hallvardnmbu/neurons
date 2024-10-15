// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, feedback, network, objective, optimizer, plot, tensor};

use std::{
    fs::File,
    io::{BufRead, BufReader},
};

fn data(
    path: &str,
) -> (
    (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
    ),
    (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
    ),
    (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
    ),
) {
    let reader = BufReader::new(File::open(&path).unwrap());

    let mut x_train: Vec<tensor::Tensor> = Vec::new();
    let mut y_train: Vec<tensor::Tensor> = Vec::new();
    let mut class_train: Vec<tensor::Tensor> = Vec::new();

    let mut x_test: Vec<tensor::Tensor> = Vec::new();
    let mut y_test: Vec<tensor::Tensor> = Vec::new();
    let mut class_test: Vec<tensor::Tensor> = Vec::new();

    let mut x_val: Vec<tensor::Tensor> = Vec::new();
    let mut y_val: Vec<tensor::Tensor> = Vec::new();
    let mut class_val: Vec<tensor::Tensor> = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let record: Vec<&str> = line.split(',').collect();

        let mut data: Vec<f32> = Vec::new();
        for i in 0..571 {
            data.push(record.get(i).unwrap().parse::<f32>().unwrap());
        }
        match record.get(573).unwrap() {
            &"Train" => {
                x_train.push(tensor::Tensor::single(data));
                y_train.push(tensor::Tensor::single(vec![record
                    .get(571)
                    .unwrap()
                    .parse::<f32>()
                    .unwrap()]));
                class_train.push(tensor::Tensor::one_hot(
                    record.get(572).unwrap().parse::<usize>().unwrap() - 1, // For zero-indexed.
                    28,
                ));
            }
            &"Test" => {
                x_test.push(tensor::Tensor::single(data));
                y_test.push(tensor::Tensor::single(vec![record
                    .get(571)
                    .unwrap()
                    .parse::<f32>()
                    .unwrap()]));
                class_test.push(tensor::Tensor::one_hot(
                    record.get(572).unwrap().parse::<usize>().unwrap() - 1, // For zero-indexed.
                    28,
                ));
            }
            &"Val" => {
                x_val.push(tensor::Tensor::single(data));
                y_val.push(tensor::Tensor::single(vec![record
                    .get(571)
                    .unwrap()
                    .parse::<f32>()
                    .unwrap()]));
                class_val.push(tensor::Tensor::one_hot(
                    record.get(572).unwrap().parse::<usize>().unwrap() - 1, // For zero-indexed.
                    28,
                ));
            }
            _ => panic!("> Unknown class."),
        }
    }

    // let mut generator = random::Generator::create(12345);
    // let mut indices: Vec<usize> = (0..x.len()).collect();
    // generator.shuffle(&mut indices);

    (
        (x_train, y_train, class_train),
        (x_test, y_test, class_test),
        (x_val, y_val, class_val),
    )
}

fn main() {
    // Load the ftir dataset
    let ((x_train, y_train, class_train), (x_test, y_test, class_test), (x_val, y_val, class_val)) =
        data("./examples/datasets/ftir.csv");

    let x_train: Vec<&tensor::Tensor> = x_train.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y_train.iter().collect();
    let class_train: Vec<&tensor::Tensor> = class_train.iter().collect();

    let x_test: Vec<&tensor::Tensor> = x_test.iter().collect();
    let y_test: Vec<&tensor::Tensor> = y_test.iter().collect();
    let class_test: Vec<&tensor::Tensor> = class_test.iter().collect();

    let x_val: Vec<&tensor::Tensor> = x_val.iter().collect();
    let y_val: Vec<&tensor::Tensor> = y_val.iter().collect();
    let class_val: Vec<&tensor::Tensor> = class_val.iter().collect();

    println!("Train data {}x{}", x_train.len(), x_train[0].shape,);
    println!("Test data {}x{}", x_test.len(), x_test[0].shape,);
    println!("Validation data {}x{}", x_val.len(), x_val[0].shape,);

    vec!["REGRESSION", "CLASSIFICATION"]
        .iter()
        .for_each(|method| {
            // Create the network
            let mut network = network::Network::new(tensor::Shape::Single(571));

            network.dense(128, activation::Activation::ReLU, false, None);

            network.feedback(
                vec![
                    feedback::Layer::Dense(256, activation::Activation::ReLU, false, None),
                    feedback::Layer::Dense(128, activation::Activation::ReLU, false, None),
                ],
                2,
                false,
                feedback::Accumulation::Mean,
            );

            if method == &"REGRESSION" {
                network.dense(1, activation::Activation::Linear, false, None);
                network.set_objective(objective::Objective::RMSE, None);
            } else {
                network.dense(28, activation::Activation::Softmax, false, None);
                network.set_objective(objective::Objective::CrossEntropy, None);
            }

            // Include skip connection bypassing the feedback block
            // network.connect(1, 2);
            // network.set_accumulation(feedback::Accumulation::Add);

            network.set_optimizer(optimizer::Adam::create(0.001, 0.9, 0.999, 1e-8, None));

            println!("{}", network);

            // Train the network
            let (train_loss, val_loss, val_acc);
            if method == &"REGRESSION" {
                println!("> Training the network for regression.");

                (train_loss, val_loss, val_acc) = network.learn(
                    &x_train,
                    &y_train,
                    Some((&x_val, &y_val, 50)),
                    16,
                    500,
                    Some(100),
                );
            } else {
                println!("> Training the network for classification.");

                (train_loss, val_loss, val_acc) = network.learn(
                    &x_train,
                    &class_train,
                    Some((&x_val, &class_val, 50)),
                    16,
                    500,
                    Some(100),
                );
            }
            plot::loss(
                &train_loss,
                &val_loss,
                &val_acc,
                &format!("FEEDBACK : FTIR : {}", method),
                &format!("./static/ftir-mlp-feedback-{}.png", method.to_lowercase()),
            );

            if method == &"REGRESSION" {
                // Use the network
                let prediction = network.predict(x_test.get(0).unwrap());
                println!(
                    "Prediction. Target: {}. Output: {}.",
                    y_test[0].data, prediction.data
                );
            } else {
                // Validate the network
                let (val_loss, val_acc) = network.validate(&x_test, &class_test, 1e-6);
                println!(
                    "Final validation accuracy: {:.2} % and loss: {:.5}",
                    val_acc * 100.0,
                    val_loss
                );

                // Use the network
                let prediction = network.predict(x_test.get(0).unwrap());
                println!(
                    "Prediction. Target: {}. Output: {}.",
                    class_test[0].argmax(),
                    prediction.argmax()
                );
            }
        });
}
