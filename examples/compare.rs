// Copyright (C) 2024 Hallvard Høyland Lavik
//
// Code for comparison between the various architectures.
// The respective loss and accuracies is stored to the file `~/output/compare.txt`.
//
// In addition, some simple probing of the networks are done.
// Namely, validating the trained networks with and without feedback and skip connections.
//
// for (
//   REGULAR,
//   FEEDBACK[approach=1],
//   FEEDBACK[approach=2, loops=2],
//   FEEDBACK[approach=2, loops=3]
// ) do {
//
//   for (NOSKIP, SKIP) do {
//
//     for (CLASSIFICATION, REGRESSION) do {
//
//       create the network
//       train the network
//       validate the network
//       store the loss and accuracy
//       probe the network
//       store the probing results
//
//     }
//   }
// }

use neurons::{activation, feedback, network, objective, optimizer, tensor};

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
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

    // Create the results file.
    let mut file = File::create("./output/compare.txt").unwrap();

    vec!["REGULAR", "FB1", "FB22", "FB23"]
        .iter()
        .for_each(|method| {
            println!("Method: {}", method);
            writeln!(file, "Method: {}", method).unwrap();

            vec![false, true].iter().for_each(|skip| {
                println!("> Skip: {}", skip);
                writeln!(file, "> Skip: {}", skip).unwrap();

                vec!["CLS", "REG"].iter().for_each(|problem| {
                    println!("  Problem: {}", problem);
                    writeln!(file, "  Problem: {}", problem).unwrap();

                    let mut network: network::Network;

                    // Create the network based on the architecture.
                    network = network::Network::new(tensor::Shape::Single(571));
                    network.dense(128, activation::Activation::ReLU, false, None);

                    // Check if the method is regular or feedback.
                    if method == &"REGULAR" || method == &"FB1" {
                        network.dense(256, activation::Activation::ReLU, false, None);
                        network.dense(128, activation::Activation::ReLU, false, None);

                        // Add the feedback loop if applicable.
                        if method == &"FB1" {
                            network.loopback(2, 1, Arc::new(|_loops| 1.0));
                        }
                    } else {
                        network.feedback(
                            vec![
                                feedback::Layer::Dense(
                                    256,
                                    activation::Activation::ReLU,
                                    false,
                                    None,
                                ),
                                feedback::Layer::Dense(
                                    128,
                                    activation::Activation::ReLU,
                                    false,
                                    None,
                                ),
                            ],
                            method.chars().nth(3).unwrap().to_digit(10).unwrap() as usize,
                            false,
                            feedback::Accumulation::Mean,
                        );
                    }

                    // Set the output layer based on the problem.
                    if method == &"REGRESSION" {
                        network.dense(1, activation::Activation::Linear, false, None);
                        network.set_objective(objective::Objective::RMSE, None);
                    } else {
                        network.dense(28, activation::Activation::Softmax, false, None);
                        network.set_objective(objective::Objective::CrossEntropy, None);
                    }

                    network.set_optimizer(optimizer::Adam::create(0.001, 0.9, 0.999, 1e-8, None));
                    println!("{}", network);

                    // Train the network
                    let (train_loss, val_loss, val_acc);
                    if method == &"REG" {
                        println!("  > Training the network for regression.");

                        (train_loss, val_loss, val_acc) = network.learn(
                            &x_train,
                            &y_train,
                            Some((&x_val, &y_val, 50)),
                            16,
                            500,
                            Some(100),
                        );
                    } else {
                        println!("  > Training the network for classification.");

                        (train_loss, val_loss, val_acc) = network.learn(
                            &x_train,
                            &class_train,
                            Some((&x_val, &class_val, 50)),
                            16,
                            500,
                            Some(100),
                        );
                    }

                    // Store the loss and accuracy.
                    writeln!(file, "   train loss: {:?}", train_loss).unwrap();
                    writeln!(file, "   val   loss: {:?}", val_loss).unwrap();
                    writeln!(file, "   val    acc: {:?}", val_acc).unwrap();

                    // Probe the network (if applicable).
                    if method != &"REGULAR" {
                        println!("   > Without feedback.");
                        writeln!(file, "   > Without feedback.").unwrap();

                        // Store the network's loopbacks and layers to restore them later.
                        let loopbacks = network.loopbacks.clone();
                        let layers = network.layers.clone();

                        // Remove the feedback loop.
                        if method == &"FB1" {
                            network.loopbacks = HashMap::new();
                        } else {
                            match &mut network.layers.get_mut(1).unwrap() {
                                network::Layer::Feedback(fb) => {
                                    // Only keep the first two layers.
                                    fb.layers = fb.layers.drain(0..2).collect();
                                }
                                _ => panic!("Invalid layer."),
                            };
                        }

                        let (val_loss, val_acc);
                        if method == &"REG" {
                            (val_loss, val_acc) = network.validate(&x_test, &y_test, 1e-6);
                        } else {
                            (val_loss, val_acc) = network.validate(&x_test, &class_test, 1e-6);
                        }
                        writeln!(file, "     val loss: {:?}", val_loss).unwrap();
                        writeln!(file, "     val  acc: {}", val_acc).unwrap();

                        // Restore the feedback loop.
                        network.loopbacks = loopbacks;
                        network.layers = layers;
                    }
                    if *skip {
                        println!("   > Without skip.");
                        writeln!(file, "   > Without skip.").unwrap();

                        network.connect = HashMap::new();

                        let (val_loss, val_acc);
                        if method == &"REG" {
                            (val_loss, val_acc) = network.validate(&x_test, &y_test, 1e-6);
                        } else {
                            (val_loss, val_acc) = network.validate(&x_test, &class_test, 1e-6);
                        }
                        writeln!(file, "     val loss: {:?}", val_loss).unwrap();
                        writeln!(file, "     val  acc: {}", val_acc).unwrap();
                    }
                });
            });
        });
}
