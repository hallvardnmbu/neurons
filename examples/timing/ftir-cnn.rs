// Copyright (C) 2024 Hallvard Høyland Lavik
//
// Code for comparison between the various architectures and their time differences.

use neurons::{activation, feedback, network, objective, optimizer, tensor};

use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
    time,
};

const RUNS: usize = 5;
const EPOCHS: i32 = 1;

fn data(
    path: &str,
) -> (
    Vec<tensor::Tensor>,
    Vec<tensor::Tensor>,
    Vec<tensor::Tensor>,
) {
    let reader = BufReader::new(File::open(&path).unwrap());

    let mut x: Vec<tensor::Tensor> = Vec::new();
    let mut y: Vec<tensor::Tensor> = Vec::new();
    let mut c: Vec<tensor::Tensor> = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let record: Vec<&str> = line.split(',').collect();

        let mut data: Vec<f32> = Vec::new();
        for i in 0..571 {
            data.push(record.get(i).unwrap().parse::<f32>().unwrap());
        }
        let data: Vec<Vec<Vec<f32>>> = vec![vec![data]];
        x.push(tensor::Tensor::triple(data));
        y.push(tensor::Tensor::single(vec![record
            .get(571)
            .unwrap()
            .parse::<f32>()
            .unwrap()]));
        c.push(tensor::Tensor::one_hot(
            record.get(572).unwrap().parse::<usize>().unwrap() - 1, // For zero-indexed.
            28,
        ));
    }

    (x, y, c)
}

fn main() {
    // Load the ftir dataset
    let (x, y, c) = data("./examples/datasets/ftir.csv");

    let x: Vec<&tensor::Tensor> = x.iter().collect();
    let y: Vec<&tensor::Tensor> = y.iter().collect();
    let c: Vec<&tensor::Tensor> = c.iter().collect();

    // Create the results file.
    let mut file = File::create("./output/timing/ftir-cnn.json").unwrap();
    writeln!(file, "[").unwrap();
    writeln!(file, "  {{").unwrap();

    vec!["REGULAR", "FB1", "FB2x2", "FB2x3"]
        .iter()
        .for_each(|method| {
            println!("Method: {}", method);
            vec![false, true].iter().for_each(|skip| {
                println!("  Skip: {}", skip);
                vec!["CLASSIFICATION", "REGRESSION"]
                    .iter()
                    .for_each(|problem| {
                        println!("   Problem: {}", problem);

                        let mut train_times: Vec<f64> = Vec::new();
                        let mut valid_times: Vec<f64> = Vec::new();

                        for _ in 0..RUNS {
                            // Create the network based on the architecture.
                            let mut network: network::Network;
                            network = network::Network::new(tensor::Shape::Triple(1, 1, 571));

                            // Check if the method is regular or feedback.
                            if method == &"REGULAR" || method == &"FB1" {
                                network.convolution(
                                    1,
                                    (1, 9),
                                    (1, 1),
                                    (0, 4),
                                    (1, 1),
                                    activation::Activation::ReLU,
                                    None,
                                );
                                network.convolution(
                                    1,
                                    (1, 9),
                                    (1, 1),
                                    (0, 4),
                                    (1, 1),
                                    activation::Activation::ReLU,
                                    None,
                                );
                                network.dense(32, activation::Activation::ReLU, false, None);

                                // Add the feedback loop if applicable.
                                if method == &"FB1" {
                                    network.loopback(1, 0, Arc::new(|_loops| 1.0));
                                }
                            } else {
                                network.feedback(
                                    vec![
                                        feedback::Layer::Convolution(
                                            1,
                                            activation::Activation::ReLU,
                                            (1, 9),
                                            (1, 1),
                                            (0, 4),
                                            (1, 1),
                                            None,
                                        ),
                                        feedback::Layer::Convolution(
                                            1,
                                            activation::Activation::ReLU,
                                            (1, 9),
                                            (1, 1),
                                            (0, 4),
                                            (1, 1),
                                            None,
                                        ),
                                    ],
                                    method.chars().last().unwrap().to_digit(10).unwrap() as usize,
                                    false,
                                    feedback::Accumulation::Mean,
                                );
                                network.dense(32, activation::Activation::ReLU, false, None);
                            }

                            // Set the output layer based on the problem.
                            if problem == &"REGRESSION" {
                                network.dense(1, activation::Activation::Linear, false, None);
                                network.set_objective(objective::Objective::RMSE, None);
                            } else {
                                network.dense(28, activation::Activation::Softmax, false, None);
                                network.set_objective(
                                    objective::Objective::CrossEntropy,
                                    Some((-5.0, 5.0)),
                                );
                            }

                            // Add the skip connection if applicable.
                            if *skip {
                                network.connect(0, network.layers.len() - 2);
                            }

                            network.set_optimizer(optimizer::Adam::create(
                                0.001, 0.9, 0.999, 1e-8, None,
                            ));

                            let start = time::Instant::now();

                            // Train the network
                            if problem == &"REGRESSION" {
                                (_, _, _) = network.learn(&x, &y, None, 32, EPOCHS, None);
                            } else {
                                (_, _, _) = network.learn(&x, &c, None, 32, EPOCHS, None);
                            }

                            let duration = start.elapsed().as_secs_f64();
                            train_times.push(duration);

                            let start = time::Instant::now();

                            // Validate the network
                            (_) = network.predict_batch(&x);

                            let duration = start.elapsed().as_secs_f64();
                            valid_times.push(duration);
                        }

                        if method == &"FB2x3" && *skip && problem == &"REGRESSION" {
                            writeln!(
                                file,
                                "    \"{}-{}-{}\": {{\"train\": {:?}, \"validate\": {:?}}}",
                                method, skip, problem, train_times, valid_times
                            )
                            .unwrap();
                        } else {
                            writeln!(
                                file,
                                "    \"{}-{}-{}\": {{\"train\": {:?}, \"validate\": {:?}}},",
                                method, skip, problem, train_times, valid_times
                            )
                            .unwrap();
                        }
                    });
            });
        });
    writeln!(file, "  }}").unwrap();
    writeln!(file, "]").unwrap();
}
