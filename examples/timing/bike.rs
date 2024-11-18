// Copyright (C) 2024 Hallvard Høyland Lavik
//
// Code for comparison between the various architectures and their time differences.

use neurons::{activation, feedback, network, objective, optimizer, random, tensor};

use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
    time,
};

const RUNS: usize = 5;
const EPOCHS: i32 = 1;

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
    // Load the bike dataset
    let (x, y) = data("./examples/datasets/bike/hour.csv");
    let x: Vec<&tensor::Tensor> = x.iter().collect();
    let y: Vec<&tensor::Tensor> = y.iter().collect();

    // Create the results file.
    let mut file = File::create("./output/timing/bike.json").unwrap();
    writeln!(file, "[").unwrap();
    writeln!(file, "  {{").unwrap();

    vec![
        "REGULAR", "FB1x2", "FB1x3", "FB1x4", "FB2x2", "FB2x3", "FB2x4",
    ]
    .iter()
    .for_each(|method| {
        println!("Method: {}", method);
        vec![false, true].iter().for_each(|skip| {
            println!("  Skip: {}", skip);
            vec!["REGRESSION"].iter().for_each(|problem| {
                println!("   Problem: {}", problem);

                let mut train_times: Vec<f64> = Vec::new();
                let mut valid_times: Vec<f64> = Vec::new();

                for _ in 0..RUNS {
                    // Create the network based on the architecture.
                    let mut network: network::Network;
                    network = network::Network::new(tensor::Shape::Single(12));

                    // Check if the method is regular or feedback.
                    if method == &"REGULAR" || method.contains(&"FB1") {
                        network.dense(24, activation::Activation::ReLU, false, None);
                        network.dense(24, activation::Activation::ReLU, false, None);
                        network.dense(24, activation::Activation::ReLU, false, None);

                        // Add the feedback loop if applicable.
                        if method.contains(&"FB1") {
                            network.loopback(
                                2,
                                1,
                                method.chars().last().unwrap().to_digit(10).unwrap() as usize - 1,
                                Arc::new(|_loops| 1.0),
                            );
                        }
                    } else {
                        network.dense(24, activation::Activation::ReLU, false, None);
                        network.feedback(
                            vec![
                                feedback::Layer::Dense(
                                    24,
                                    activation::Activation::ReLU,
                                    false,
                                    None,
                                ),
                                feedback::Layer::Dense(
                                    24,
                                    activation::Activation::ReLU,
                                    false,
                                    None,
                                ),
                            ],
                            method.chars().last().unwrap().to_digit(10).unwrap() as usize,
                            false,
                            false,
                            feedback::Accumulation::Mean,
                        );
                    }

                    // Set the output layer based on the problem.
                    if problem == &"REGRESSION" {
                        network.dense(1, activation::Activation::Linear, false, None);
                        network.set_objective(objective::Objective::RMSE, None);
                    } else {
                        panic!("Invalid problem type.");
                    }

                    // Add the skip connection if applicable.
                    if *skip {
                        network.connect(1, network.layers.len() - 1);
                    }

                    network.set_optimizer(optimizer::Adam::create(0.01, 0.9, 0.999, 1e-4, None));

                    let start = time::Instant::now();

                    // Train the network
                    if problem == &"CLASSIFICATION" {
                        panic!("Invalid problem type.");
                    } else {
                        (_, _, _) = network.learn(&x, &y, None, 64, EPOCHS, None);
                    }

                    let duration = start.elapsed().as_secs_f64();
                    train_times.push(duration);

                    let start = time::Instant::now();

                    // Validate the network
                    if problem == &"CLASSIFICATION" {
                        panic!("Invalid problem type.");
                    } else {
                        (_) = network.predict_batch(&x);
                    }

                    let duration = start.elapsed().as_secs_f64();
                    valid_times.push(duration);
                }

                if method == &"FB2x4" && *skip && problem == &"REGRESSION" {
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
