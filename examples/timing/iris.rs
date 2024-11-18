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
    let x: Vec<&tensor::Tensor> = x.iter().collect();
    let y: Vec<&tensor::Tensor> = y.iter().collect();

    // Create the results file.
    let mut file = File::create("./output/timing/iris.json").unwrap();
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
            vec!["CLASSIFICATION"].iter().for_each(|problem| {
                println!("   Problem: {}", problem);

                let mut train_times: Vec<f64> = Vec::new();
                let mut valid_times: Vec<f64> = Vec::new();

                for _ in 0..RUNS {
                    // Create the network based on the architecture.
                    let mut network: network::Network;
                    network = network::Network::new(tensor::Shape::Single(4));

                    // Check if the method is regular or feedback.
                    if method == &"REGULAR" || method.contains(&"FB1") {
                        network.dense(25, activation::Activation::ReLU, false, None);
                        network.dense(25, activation::Activation::ReLU, false, None);
                        network.dense(25, activation::Activation::ReLU, false, None);

                        // Add the feedback loop if applicable.
                        if method.contains(&"FB1") {
                            network.loopback(
                                2,
                                1,
                                method.chars().last().unwrap().to_digit(10).unwrap() as usize - 1,
                                Arc::new(|_loops| 1.0),
                                false,
                            );
                        }
                    } else {
                        network.dense(25, activation::Activation::ReLU, false, None);
                        network.feedback(
                            vec![
                                feedback::Layer::Dense(
                                    25,
                                    activation::Activation::ReLU,
                                    false,
                                    None,
                                ),
                                feedback::Layer::Dense(
                                    25,
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
                        panic!("Invalid problem type.");
                    } else {
                        network.dense(3, activation::Activation::Softmax, false, None);
                        network.set_objective(objective::Objective::CrossEntropy, None);
                    }

                    // Add the skip connection if applicable.
                    if *skip {
                        network.connect(1, network.layers.len() - 1);
                    }

                    network.set_optimizer(optimizer::Adam::create(0.0001, 0.95, 0.999, 1e-7, None));

                    let start = time::Instant::now();

                    // Train the network
                    if problem == &"REGRESSION" {
                        panic!("Invalid problem type.");
                    } else {
                        (_, _, _) = network.learn(&x, &y, None, 1, EPOCHS, None);
                    }

                    let duration = start.elapsed().as_secs_f64();
                    train_times.push(duration);

                    let start = time::Instant::now();

                    // Validate the network
                    if problem == &"REGRESSION" {
                        panic!("Invalid problem type.");
                    } else {
                        (_) = network.predict_batch(&x);
                    }

                    let duration = start.elapsed().as_secs_f64();
                    valid_times.push(duration);
                }

                if method == &"FB2x4" && *skip && problem == &"CLASSIFICATION" {
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
