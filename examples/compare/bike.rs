// Copyright (C) 2024 Hallvard Høyland Lavik
//
// Code for comparison between the various architectures.
// The respective loss and accuracies is stored to the file `~/output/compare/bike.json`.
//
// In addition, some simple probing of the networks are done.
// Namely, validating the trained networks with and without feedback and skip connections.
//
// for (
//   REGULAR,
//   FEEDBACK[approach=1, loops=2],
//   FEEDBACK[approach=1, loops=3],
//   FEEDBACK[approach=1, loops=4],
//   FEEDBACK[approach=2, loops=2],
//   FEEDBACK[approach=2, loops=3],
//   FEEDBACK[approach=2, loops=4]
// ) do {
//
//   for (NOSKIP, SKIP) do {
//
//     for (CLASSIFICATION) do {
//
//       for (run in RUNS) do {
//
//         create the network
//         train the network
//         validate the network
//         store the loss and accuracy
//         probe the network
//         store the probing results
//
//       }
//     }
//   }
// }

use neurons::{activation, feedback, network, objective, optimizer, random, tensor};

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};

const RUNS: usize = 5;

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

    let split = (x.len() as f32 * 0.8) as usize;
    let x = x.split_at(split);
    let y = y.split_at(split);

    let x_train: Vec<&tensor::Tensor> = x.0.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y.0.iter().collect();
    let x_test: Vec<&tensor::Tensor> = x.1.iter().collect();
    let y_test: Vec<&tensor::Tensor> = y.1.iter().collect();

    println!("Train data {}x{}", x_train.len(), x_train[0].shape,);
    println!("Test data {}x{}", x_test.len(), x_test[0].shape,);

    // Create the results file.
    let mut file = File::create("./output/compare/bike.json").unwrap();
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
                writeln!(file, "    \"{}-{}-{}\": {{", method, skip, problem).unwrap();

                for run in 1..RUNS + 1 {
                    println!("    Run: {}", run);
                    writeln!(file, "      \"run-{}\": {{", run).unwrap();

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
                                false,
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

                    // Train the network
                    let (train_loss, val_loss, val_acc);
                    if problem == &"REGRESSION" {
                        (train_loss, val_loss, val_acc) = network.learn(
                            &x_train,
                            &y_train,
                            Some((&x_test, &y_test, 25)),
                            64,
                            600,
                            None,
                        );
                    } else {
                        panic!("Invalid problem type.");
                    }

                    // Store the loss and accuracy.
                    writeln!(file, "        \"train\": {{").unwrap();
                    writeln!(file, "          \"trn-loss\": {:?},", train_loss).unwrap();
                    writeln!(file, "          \"val-loss\": {:?},", val_loss).unwrap();
                    writeln!(file, "          \"val-acc\": {:?}", val_acc).unwrap();

                    // Probe the network (if applicable).
                    if method != &"REGULAR" {
                        println!("    > Without feedback.");

                        // Store the network's loopbacks and layers to restore them later.
                        let loopbacks = network.loopbacks.clone();
                        let layers = network.layers.clone();

                        // Remove the feedback loop.
                        if method.contains(&"FB1") {
                            network.loopbacks = HashMap::new();
                        } else {
                            match &mut network.layers.get_mut(1).unwrap() {
                                network::Layer::Feedback(fb) => {
                                    // Only keep the first layer.
                                    fb.layers = fb.layers.drain(0..2).collect();
                                }
                                _ => panic!("Invalid layer."),
                            };
                        }

                        let (test_loss, test_acc);
                        if problem == &"REGRESSION" {
                            (test_loss, test_acc) = network.validate(&x_test, &y_test, 1e-6);
                        } else {
                            panic!("Invalid problem type.");
                        }

                        writeln!(file, "        }},").unwrap();
                        writeln!(file, "        \"no-feedback\": {{").unwrap();
                        writeln!(file, "          \"tst-loss\": {},", test_loss).unwrap();
                        writeln!(file, "          \"tst-acc\": {}", test_acc).unwrap();

                        // Restore the feedback loop.
                        network.loopbacks = loopbacks;
                        network.layers = layers;
                    }
                    if *skip {
                        println!("    > Without skip.");
                        network.connect = HashMap::new();

                        let (test_loss, test_acc);
                        if problem == &"REGRESSION" {
                            (test_loss, test_acc) = network.validate(&x_test, &y_test, 1e-6);
                        } else {
                            panic!("Invalid problem type.");
                        }

                        writeln!(file, "        }},").unwrap();
                        writeln!(file, "        \"no-skip\": {{").unwrap();
                        writeln!(file, "          \"tst-loss\": {},", test_loss).unwrap();
                        writeln!(file, "          \"tst-acc\": {}", test_acc).unwrap();
                    }
                    writeln!(file, "        }}").unwrap();

                    if run == RUNS {
                        writeln!(file, "      }}").unwrap();
                        if method == &"FB2x4" && *skip && problem == &"REGRESSION" {
                            writeln!(file, "    }}").unwrap();
                        } else {
                            writeln!(file, "    }},").unwrap();
                        }
                    } else {
                        writeln!(file, "      }},").unwrap();
                    }
                }
            });
        });
    });
    writeln!(file, "  }}").unwrap();
    writeln!(file, "]").unwrap();
}
