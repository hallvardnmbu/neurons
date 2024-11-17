// Copyright (C) 2024 Hallvard Høyland Lavik
//
// Code for comparison between the various architectures.
// The respective loss and accuracies is stored to the file `~/output/compare/fashion-2.json`.
//
// In addition, some simple probing of the networks are done.
// Namely, validating the trained networks with and without feedback and skip connections.
//
// for (
//   REGULAR,
//   FEEDBACK[approach=1, loops=2],
//   FEEDBACK[approach=1, loops=3],
//   FEEDBACK[approach=2, loops=2],
//   FEEDBACK[approach=2, loops=3]
// ) do {
//
//   for (NOSKIP, SKIP) do {
//
//     for (CLASSIFICATION, REGRESSION) do {
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

use neurons::{activation, feedback, network, objective, optimizer, tensor};

use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read, Result, Write},
    sync::Arc,
};

const RUNS: usize = 5;

fn read(reader: &mut dyn Read) -> Result<u32> {
    let mut buffer = [0; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

fn load_mnist(path: &str) -> Result<Vec<tensor::Tensor>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut images: Vec<tensor::Tensor> = Vec::new();

    let _magic_number = read(&mut reader)?;
    let num_images = read(&mut reader)?;
    let num_rows = read(&mut reader)?;
    let num_cols = read(&mut reader)?;

    for _ in 0..num_images {
        let mut image: Vec<Vec<f32>> = Vec::new();
        for _ in 0..num_rows {
            let mut row: Vec<f32> = Vec::new();
            for _ in 0..num_cols {
                let mut pixel = [0];
                reader.read_exact(&mut pixel)?;
                row.push(pixel[0] as f32 / 255.0);
            }
            image.push(row);
        }
        images.push(tensor::Tensor::triple(vec![image]).resize(tensor::Shape::Triple(1, 14, 14)));
    }

    Ok(images)
}

fn load_labels(file_path: &str, numbers: usize) -> Result<Vec<tensor::Tensor>> {
    let mut reader = BufReader::new(File::open(file_path)?);
    let _magic_number = read(&mut reader)?;
    let num_labels = read(&mut reader)?;

    let mut _labels = vec![0; num_labels as usize];
    reader.read_exact(&mut _labels)?;

    Ok(_labels
        .iter()
        .map(|&x| tensor::Tensor::one_hot(x as usize, numbers))
        .collect())
}

fn main() {
    let x_train = load_mnist("./examples/datasets/mnist-fashion/train-images-idx3-ubyte").unwrap();
    let class_train = load_labels(
        "./examples/datasets/mnist-fashion/train-labels-idx1-ubyte",
        10,
    )
    .unwrap();
    let x_test = load_mnist("./examples/datasets/mnist-fashion/t10k-images-idx3-ubyte").unwrap();
    let class_test = load_labels(
        "./examples/datasets/mnist-fashion/t10k-labels-idx1-ubyte",
        10,
    )
    .unwrap();

    let x_train: Vec<&tensor::Tensor> = x_train.iter().collect();
    let class_train: Vec<&tensor::Tensor> = class_train.iter().collect();
    let x_test: Vec<&tensor::Tensor> = x_test.iter().collect();
    let class_test: Vec<&tensor::Tensor> = class_test.iter().collect();

    println!("Train data {}x{}", x_train.len(), x_train[0].shape,);
    println!("Test data {}x{}\n", x_test.len(), x_test[0].shape,);

    // Create the results file.
    let mut file = File::create("./output/compare/fashion-2.json").unwrap();
    writeln!(file, "[").unwrap();
    writeln!(file, "  {{").unwrap();

    vec!["REGULAR", "FB1x2", "FB1x3", "FB2x2", "FB2x3"]
        .iter()
        .for_each(|method| {
            println!("Method: {}", method);
            vec![false, true].iter().for_each(|skip| {
                println!("  Skip: {}", skip);
                vec!["CLASSIFICATION"].iter().for_each(|problem| {
                    println!("   Problem: {}", problem);
                    writeln!(file, "    \"{}-{}-{}\": {{", method, skip, problem).unwrap();

                    for run in 1..RUNS + 1 {
                        println!("    Run: {}", run);
                        writeln!(file, "      \"run-{}\": {{", run).unwrap();

                        // Create the network based on the architecture.
                        let mut network: network::Network;
                        network = network::Network::new(tensor::Shape::Triple(1, 14, 14));

                        // Check if the method is regular or feedback.
                        if method == &"REGULAR" || method.contains(&"FB1") {
                            network.convolution(
                                1,
                                (3, 3),
                                (1, 1),
                                (0, 0),
                                (1, 1),
                                activation::Activation::ReLU,
                                None,
                            );
                            network.convolution(
                                1,
                                (3, 3),
                                (1, 1),
                                (0, 0),
                                (1, 1),
                                activation::Activation::ReLU,
                                None,
                            );
                            network.deconvolution(
                                1,
                                (3, 3),
                                (1, 1),
                                (0, 0),
                                activation::Activation::ReLU,
                                None,
                            );
                            network.deconvolution(
                                1,
                                (3, 3),
                                (1, 1),
                                (0, 0),
                                activation::Activation::ReLU,
                                None,
                            );
                            network.maxpool((2, 2), (2, 2));

                            // Add the feedback loop if applicable.
                            if method.contains(&"FB1") {
                                network.loopback(
                                    3,
                                    0,
                                    method.chars().last().unwrap().to_digit(10).unwrap() as usize
                                        - 1,
                                    Arc::new(|_loops| 1.0),
                                );
                            }
                        } else {
                            network.feedback(
                                vec![
                                    feedback::Layer::Convolution(
                                        1,
                                        activation::Activation::ReLU,
                                        (3, 3),
                                        (1, 1),
                                        (0, 0),
                                        (1, 1),
                                        None,
                                    ),
                                    feedback::Layer::Convolution(
                                        1,
                                        activation::Activation::ReLU,
                                        (3, 3),
                                        (1, 1),
                                        (0, 0),
                                        (1, 1),
                                        None,
                                    ),
                                    feedback::Layer::Deconvolution(
                                        1,
                                        activation::Activation::ReLU,
                                        (3, 3),
                                        (1, 1),
                                        (0, 0),
                                        None,
                                    ),
                                    feedback::Layer::Deconvolution(
                                        1,
                                        activation::Activation::ReLU,
                                        (3, 3),
                                        (1, 1),
                                        (0, 0),
                                        None,
                                    ),
                                ],
                                method.chars().last().unwrap().to_digit(10).unwrap() as usize,
                                false,
                                feedback::Accumulation::Mean,
                            );
                            network.maxpool((2, 2), (2, 2));
                        }

                        // Set the output layer based on the problem.
                        if problem == &"REGRESSION" {
                            network.dense(1, activation::Activation::Linear, false, None);
                            network.set_objective(objective::Objective::RMSE, None);
                        } else {
                            network.dense(10, activation::Activation::Softmax, true, None);
                            network.set_objective(objective::Objective::CrossEntropy, None);
                        }

                        // Add the skip connection if applicable.
                        if *skip {
                            network.connect(0, network.layers.len() - 2);
                        }

                        network
                            .set_optimizer(optimizer::Adam::create(0.001, 0.9, 0.999, 1e-8, None));

                        // Train the network
                        let (train_loss, val_loss, val_acc);
                        if problem == &"REGRESSION" {
                            unimplemented!("Regression not implemented.");
                        } else {
                            (train_loss, val_loss, val_acc) = network.learn(
                                &x_train,
                                &class_train,
                                Some((&x_test, &class_test, 10)),
                                32,
                                50,
                                None,
                            );
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
                                match &mut network.layers.get_mut(0).unwrap() {
                                    network::Layer::Feedback(fb) => {
                                        // Only keep the first four layers.
                                        fb.layers = fb.layers.drain(0..4).collect();
                                    }
                                    _ => panic!("Invalid layer."),
                                };
                            }

                            let (test_loss, test_acc);
                            if problem == &"REGRESSION" {
                                unimplemented!("Regression not implemented.");
                            } else {
                                (test_loss, test_acc) =
                                    network.validate(&x_test, &class_test, 1e-6);
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
                                unimplemented!("Regression not implemented.");
                            } else {
                                (test_loss, test_acc) =
                                    network.validate(&x_test, &class_test, 1e-6);
                            }

                            writeln!(file, "        }},").unwrap();
                            writeln!(file, "        \"no-skip\": {{").unwrap();
                            writeln!(file, "          \"tst-loss\": {},", test_loss).unwrap();
                            writeln!(file, "          \"tst-acc\": {}", test_acc).unwrap();
                        }
                        writeln!(file, "        }}").unwrap();

                        if run == RUNS {
                            writeln!(file, "      }}").unwrap();
                            if method == &"FB2x3" && *skip && problem == &"CLASSIFICATION" {
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
