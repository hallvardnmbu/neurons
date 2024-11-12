// Copyright (C) 2024 Hallvard Høyland Lavik
//
// Code for comparison between the various architectures and their time differences.

use neurons::{activation, feedback, network, objective, optimizer, tensor};

use std::{
    fs::File,
    io::{BufReader, Read, Result, Write},
    sync::Arc,
    thread::panicking,
    time,
};

const RUNS: usize = 5;
const EPOCHS: i32 = 1;

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
        images.push(tensor::Tensor::triple(vec![image]));
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
    let mut x = load_mnist("./examples/datasets/mnist/train-images-idx3-ubyte").unwrap();
    let mut y = load_labels("./examples/datasets/mnist/train-labels-idx1-ubyte", 10).unwrap();
    let x_test = load_mnist("./examples/datasets/mnist/t10k-images-idx3-ubyte").unwrap();
    let class_test = load_labels("./examples/datasets/mnist/t10k-labels-idx1-ubyte", 10).unwrap();

    x.extend(x_test);
    y.extend(class_test);

    let x: Vec<&tensor::Tensor> = x.iter().collect();
    let y: Vec<&tensor::Tensor> = y.iter().collect();

    // Create the results file.
    let mut file = File::create("./output/timing/mnist.json").unwrap();
    writeln!(file, "[").unwrap();
    writeln!(file, "  {{").unwrap();

    vec!["REGULAR", "FB1", "FB2x2", "FB2x3"]
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
                        network = network::Network::new(tensor::Shape::Triple(1, 28, 28));
                        network.convolution(
                            1,
                            (3, 3),
                            (1, 1),
                            (1, 1),
                            (1, 1),
                            activation::Activation::ReLU,
                            None,
                        );

                        // Check if the method is regular or feedback.
                        if method == &"REGULAR" || method == &"FB1" {
                            network.convolution(
                                1,
                                (3, 3),
                                (1, 1),
                                (1, 1),
                                (1, 1),
                                activation::Activation::ReLU,
                                None,
                            );
                            network.convolution(
                                1,
                                (3, 3),
                                (1, 1),
                                (1, 1),
                                (1, 1),
                                activation::Activation::ReLU,
                                None,
                            );
                            network.maxpool((2, 2), (2, 2));

                            // Add the feedback loop if applicable.
                            if method == &"FB1" {
                                network.loopback(2, 0, Arc::new(|_loops| 1.0));
                            }
                        } else {
                            network.feedback(
                                vec![feedback::Layer::Convolution(
                                    1,
                                    activation::Activation::ReLU,
                                    (3, 3),
                                    (1, 1),
                                    (1, 1),
                                    (1, 1),
                                    None,
                                )],
                                method.chars().last().unwrap().to_digit(10).unwrap() as usize,
                                false,
                                feedback::Accumulation::Mean,
                            );
                            network.convolution(
                                1,
                                (3, 3),
                                (1, 1),
                                (1, 1),
                                (1, 1),
                                activation::Activation::ReLU,
                                None,
                            );
                            network.maxpool((2, 2), (2, 2));
                        }

                        // Set the output layer based on the problem.
                        if problem == &"REGRESSION" {
                            panic!("Invalid problem type.");
                        } else {
                            network.dense(10, activation::Activation::Softmax, true, None);
                            network.set_objective(objective::Objective::CrossEntropy, None);
                        }

                        // Add the skip connection if applicable.
                        if *skip {
                            network.connect(1, network.layers.len() - 2);
                        }

                        network
                            .set_optimizer(optimizer::Adam::create(0.001, 0.9, 0.999, 1e-8, None));

                        let start = time::Instant::now();

                        // Train the network
                        if problem == &"REGRESSION" {
                            panic!("Invalid problem type.");
                        } else {
                            (_, _, _) = network.learn(&x, &y, None, 32, EPOCHS, None);
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

                    if method == &"FB2x3" && *skip && problem == &"CLASSIFICATION" {
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
