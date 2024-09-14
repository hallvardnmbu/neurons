// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, feedback, network, objective, optimizer, tensor};

use std::fs::File;
use std::io::{BufReader, Read, Result};
use std::time;

fn read(reader: &mut dyn Read) -> Result<u32> {
    let mut buffer = [0; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

fn load_images(path: &str) -> Result<Vec<tensor::Tensor>> {
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
    let x_train = load_images("./examples/datasets/mnist/train-images-idx3-ubyte").unwrap();
    let y_train = load_labels("./examples/datasets/mnist/train-labels-idx1-ubyte", 10).unwrap();

    let x_train: Vec<&tensor::Tensor> = x_train.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y_train.iter().collect();

    let mut times: Vec<time::Duration> = Vec::new();

    for iteration in 0..10 {
        let start = time::Instant::now();

        // Create the network
        let mut network = network::Network::new(
            tensor::Shape::Triple(1, 14, 14),
            feedback::Accumulation::Add,
        );

        network.convolution(
            8,
            (3, 3),
            (1, 1),
            (0, 0),
            activation::Activation::ReLU,
            Some(0.05),
        );
        network.maxpool((2, 2), (2, 2));
        network.dense(10, activation::Activation::Softmax, true, None);

        network.set_optimizer(optimizer::Adam::create(
            0.001,      // Learning rate
            0.9,        // Beta1
            0.999,      // Beta2
            1e-8,       // Epsilon
            Some(0.01), // Decay
        ));
        network.set_objective(
            objective::Objective::CrossEntropy, // Objective function
            None,                               // Gradient clipping
        );

        // Train the network
        let (train_loss, _) = network.learn(&x_train, &y_train, None, 128, 10, None);

        println!("Iteration: {}, Loss: {:?}", iteration, train_loss);

        let duration = start.elapsed();
        times.push(duration);
    }

    let sum: time::Duration = times.iter().sum();
    let avg = sum / times.len() as u32;
    println!("Average time: {:?}", avg);
}
