// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, feedforward, objective, optimizer, plot, tensor};

use std::fs::File;
use std::io::{BufReader, Read, Result};

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
                row.push(pixel[0] as f32);
            }
            image.push(row);
        }
        images.push(tensor::Tensor::from(vec![image]));
    }

    Ok(images)
}

fn load_labels(file_path: &str, numbers: f32) -> Result<Vec<tensor::Tensor>> {
    let mut reader = BufReader::new(File::open(file_path)?);
    let _magic_number = read(&mut reader)?;
    let num_labels = read(&mut reader)?;

    let mut _labels = vec![0; num_labels as usize];
    reader.read_exact(&mut _labels)?;

    Ok(_labels
        .iter()
        .map(|&x| tensor::Tensor::one_hot(x as f32, numbers))
        .collect())
}

fn main() {
    let x_train = load_images("./datasets/mnist/train-images-idx3-ubyte").unwrap();
    let y_train = load_labels("./datasets/mnist/train-labels-idx1-ubyte", 10f32).unwrap();
    let x_test = load_images("./datasets/mnist/t10k-images-idx3-ubyte").unwrap();
    let y_test = load_labels("./datasets/mnist/t10k-labels-idx1-ubyte", 10f32).unwrap();
    println!(
        "Train: {} images, Test: {} images",
        x_train.len(),
        x_test.len()
    );

    let mut network = feedforward::Feedforward::new(tensor::Shape::Tensor(1, 28, 28));

    network.convolution(
        5,
        (3, 3),
        (1, 1),
        (1, 1),
        activation::Activation::ReLU,
        false,
        Some(0.1),
    );
    network.convolution(
        3,
        (3, 3),
        (1, 1),
        (0, 0),
        activation::Activation::ReLU,
        false,
        Some(0.1),
    );
    network.dense(10, activation::Activation::Softmax, true, Some(0.1));

    println!("{}", network);

    network.set_optimizer(optimizer::Optimizer::RMSprop(optimizer::RMSprop {
        learning_rate: 0.001,
        alpha: 0.0,
        epsilon: 1e-8,

        decay: Some(0.01),
        momentum: Some(0.01),
        centered: Some(true),

        velocity: vec![], // To be filled by the network
        gradient: vec![], // To be filled by the network
        buffer: vec![],   // To be filled by the network
    }));
    network.set_objective(
        objective::Objective::CrossEntropy, // Objective function
        Some((-1f32, 1f32)),                // Gradient clipping
    );

    // Train the network
    let _epoch_loss = network.learn(&x_train, &y_train, 10);

    // Validate the network
    let (val_acc, val_loss) = network.validate(&x_test, &y_test, 0.1);
    println!("1. Validation acc: {}, loss: {}", val_acc, val_loss);

    // Use the network
    let prediction = network.predict(x_test.get(0).unwrap());
    println!("2. Target: {}, Output: {}", y_test[0].data, prediction);

    let x = x_test.get(5).unwrap();
    let y = y_test.get(5).unwrap();
    plot::heatmap(&x, &y.data.to_string(), "input.png");
    let (pre, post) = network.forward(x);

    for (i, (i_pre, i_post)) in pre.iter().zip(post.iter()).enumerate() {
        let pre_title = format!("pre_layer_{}", i);
        let post_title = format!("post_layer_{}", i);
        let pre_file = format!("pre_layer_{}.png", i);
        let post_file = format!("post_layer_{}.png", i);
        plot::heatmap(&i_pre, &pre_title, &pre_file);
        plot::heatmap(&i_post, &post_title, &post_file);
    }
}
