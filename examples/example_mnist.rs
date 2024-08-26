// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, objective, optimizer, plot, tensor};

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
                row.push(pixel[0] as f32 / 255.0);
            }
            image.push(row);
        }
        images.push(tensor::Tensor::from(vec![image]).resize(tensor::Shape::Tensor(1, 14, 14)));
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
    let x_train = load_images("./examples/datasets/mnist/train-images-idx3-ubyte").unwrap();
    let y_train = load_labels("./examples/datasets/mnist/train-labels-idx1-ubyte", 10f32).unwrap();
    let x_test = load_images("./examples/datasets/mnist/t10k-images-idx3-ubyte").unwrap();
    let y_test = load_labels("./examples/datasets/mnist/t10k-labels-idx1-ubyte", 10f32).unwrap();
    println!(
        "Train: {} images, Test: {} images",
        x_train.len(),
        x_test.len()
    );

    let x_train: Vec<&tensor::Tensor> = x_train.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y_train.iter().collect();
    let x_test: Vec<&tensor::Tensor> = x_test.iter().collect();
    let y_test: Vec<&tensor::Tensor> = y_test.iter().collect();

    let mut network = network::Network::new(tensor::Shape::Tensor(1, 14, 14));

    network.convolution(
        8,
        (3, 3),
        (1, 1),
        (0, 0),
        activation::Activation::ReLU,
        None,
    );
    network.maxpool((2, 2), (2, 2));
    network.convolution(
        8,
        (3, 3),
        (1, 1),
        (0, 0),
        activation::Activation::ReLU,
        None,
    );
    network.dense(128, activation::Activation::ReLU, true, Some(0.25));
    network.dense(10, activation::Activation::Softmax, true, None);

    network.set_optimizer(optimizer::Optimizer::Adam(optimizer::Adam {
        learning_rate: 0.001,
        decay: None,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        velocity: vec![],
        momentum: vec![],
    }));
    network.set_objective(
        objective::Objective::CrossEntropy, // Objective function
        None,                               // Gradient clipping
    );

    println!("{}", network);

    // Train the network
    let (train_loss, val_loss) = network.learn(&x_train, &y_train, 0.1, 128, 50, Some(10));
    plot::loss(&train_loss, &val_loss, "Loss per epoch", "loss.png");

    // Validate the network
    let (val_loss, val_acc) = network.validate(&x_test, &y_test, 0.1);
    println!("1. Validation acc: {}, loss: {}", val_acc, val_loss);

    // Use the network
    let prediction = network.predict(x_test.get(0).unwrap());
    println!(
        "2. Target: {}, Output: {}",
        y_test[0].argmax(),
        prediction.argmax()
    );

    let x = x_test.get(5).unwrap();
    let y = y_test.get(5).unwrap();
    plot::heatmap(&x, &format!("Target: {}", y.argmax()), "input.png");

    let (pre, post, _) = network.forward(x);
    for (i, (i_pre, i_post)) in pre.iter().zip(post.iter()).enumerate() {
        let pre_title = format!("layer_{}_pre", i);
        let post_title = format!("layer_{}_post", i);
        let pre_file = format!("layer_{}_pre.png", i);
        let post_file = format!("layer_{}_post.png", i);
        plot::heatmap(&i_pre, &pre_title, &pre_file);
        plot::heatmap(&i_post, &post_title, &post_file);
    }
}
