// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, objective, optimizer, plot, tensor};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Result};

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
    let _labels: HashMap<u8, &str> = [
        (0, "top"),
        (1, "trouser"),
        (2, "pullover"),
        (3, "dress"),
        (4, "coat"),
        (5, "sandal"),
        (6, "shirt"),
        (7, "sneaker"),
        (8, "bag"),
        (9, "ankle boot"),
    ]
    .iter()
    .cloned()
    .collect();
    let x_train = load_mnist("./examples/datasets/mnist-fashion/train-images-idx3-ubyte").unwrap();
    let y_train = load_labels(
        "./examples/datasets/mnist-fashion/train-labels-idx1-ubyte",
        10,
    )
    .unwrap();
    let x_test = load_mnist("./examples/datasets/mnist-fashion/t10k-images-idx3-ubyte").unwrap();
    let y_test = load_labels(
        "./examples/datasets/mnist-fashion/t10k-labels-idx1-ubyte",
        10,
    )
    .unwrap();
    println!(
        "Train: {} images, Test: {} images",
        x_train.len(),
        x_test.len()
    );

    let x_train: Vec<&tensor::Tensor> = x_train.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y_train.iter().collect();
    let x_test: Vec<&tensor::Tensor> = x_test.iter().collect();
    let y_test: Vec<&tensor::Tensor> = y_test.iter().collect();

    let mut network = network::Network::new(tensor::Shape::Triple(1, 14, 14));

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
    network.dense(10, activation::Activation::Softmax, true, None);

    network.set_optimizer(optimizer::Adam::create(0.001, 0.9, 0.999, 1e-8, None));
    network.set_objective(objective::Objective::CrossEntropy, None);

    println!("{}", network);

    // Train the network
    let (train_loss, val_loss, val_acc) = network.learn(
        &x_train,
        &y_train,
        Some((&x_test, &y_test, 10)),
        32,
        25,
        Some(5),
    );
    plot::loss(
        &train_loss,
        &val_loss,
        &val_acc,
        "PLAIN : Fashion-MNIST",
        "./output/mnist-fashion/plain.png",
    );

    // Validate the network
    let (val_loss, val_acc) = network.validate(&x_test, &y_test, 1e-6);
    println!(
        "Final validation accuracy: {:.2} % and loss: {:.5}",
        val_acc * 100.0,
        val_loss
    );

    // Use the network
    let prediction = network.predict(x_test.get(0).unwrap());
    println!(
        "Prediction on input: Target: {}. Output: {}.",
        y_test[0].argmax(),
        prediction.argmax()
    );

    // let x = x_test.get(5).unwrap();
    // let y = y_test.get(5).unwrap();
    // plot::heatmap(
    //     &x,
    //     &format!("{}", &labels[&(y.argmax() as u8)]),
    //     "./output/mnist-fashion/input.png",
    // );

    // Plot the pre- and post-activation heatmaps for each (image) layer.
    // let (pre, post, _) = network.forward(x);
    // for (i, (i_pre, i_post)) in pre.iter().zip(post.iter()).enumerate() {
    //     let pre_title = format!("layer_{}_pre", i);
    //     let post_title = format!("layer_{}_post", i);
    //     let pre_file = format!("layer_{}_pre.png", i);
    //     let post_file = format!("layer_{}_post.png", i);
    //     plot::heatmap(&i_pre, &pre_title, &pre_file);
    //     plot::heatmap(&i_post, &post_title, &post_file);
    // }
}
