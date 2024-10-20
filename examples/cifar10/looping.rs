// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, feedback, network, objective, optimizer, plot, random, tensor};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::Arc;

pub fn load_cifar10(file_path: &str) -> (Vec<tensor::Tensor>, Vec<tensor::Tensor>) {
    let file = File::open(file_path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0u8; 1 + 3072];

    let mut labels = Vec::new();
    let mut images = Vec::new();

    while reader.read_exact(&mut buffer).is_ok() {
        let label = buffer[0];
        let mut image = vec![vec![vec![0.0f32; 32]; 32]; 3];

        for channel in 0..3 {
            for row in 0..32 {
                for col in 0..32 {
                    let index = 1 + channel * 1024 + row * 32 + col;
                    image[channel][row][col] = buffer[index] as f32 / 255.0;
                }
            }
        }

        labels.push(label as usize);
        images.push(tensor::Tensor::triple(image));
    }

    let mut generator = random::Generator::create(12345);
    let mut indices: Vec<usize> = (0..labels.len()).collect();
    generator.shuffle(&mut indices);

    let images: Vec<tensor::Tensor> = indices.iter().map(|&i| images[i].clone()).collect();
    let labels: Vec<tensor::Tensor> = indices
        .iter()
        .map(|&i| tensor::Tensor::one_hot(labels[i], 10))
        .collect();

    (images, labels)
}

fn main() {
    let _labels: HashMap<u8, &str> = [
        (0, "airplane"),
        (1, "automobile"),
        (2, "bird"),
        (3, "cat"),
        (4, "deer"),
        (5, "dog"),
        (6, "frog"),
        (7, "horse"),
        (8, "ship"),
        (9, "truck"),
    ]
    .iter()
    .cloned()
    .collect();
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    for i in 1..6 {
        let (x_batch, y_batch) =
            load_cifar10(&format!("./examples/datasets/cifar10/data_batch_{}.bin", i));
        x_train.extend(x_batch);
        y_train.extend(y_batch);
    }
    let (x_test, y_test) = load_cifar10("./examples/datasets/cifar10/test_batch.bin");
    println!(
        "Train: {} images, Test: {} images",
        x_train.len(),
        x_test.len()
    );

    let x_train: Vec<&tensor::Tensor> = x_train.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y_train.iter().collect();
    let x_test: Vec<&tensor::Tensor> = x_test.iter().collect();
    let y_test: Vec<&tensor::Tensor> = y_test.iter().collect();

    let mut network = network::Network::new(tensor::Shape::Triple(3, 32, 32));

    network.convolution(
        32,
        (5, 5),
        (1, 1),
        (1, 1),
        (1, 1),
        activation::Activation::ReLU,
        None,
    );
    for _ in 0..3 {
        network.convolution(
            32,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            activation::Activation::ReLU,
            None,
        );
    }
    network.maxpool((2, 2), (2, 2));
    network.dense(128, activation::Activation::ReLU, true, None);
    network.dense(10, activation::Activation::Softmax, true, None);

    network.loopback(
        3,                             // From layer X's output.
        1,                             // To layer Y's input.
        Arc::new(|loops| 1.0 / loops), // Gradient scaling.
    );
    network.set_accumulation(
        feedback::Accumulation::Add, // How the skip-connection is accumulated.
        feedback::Accumulation::Add, // How the pre- and post-activations are accumulated.
    );

    network.set_optimizer(optimizer::Adam::create(0.001, 0.9, 0.999, 1e-8, None));
    network.set_objective(
        objective::Objective::CrossEntropy, // Objective function
        None,                               // Gradient clipping
    );

    println!("{}", network);

    // Train the network
    let (train_loss, val_loss, val_acc) = network.learn(
        &x_train,
        &y_train,
        Some((&x_test, &y_test, 10)),
        128,
        25,
        Some(5),
    );
    plot::loss(
        &train_loss,
        &val_loss,
        &val_acc,
        "LOOP : CIFAR-10",
        "./output/cifar/looping.png",
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
    //     &format!("Target: {}", y.argmax()),
    //     "./output/cifar/input.png",
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
