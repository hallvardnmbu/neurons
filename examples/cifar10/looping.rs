// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, objective, optimizer, plot, random, tensor};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::sync::Arc;
use std::time;

const IMAGE_SIZE: usize = 32;
const NUM_CHANNELS: usize = 3;
const IMAGE_BYTES: usize = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS;

pub fn load_cifar10(file_path: &str) -> (Vec<tensor::Tensor>, Vec<tensor::Tensor>) {
    let file = File::open(file_path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0u8; 1 + IMAGE_BYTES];

    let mut labels = Vec::new();
    let mut images = Vec::new();

    while reader.read_exact(&mut buffer).is_ok() {
        let label = buffer[0];
        let mut image = vec![vec![vec![0.0f32; IMAGE_SIZE]; IMAGE_SIZE]; NUM_CHANNELS];

        for channel in 0..NUM_CHANNELS {
            for row in 0..IMAGE_SIZE {
                for col in 0..IMAGE_SIZE {
                    let index = 1 + channel * IMAGE_SIZE * IMAGE_SIZE + row * IMAGE_SIZE + col;
                    image[channel][row][col] = buffer[index] as f32 / 255.0;
                }
            }
        }

        labels.push(tensor::Tensor::one_hot(label as usize, 10));
        images.push(tensor::Tensor::triple(image));
    }

    (images, labels)
}

pub fn shuffle(
    x: Vec<tensor::Tensor>,
    y: Vec<tensor::Tensor>,
) -> (Vec<tensor::Tensor>, Vec<tensor::Tensor>) {
    let mut generator = random::Generator::create(
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .subsec_micros() as u64,
    );

    let mut indices: Vec<usize> = (0..y.len()).collect();
    generator.shuffle(&mut indices);

    let a: Vec<tensor::Tensor> = indices.iter().map(|&i| x[i].clone()).collect();
    let b: Vec<tensor::Tensor> = indices.iter().map(|&i| y[i].clone()).collect();

    (a, b)
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

    // Shuffle the data.
    // let (x_train, y_train) = shuffle(x_train, y_train);
    // let (x_test, y_test) = shuffle(x_test, y_test);

    let x_train: Vec<&tensor::Tensor> = x_train.iter().collect();
    let y_train: Vec<&tensor::Tensor> = y_train.iter().collect();
    let x_test: Vec<&tensor::Tensor> = x_test.iter().collect();
    let y_test: Vec<&tensor::Tensor> = y_test.iter().collect();

    // plot::heatmap(
    //     &x_train[0],
    //     &format!("{}", &labels[&(y_train[0].argmax() as u8)]),
    //     "./output/cifar/input.png",
    // );

    let mut network = network::Network::new(tensor::Shape::Triple(3, 32, 32));

    network.convolution(
        32,
        (3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        activation::Activation::ReLU,
        None,
    );
    network.convolution(
        32,
        (3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        activation::Activation::ReLU,
        None,
    );
    network.maxpool((2, 2), (2, 2));
    network.convolution(
        32,
        (3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        activation::Activation::ReLU,
        None,
    );
    network.convolution(
        32,
        (3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        activation::Activation::ReLU,
        None,
    );
    network.maxpool((2, 2), (2, 2));
    network.dense(512, activation::Activation::ReLU, true, None);
    network.dense(10, activation::Activation::Softmax, true, None);

    network.set_optimizer(optimizer::Adam::create(0.001, 0.9, 0.999, 1e-8, None));
    network.set_objective(objective::Objective::CrossEntropy, None);

    network.loopback(1, 1, 1, Arc::new(|_loops| 1.0), false);
    network.loopback(4, 3, 1, Arc::new(|_loops| 1.0), false);

    println!("{}", network);

    // Train the network
    let (train_loss, val_loss, val_acc) = network.learn(
        &x_train,
        &y_train,
        Some((&x_test, &y_test, 5)),
        128,
        50,
        Some(1),
    );
    plot::loss(
        &train_loss,
        &val_loss,
        &val_acc,
        "LOOP : CIFAR-10",
        "./output/cifar/loop.png",
    );

    // Store the training metrics
    let mut writer = File::create("./output/cifar/loop.csv").unwrap();
    writer.write_all(b"train_loss,val_loss,val_acc\n").unwrap();
    for i in 0..train_loss.len() {
        writer
            .write_all(format!("{},{},{}\n", train_loss[i], val_loss[i], val_acc[i]).as_bytes())
            .unwrap();
    }

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
        "Prediction Target: {}. Output: {}.",
        y_test[0].argmax(),
        prediction.argmax()
    );
}
