/*
Copyright 2024 Hallvard Høyland Lavik

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 */

use neurons::{activation, feedforward, objective, optimizer, tensor, plot};

use std::fs::File;
use std::io::{Read, BufReader, Result};

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

fn load_labels(file_path: &str) -> Result<Vec<tensor::Tensor>> {
    let mut reader = BufReader::new(File::open(file_path)?);
    let _magic_number = read(&mut reader)?;
    let num_labels = read(&mut reader)?;

    let mut _labels = vec![0; num_labels as usize];
    reader.read_exact(&mut _labels)?;

    Ok(_labels.iter().map(|&x| tensor::Tensor::from_single(vec![x as f32])).collect())
}

fn main() {
    let x_train = load_images("./datasets/mnist/t10k-images.idx3-ubyte").unwrap();
    let y_train = load_labels("./datasets/mnist/t10k-labels.idx1-ubyte").unwrap();
    let x_test = load_images("./datasets/mnist/train-images.idx3-ubyte").unwrap();
    let y_test = load_labels("./datasets/mnist/train-labels.idx1-ubyte").unwrap();

    let mut network = feedforward::Feedforward::new(tensor::Shape::Convolution(1, 28, 28));

    network.add_convolution(5, (5, 5), (1, 1), (1, 1),
                            activation::Activation::ReLU, false, Some(0.1));
    network.add_convolution(1, (5, 5), (1, 1), (1, 1),
                            activation::Activation::ReLU, false, Some(0.1));
    network.add_dense(1, activation::Activation::Linear, true, Some(0.1));

    println!("{}", network);

    plot::heatmap(&x_train[5], &y_train[5].data.to_string(), "input.png");

    network.set_optimizer(
        optimizer::Optimizer::RMSprop(
            optimizer::RMSprop {
                learning_rate: 0.001,
                alpha: 0.0,
                epsilon: 1e-8,

                decay: Some(0.01),
                momentum: Some(0.01),
                centered: Some(true),

                velocity: vec![],           // To be filled by the network
                gradient: vec![],           // To be filled by the network
                buffer: vec![],             // To be filled by the network
            }
        )
    );
    network.set_objective(
        objective::Objective::AE, // Objective function
        Some((-1f32, 1f32))                 // Gradient clipping
    );

    // Train the network
    let _epoch_loss = network.learn(&x_train, &y_train, 3);

    // Validate the network
    let (val_acc, val_loss) = network.validate(&x_test, &y_test, 0.1);
    println!("1. Validation acc: {}, loss: {}", val_acc, val_loss);

    // Use the network
    let prediction = network.predict(x_test.get(0).unwrap());
    println!("2. Input: {}, Target: {}, Output: {}", x_test[0].data, y_test[0].data, prediction);
}
