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

extern crate csv;
extern crate rand;

use rand::prelude::SliceRandom;

use neurons::network;
use neurons::activation::Activation;
use neurons::objective::Objective;
use neurons::optimizer::Optimizer;

fn data(path: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut reader = csv::Reader::from_path(path).unwrap();

    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<Vec<f32>> = Vec::new();

    reader.records().for_each(|record| {
        let record = record.unwrap();
        x.push(vec![
            record.get(1).unwrap().parse::<f32>().unwrap(),
            record.get(2).unwrap().parse::<f32>().unwrap(),
            record.get(3).unwrap().parse::<f32>().unwrap(),
            record.get(4).unwrap().parse::<f32>().unwrap(),
        ]);
        y.push(
            match record.get(5).unwrap() {
                "Iris-setosa" => vec![1.0, 0.0, 0.0],
                "Iris-versicolor" => vec![0.0, 1.0, 0.0],
                "Iris-virginica" => vec![0.0, 0.0, 1.0],
                _ => panic!("Unknown class"),
            }
        );
    });

    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.shuffle(&mut rng);

    let x: Vec<Vec<f32>> = indices.iter().map(|&i| x[i].clone()).collect();
    let y: Vec<Vec<f32>> = indices.iter().map(|&i| y[i].clone()).collect();

    let split = (x.len() as f32 * 0.8) as usize;
    let x = x.split_at(split);
    let y = y.split_at(split);

    let x_train = x.0.to_vec();
    let y_train = y.0.to_vec();
    let x_test = x.1.to_vec();
    let y_test = y.1.to_vec();

    (x_train, y_train, x_test, y_test)
}

fn main() {
    // Load the iris dataset
    let (x_train, y_train, x_test, y_test) = data("./datasets/iris.csv");
    println!("Train data {}x{}: {:?} => {:?}",
             x_train[0].len(), x_train.len(), x_train[0], y_train[0]);
    println!("Test data {}x{}: {:?} => {:?}",
             x_test[0].len(), x_test.len(), x_test[0], y_test[0]);

    // Create the network
    let mut network = network::Network::new();

    network.add_layer(4, 5, Activation::ReLU, true);
    network.add_layer(5, 3, Activation::Softmax, false);

    network.set_optimizer(Optimizer::SGD, 0.5);
    network.set_objective(Objective::RMSE);

    // Train the network
    let _epoch_loss = network.train(&x_train, &y_train, 1000);

    // Validate the network
    let val_loss = network.validate(&x_test, &y_test);
    println!("1. Validation loss: {:?}", val_loss);

    // Use the network
    let prediction = network.predict(x_test.get(0).unwrap());
    println!("2. Input: {:?}, Target: {:?}, Output: {:?}", x_test[0], y_test[0], prediction);

    // // Use the network on batch
    // let predictions = net.predict_batch(&x_test);
    // println!("3. Input: {:?},\n   Target: {:?},\n   Output: {:?}",
    //          x_test[..5], y_test[..5],  predictions[..5]);
}