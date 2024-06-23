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

use neurons::tensor;
use neurons::feedforward;
use neurons::activation;
use neurons::objective;
use neurons::optimizer;

fn main() {
    // Create the training data for the binary AND operation
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]
    ];
    let targets: Vec<Vec<f32>> = vec![
        vec![0.0], vec![0.0], vec![0.0], vec![1.0]
    ];

    // Create the network
    let mut network = feedforward::Feedforward::new(tensor::Shape::Dense(2));

    network.add_dense(10, activation::Activation::Linear, true, None);
    network.add_dense(1, activation::Activation::Sigmoid, false, None);

    network.set_optimizer(
        optimizer::Optimizer::SGD(
            optimizer::SGD {
                learning_rate: 0.1,
                decay: Some(0.01),
            }
        )
    );
    network.set_objective(objective::Objective::BinaryCrossEntropy, None);

    // Train the network
    let _epoch_loss = network.learn(&inputs, &targets, 250);

    // Validate the network
    let (val_acc, val_loss) = network.validate(&inputs, &targets, 0.1);
    println!("1. Validation acc: {} loss: {}", val_acc, val_loss);

    // Use the network
    let prediction = network.predict(inputs.get(0).unwrap());
    println!("2. Input: {:?}, Target: {:?}, Output: {:?}", inputs[0], targets[0], prediction);

    // Use the network on batch
    let predictions = network.predict_batch(&inputs);
    println!("3. Input: {:?},\n   Target: {:?},\n   Output: {:?}", inputs, targets, predictions);
}