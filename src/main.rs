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

use neurons::network;
use neurons::activation::Activation;
use neurons::objective::Objective;
use neurons::optimizer::Optimizer;

fn main() {
    let nodes = vec![1, 3, 5, 1];
    let biases = vec![false, true, true];
    let activations = vec![Activation::Sigmoid, Activation::Linear, Activation::ReLU];
    let lr = 0.01f32;
    let optimizer = Optimizer::SGD;
    let objective = Objective::MSE;

    let mut net = network::Network::create(nodes, biases, activations, lr, optimizer, objective);

    println!("{}", net);

    let x = vec![1.0];
    let (_, _, out) = net.forward(&x);

    let y = vec![1000.0];
    let ((_, gradient), inters, outs, _) = net.loss(&y, &out);

    net.backward(gradient, inters, outs);
}
