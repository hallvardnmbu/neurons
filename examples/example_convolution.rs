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

    let mut network = feedforward::Feedforward::new(tensor::Shape::Convolution(1, 28, 28));

    network.add_convolution(5, (5, 5), (1, 1), (1, 1),
                            activation::Activation::ReLU, false, Some(0.1));
    network.add_convolution(1, (5, 5), (1, 1), (1, 1),
                            activation::Activation::ReLU, false, Some(0.1));
    network.add_dense(2, activation::Activation::Softmax, true, Some(0.1));

    println!("{}", network);

    let x: Vec<Vec<Vec<f32>>> = vec![vec![vec![1.0; 28]; 28]; 3];
    println!("x: {}x{}x{}", x.len(), x[0].len(), x[0][0].len());

    let (pre, post) = network.forward(&x);
    println!("pre-activation: {}x{}x{}", pre.len(), pre[0].len(), pre[0][0].len());
    println!("post-activation: {}x{}x{}", post.len(), post[0].len(), post[0][0].len());
}
