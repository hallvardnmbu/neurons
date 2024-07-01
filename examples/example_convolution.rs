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
use neurons::plot;

fn main() {
    let mut network = feedforward::Feedforward::new(tensor::Shape::Tensor(1, 24, 24));

    network.convolution(1, (3, 3), (1, 1), (0, 0),
                        activation::Activation::ReLU, false, Some(0.1));
    network.convolution(1, (3, 3), (1, 1), (0, 0),
                        activation::Activation::ReLU, false, Some(0.1));

    println!("{}", network);

    let x = tensor::Tensor::random(tensor::Shape::Tensor(1, 24, 24), 0.0, 1.0);
    println!("x: {}", &x.shape);

    let (pre, post) = network.forward(&x);
    println!("pre-activation: {}", &pre[pre.len() - 1].shape);
    println!("post-activation: {}", &post[post.len() - 1].shape);

    plot::heatmap(&x, "Input", "input.png");
    plot::heatmap(&pre[pre.len() - 1], "Pre-activation", "pre.png");
    plot::heatmap(&post[post.len() - 1], "Post-activation", "post.png");
}
