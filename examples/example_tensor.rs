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

fn main() {

    // ---------------------------------------------------------------------------------------------
    // Dense tensor (1D):

    let mut x = tensor::Tensor {
        shape: tensor::Shape::Dense(6),
        data: tensor::Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    };
    println!("Tensor: {}", x);

    let flat = x.flatten();
    println!("Flattened: {}", flat);

    let y = x.reshape(tensor::Shape::Convolution(1, 3, 2));
    println!("Reshaped: {}", y);

    // ---------------------------------------------------------------------------------------------
    // Convolutional tensor (3D):

    let mut x = tensor::Tensor {
        shape: tensor::Shape::Convolution(1, 3, 2),
        data: tensor::Data::Tensor(vec![
            vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![5.0, 6.0],
            ],
        ]),
    };
    println!("Tensor: {}", x);

    let flat = x.flatten();
    println!("Flattened: {}", flat);

    let y = x.reshape(tensor::Shape::Convolution(1, 6, 1));
    println!("Reshaped: {}", y);
}
