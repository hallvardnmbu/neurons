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

use crate::random;
use crate::tensor;
use crate::activation;
use crate::algebra::*;

/// A dense layer in a neural network.
///
/// # Attributes
///
/// * `weights` - The weights of the layer.
/// * `bias` - The bias of the layer.
/// * `activation` - The activation function of the layer.
pub struct Convolution {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,

    pub(crate) kernels: Vec<Vec<Vec<f32>>>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) activation: activation::Function,

    dropout: Option<f32>,
    stride: (usize, usize),
    padding: (usize, usize),

    pub flatten_output: bool,
    pub training: bool,
}

impl std::fmt::Display for Convolution {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Convolution{}({} -> {}, kernel: {}x{}x{}, stride: {:?}, padding: {:?}, bias: {})",
               self.activation, self.inputs, self.outputs,
               self.kernels.len(), self.kernels[0].len(), self.kernels[0][0].len(),
               self.stride, self.padding, self.bias.is_some())
    }
}

impl Convolution {

    /// Calculates the output size of the convolutional layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The `tensor::Shape` of the input to the layer.
    /// * `channels` - The number of output channels from the layer (i.e., number of filters).
    /// * `kernel` - The size of each filter.
    /// * `stride` - The stride of the filter.
    /// * `padding` - The padding applied to the input before convolving.
    ///
    /// # Returns
    ///
    /// The shape of the output from the layer.
    fn calculate_output_size(
        input: &tensor::Shape,
        channels: &usize,
        kernel: &(usize, usize),
        stride: &(usize, usize),
        padding: &(usize, usize)
    ) -> tensor::Shape {
        let input: &(usize, usize, usize) = match input {
            tensor::Shape::Dense(shape) => {
                let root = (*shape as f32).sqrt() as usize;
                &(1, root, root)
            },
            tensor::Shape::Convolution(ch, he, wi) => &(*ch, *he, *wi),
        };

        let height = (input.1 + 2 * padding.0 - kernel.0) / stride.0 + 1;
        let width = (input.2 + 2 * padding.1 - kernel.1) / stride.1 + 1;

        tensor::Shape::Convolution(*channels, height, width)
    }

    /// Creates a new convolutional layer with random weights and bias.
    ///
    /// # Arguments
    ///
    /// * `input` - The shape of the input to the layer.
    /// * `channels` - The number of output channels from the layer (i.e., number of filters).
    /// * `activation` - The activation function of the layer.
    /// * `bias` - Whether the filters should have a bias.
    /// * `kernel` - The size of each filter.
    /// * `stride` - The stride of the filter.
    /// * `padding` - The padding applied to the input before convolving.
    /// * `dropout` - The dropout rate of the layer (when training).
    ///
    /// # Returns
    ///
    /// A new layer with random weights and bias with the given dimensions.
    pub fn create(
        input: tensor::Shape,
        channels: usize,
        activation: &activation::Activation,
        bias: bool,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dropout: Option<f32>,
    ) -> Self {
        let mut generator = random::Generator::create(12345);
        Convolution {
            inputs: input.clone(),
            outputs: Convolution::calculate_output_size(
                &input, &channels, &kernel, &stride, &padding
            ),

            kernels: (0..channels)
                .map(|_| (0..kernel.0)
                    .map(|_| (0..kernel.1)
                        .map(|_| generator.generate(-1.0, 1.0))
                        .collect())
                    .collect())
                .collect(),
            bias: match bias {
                true => Some(
                    (0..channels)
                        .map(|_| generator.generate(-1.0, 1.0))
                        .collect()),
                false => None,
            },
            activation: activation::Function::create(&activation),
            dropout,
            stride,
            padding,
            training: false,
            flatten_output: false,
        }
    }

    /// Applies the forward pass (convolution) to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor to the layer.
    ///
    /// # Returns
    ///
    /// The pre-activation and post-activation tensors of the layer.
    pub fn forward(&self, x: &tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {
        let (mut x, (mut channels, mut rows, mut columns)) = match x.shape {
            tensor::Shape::Dense(size) => {
                let root = (size as f32).sqrt() as usize;
                (x.reshape(tensor::Shape::Convolution(1, root, root)).get_data(),  (1, root, root))
            },
            tensor::Shape::Convolution(ch, he, wi) => {
                (x.get_data(), (ch, he, wi))
            },
        };

        // Padding.
        if self.padding.0 > 0 || self.padding.1 > 0 {
            let mut padded: Vec<Vec<Vec<f32>>> = vec![
                vec![vec![0.0; columns + 2 * self.padding.1]; rows + 2 * self.padding.0]; channels
            ];

            for i in 0..channels {
                for j in 0..rows {
                    for k in 0..columns {
                        padded[i][j + self.padding.0][k + self.padding.1] = x[i][j][k];
                    }
                }
            }

            rows += 2 * self.padding.0;
            columns += 2 * self.padding.1;
            x = padded;
        }

        let height = (rows - self.kernels[0].len()) / self.stride.0 + 1;
        let width = (columns - self.kernels[0][0].len()) / self.stride.1 + 1;
        let mut output: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; width]; height]; self.kernels.len()];

        for (filter, kernel) in self.kernels.iter().enumerate() {
            for i in 0..height {
                for j in 0..width {

                    let mut sum = 0.0;
                    for c in 0..channels {
                        for k in 0..kernel.len() {
                            for l in 0..kernel[0].len() {
                                sum += kernel[k][l]
                                    * x[c][i * self.stride.0 + k][j * self.stride.1 + l];
                            }
                        }
                    }
                    output[filter][i][j] = sum;

                    if let Some(bias) = &self.bias {
                        output[filter][i][j] += bias[filter];
                    }

                }
            }
        }

        let pre = tensor::Tensor::from(output);
        let mut post = self.activation.forward(&pre);

        // Apply dropout if the network is training.
        if self.training {
            if let Some(dropout) = self.dropout {
                post.dropout(dropout);
            }
        }

        if self.flatten_output {
            post = post.flatten();
        }

        (pre, post)
    }

    /// Applies the backward pass of the layer to the gradient vector.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient vector to the layer.
    /// * `input` - The input vector to the layer.
    /// * `output` - The output vector of the layer.
    ///
    /// # Returns
    ///
    /// The weight gradient, bias gradient, and input gradient vectors of the layer.
    // pub fn backward(
    //     &self, gradient: &mut tensor::Tensor, input: &tensor::Tensor, output: &tensor::Tensor
    // ) -> (tensor::Tensor, Option<tensor::Tensor>) {
    //
    //     // let gradient_data = gradient.get_data();
    //     // let input_data = input.get_data();
    //
    //     let derivative: Vec<f32> = self.activation.backward(output);
    //
    //     let mut kernel_gradient: Vec<Vec<Vec<f32>>> = tensor::Tensor::zeros(tensor::Shape::Convolution(
    //         self.kernels.len(), self.kernels[0].len(), self.kernels[0][0].len()
    //     )).get_data();
    //     let mut bias_gradient: Option<Vec<f32>> = match &self.bias {
    //         Some(_) => Some(vec![0.0; self.kernels.len()]),
    //         None => None,
    //     };
    //
    //     let (chi, hei, wii) = match input.shape {
    //         tensor::Shape::Dense(size) => {
    //             let root = (size as f32).sqrt() as usize;
    //             (1, root, root)
    //         },
    //         tensor::Shape::Convolution(ch, he, wi) => (ch, he, wi),
    //     };
    //     let (cho, heo, wio) = match output.shape {
    //         tensor::Shape::Dense(size) => {
    //             let root = (size as f32).sqrt() as usize;
    //             (1, root, root)
    //         },
    //         tensor::Shape::Convolution(ch, he, wi) => (ch, he, wi),
    //     };
    //
    //     // for l in 0..self.kernels.len() {
    //     //     for i in 0..input_data.len() {
    //     //         for j in 0..(input_data[0].len() - self.kernels[0].len() + 1) {
    //     //             for k in 0..(input_data[0][0].len() - self.kernels[0][0].len() + 1) {
    //     //                 for m in 0..self.kernels[0].len() {
    //     //                     for n in 0..self.kernels[0][0].len() {
    //     //                         kernel_gradient[l][m][n] += gradient_data[i][j][k] * input_data[i][j + m][k + n];
    //     //                         input_gradient[i][j + m][k + n] += gradient_data[i][j][k] * self.kernels[l][m][n];
    //     //                     }
    //     //                 }
    //     //             }
    //     //         }
    //     //     }
    //     // }
    //
    //     let bias_gradient = if let Some(_) = &self.bias {
    //         // Some(tensor::Tensor::from(gradient_data.iter().map(|plane| plane.iter().map(|row| row.iter().sum::<f32>()).sum::<f32>()).collect::<Vec<f32>>()))
    //         unimplemented!("Bias gradient for convolutional layers.")
    //     } else {
    //         None
    //     };
    //
    //     gradient.data = tensor::Data::Tensor(input_gradient);
    //
    //     (tensor::Tensor::from(kernel_gradient), bias_gradient)
    // }
    pub fn backward(
        &self, mut gradient: tensor::Tensor, input: &tensor::Tensor, output: &tensor::Tensor
    ) -> (tensor::Tensor, Option<tensor::Tensor>) {

        println!("G: {}", gradient.shape);
        println!("I: {}", input.shape);

        let gradient_data = gradient.get_data();
        let input = input.get_data();

        let mut kernel_gradient: Vec<Vec<Vec<f32>>> = vec![
            vec![vec![0.0; self.kernels[0][0].len()]; self.kernels[0].len()]; self.kernels.len()
        ];
        let mut input_gradient: Vec<Vec<Vec<f32>>> = vec![
            vec![vec![0.0; input[0][0].len()]; input[0].len()]; input.len()
        ];

        println!("{:?}", kernel_gradient);
        // println!("{:?}", input_gradient);
        println!("{:?}", self.kernels);

        for i in 0..input.len() {
            for j in 0..input[0].len() {
                for k in 0..input[0][0].len() {

                    for l in 0..self.kernels.len() {
                        for m in 0..self.kernels[0].len() {
                            for n in 0..self.kernels[0][0].len() {
                                kernel_gradient[l][m][n] += gradient_data[l][j][k] * input[i][j + m][k + n];
                                input_gradient[i][j + m][k + n] += gradient_data[l][j][k] * self.kernels[l][m][n];
                            }
                        }
                    }

                }
            }
        }

        let mut bias_gradient: Option<tensor::Tensor> = match &self.bias {
            Some(_) => Some(tensor::Tensor::from(kernel_gradient.clone())),
            None => None,
        };

        gradient.data = tensor::Data::Tensor(input_gradient);

        (tensor::Tensor::from(kernel_gradient),
         bias_gradient)
    }
}
