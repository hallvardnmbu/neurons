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

        let (x, (channels, rows, columns)) = match x.shape {
            tensor::Shape::Dense(size) => {
                let root = (size as f32).sqrt() as usize;
                (x.reshape(tensor::Shape::Convolution(1, root, root)).get_data(),  (1, root, root))
            },
            tensor::Shape::Convolution(ch, he, wi) => {
                (x.get_data(), (ch, he, wi))
            },
        };

        // TODO: Padding.
        // TODO: Stride.

        let mut pre: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut post: Vec<Vec<Vec<f32>>> = Vec::new();
        for (filter, kernel) in self.kernels.iter().enumerate() {

            // Input height.
            let mut convolution_pre: Vec<Vec<f32>> = Vec::new();
            let mut convolution_post: Vec<Vec<f32>> = Vec::new();
            for height in 0..rows - kernel.len() + 1 {

                // Input width.
                let mut patches: Vec<f32> = Vec::new();
                for width in 0..columns - kernel[0].len() + 1 {

                    let mut patch = 0.0;
                    for i in 0..kernel.len() {
                        for j in 0..kernel[0].len() {
                            // Input channels.
                            for channel in 0..channels {
                                patch += x[channel][height + i][width + j] * kernel[i][j];
                            }
                        }
                    }
                    patches.push(patch);
                }
                let mut activated = self.activation.forward(&patches);
                convolution_pre.push(patches);
                if let Some(bias) = &self.bias {
                    for j in 0..activated.len() {
                        activated[j] += bias[filter];
                    }
                }

                // TODO: Is this correct dropout implementation for convolutional layers?
                // Apply dropout if the network is training.
                if self.training {
                    if let Some(dropout) = self.dropout {
                        let mut generator = random::Generator::create(12345);
                        let mask: Vec<f32> = (0..activated.len())
                            .map(|_| if generator.generate(0.0, 1.0) < dropout { 0.0 } else { 1.0 })
                            .collect();
                        mul_inplace(&mut activated, &mask);
                    }
                }
                convolution_post.push(activated);
            }
            pre.push(convolution_pre);
            post.push(convolution_post);
        }

        if self.flatten_output {
            let mut flattened: Vec<Vec<f32>> = Vec::new();
            for i in 0..post[0].len() {
                for j in 0..post[0][0].len() {
                    let mut channel: Vec<f32> = Vec::new();
                    for k in 0..post.len() {
                        channel.push(post[k][i][j]);
                    }
                    flattened.push(channel);
                }
            }
            post = vec![flattened];
        }

        (tensor::Tensor::from(pre), tensor::Tensor::from(post))
    }

    // /// Applies the backward pass of the layer to the gradient vector.
    // ///
    // /// # Arguments
    // ///
    // /// * `gradient` - The gradient vector to the layer.
    // /// * `input` - The input vector to the layer.
    // /// * `output` - The output vector of the layer.
    // ///
    // /// # Returns
    // ///
    // /// The weight gradient, bias gradient, and input gradient vectors of the layer.
    // pub fn backward(
    //     &self, gradient: &Vec<f32>, input: &Vec<f32>, output: &Vec<f32>
    // ) -> (Vec<Vec<f32>>, Option<Vec<f32>>, Vec<f32>) {
    //     let derivative: Vec<f32> = self.activation.backward(output);
    //     let delta: Vec<f32> = mul(gradient, &derivative);
    //
    //     let weight_gradient: Vec<Vec<f32>> = delta
    //         .iter().map(|d: &f32| input
    //         .iter().map(|i: &f32| i * d)
    //         .collect())
    //         .collect();
    //     let bias_gradient: Option<Vec<f32>> = match self.bias {
    //         Some(_) => Some(delta.clone()),
    //         None => None,
    //     };
    //     let input_gradient: Vec<f32> = (0..input.len())
    //         .map(|i: usize| delta
    //             .iter().zip(self.weights.iter())
    //             .map(|(d, w)| d * w[i])
    //             .sum::<f32>())
    //         .collect();
    //
    //     (weight_gradient, bias_gradient, input_gradient)
    // }
}
