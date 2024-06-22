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
    pub(crate) kernels: Vec<Vec<Vec<f32>>>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) activation: activation::Function,

    dropout: Option<f32>,
    stride: (usize, usize),
    padding: (usize, usize),

    pub training: bool,
}

impl std::fmt::Display for Convolution {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Convolution({}, channels: {}, kernel: {}x{}, stride: {:?}, padding: {:?}, bias: {})",
               self.activation, self.kernels.len(), self.kernels[0].len(), self.kernels[0][0].len(),
               self.stride, self.padding, self.bias.is_some())
    }
}

impl Convolution {

    /// Creates a new convolutional layer with random weights and bias.
    ///
    /// # Arguments
    ///
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
        channels: u16,
        activation: &activation::Activation,
        bias: bool,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dropout: Option<f32>,
    ) -> Self {
        let mut generator = random::Generator::create(12345);
        Convolution {
            kernels: (0..channels)
                .map(|_| (0..kernel.0)
                    .map(|_| (0..kernel.1)
                        .map(|_| 1.0) //generator.generate(-1.0, 1.0))
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
        }
    }

    /// Applies the forward pass (convolution) to the input vector.
    ///
    /// # Arguments
    ///
    /// * `x` - The input vector to the layer. (channels, height, width)
    ///
    /// # Returns
    ///
    /// The pre-activation and post-activation vectors of the layer. (channels, height, width)
    pub fn forward(&self, x: &Vec<Vec<Vec<f32>>>) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>) {

        // TODO: Padding.
        // TODO: Stride.

        let mut pre: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut post: Vec<Vec<Vec<f32>>> = Vec::new();
        for (filter, kernel) in self.kernels.iter().enumerate() {

            // Input height.
            let mut convolution_pre: Vec<Vec<f32>> = Vec::new();
            let mut convolution_post: Vec<Vec<f32>> = Vec::new();
            for height in 0..x[0].len() - kernel.len() + 1 {

                // Input width.
                let mut patches: Vec<f32> = Vec::new();
                for width in 0..x[0][0].len() - kernel[0].len() + 1 {

                    let mut patch = 0.0;
                    for i in 0..kernel.len() {
                        for j in 0..kernel[0].len() {
                            // Input channels.
                            for channel in 0..x.len() {
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

        (pre, post)
    }

    /// Applies the backward pass (backpropagation) to the convolutional layer.
    ///
    /// # Arguments
    ///
    /// * `x` - The input vector to the layer during the forward pass. (channels, height, width)
    /// * `pre` - The pre-activation output of the forward pass. (channels, height, width)
    /// * `post` - The post-activation output of the forward pass. (channels, height, width)
    /// * `dout` - The gradient of the loss with respect to the output of the layer. (channels, height, width)
    ///
    /// # Returns
    ///
    /// The gradient of the loss with respect to the input vector. (channels, height, width)
    pub fn backward(
        &self, x: &Vec<Vec<Vec<f32>>>, 
        pre: &Vec<Vec<Vec<f32>>>, post: &Vec<Vec<Vec<f32>>>, dout: &Vec<Vec<Vec<f32>>>
    ) -> Vec<Vec<Vec<f32>>> {
        let mut dx: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; x[0][0].len()]; x[0].len()]; x.len()];
        let mut dkernels: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; self.kernels[0][0].len()]; self.kernels[0].len()]; self.kernels.len()];
        let mut dbias: Vec<f32> = vec![0.0; self.kernels.len()];

        for (filter, kernel) in self.kernels.iter().enumerate() {
            for height in 0..x[0].len() - kernel.len() + 1 {
                for width in 0..x[0][0].len() - kernel[0].len() + 1 {
                    let dpatch = &dout[filter][height][width];
                    for i in 0..kernel.len() {
                        for j in 0..kernel[0].len() {
                            for channel in 0..x.len() {
                                dx[channel][height + i][width + j] += kernel[i][j] * dpatch;
                                dkernels[filter][i][j] += x[channel][height + i][width + j] * dpatch;
                            }
                        }
                    }
                    dbias[filter] += dpatch;
                }
            }
        }

        // Update weights and biases with gradients.
        for (filter, kernel) in self.kernels.iter_mut().enumerate() {
            for i in 0..kernel.len() {
                for j in 0..kernel[0].len() {
                    kernel[i][j] -= self.learning_rate * dkernels[filter][i][j];
                }
            }
        }
        if let Some(bias) = &mut self.bias {
            for i in 0..bias.len() {
                bias[i] -= self.learning_rate * dbias[i];
            }
        }

        dx
    }

    // fn convolve2d(a: &Vec<Vec<f32>>, kernel: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    //     let mut result = vec![
    //         vec![0.0; a[0].len() - kernel[0].len() + 1]; 
    //         a.len() - kernel.len() + 1
    //     ];
    //     for y in 0..result.len() {
    //         for x in 0..result[0].len() {
    //             for ky in 0..kernel.len() {
    //                 for kx in 0..kernel[0].len() {
    //                     result[y][x] += a[y + ky][x + kx] * kernel[ky][kx];
    //                 }
    //             }
    //         }
    //     }
    //     result
    // }
}
