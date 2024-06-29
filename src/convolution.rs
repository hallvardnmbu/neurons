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

    pub kernels: Vec<tensor::Tensor>,
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
        write!(f, "Convolution{}({} -> {}, kernel: {}x({}), stride: {:?}, padding: {:?}, bias: {})",
               self.activation, self.inputs, self.outputs,
               self.kernels.len(), self.kernels[0].shape,
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
            _ => unimplemented!("Expected a dense or convolutional input shape.")
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

            kernels: {
                let in_channels = match input {
                    tensor::Shape::Dense(size) => size,
                    tensor::Shape::Convolution(ch, _, _) => ch,
                    _ => unimplemented!("Expected a dense or convolutional input shape.")
                };
                (0..channels).map(|_|
                tensor::Tensor::from(
                    (0..in_channels)
                        .map(|_| (0..kernel.0)
                            .map(|_| (0..kernel.1)
                                .map(|_| generator.generate(-1.0, 1.0))
                                .collect())
                            .collect())
                        .collect())
                ).collect()
            },
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

    /// Applies the forward pass (convolution) to the input Tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input Tensor to the layer.
    ///
    /// # Returns
    ///
    /// The pre-activation and post-activation Tensors of the layer.
    pub fn forward(&self, x: &tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {

        // Extracting the input data and dimensions.
        let (mut x, ic, ih, iw) = match &x.data {
            tensor::Data::Vector(vector) => {
                let root = (vector.len() as f32).sqrt() as usize;
                (vec![vector.chunks(root).map(|v| v.to_vec()).collect()], 1, root, root)
            },
            tensor::Data::Tensor(tensor) => {
                (tensor.clone(), tensor.len(), tensor[0].len(), tensor[0][0].len())
            },
            _ => panic!("Expected `Vector` or `Tensor` input data.")
        };

        // Padding the input wrt. `self.padding`.
        let ph = ih + 2 * self.padding.0;
        let pw = iw + 2 * self.padding.1;
        let mut padded = vec![vec![vec![0.0; pw]; ph]; ic];
        for c in 0..ic {
            for h in 0..ih {
                for w in 0..iw {
                    padded[c][h + self.padding.0][w + self.padding.1] = x[c][h][w];
                }
            }
        }
        x = padded;

        // Extracting the `self.kernels` dimensions.
        let oc = self.kernels.len();
        let (kc, kh, kw) = match self.kernels[0].shape {
            tensor::Shape::Convolution(c, h, w) => (c, h, w),
            _ => panic!("The kernel should be a `Convolution`-shaped Tensor."),
        };
        assert_eq!(ic, kc, "The number of input channels should match the kernel channels.");

        // Defining the output dimensions and Tensor.
        let oh = (ph - kh) / self.stride.0 + 1;
        let ow = (pw - kw) / self.stride.1 + 1;
        let mut y = vec![vec![vec![0.0; ow]; oh]; oc];

        // Convolving the input with the kernels.
        for filter in 0..oc {
            let kernel = match &self.kernels[filter].data {
                tensor::Data::Tensor(kernel) => kernel,
                _ => panic!("The kernels should be `Tensor`s."),
            };

            for height in 0..oh {
                for width in 0..ow {

                    let mut sum = 0.0;
                    for c in 0..ic {
                        for kernel_h in 0..kh {
                            for kernel_w in 0..kw {
                                let h = height * self.stride.0 + kernel_h;
                                let w = width * self.stride.1 + kernel_w;
                                sum += kernel[c][kernel_h][kernel_w] * x[c][h][w];
                            }
                        }
                    }
                    y[filter][height][width] = sum;

                    if let Some(bias) = &self.bias {
                        y[filter][height][width] += bias[filter];
                    }
                }
            }
        }

        let pre = tensor::Tensor::from(y);
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
    /// The weight gradient and the bias gradient of the layer.
    pub fn backward(
        &self, mut gradient: tensor::Tensor, input: &tensor::Tensor, output: &tensor::Tensor
    ) -> (tensor::Tensor, tensor::Tensor, Option<tensor::Tensor>) {

        println!("G: {}", gradient.shape);
        println!("I: {}", input.shape);

        let gradient_data = match &gradient.data {
            tensor::Data::Vector(vector) => {
                let (channels, rows, columns) = match &self.outputs {
                    tensor::Shape::Convolution(ch, he, wi) => (ch, he, wi),
                    _ => panic!("Expected a convolutional output shape."),
                };
                let mut iter = vector.into_iter();
                (0..*channels)
                    .map(|_| (0..*rows)
                        .map(|_| (0..*columns)
                            .map(|_| *iter.next().unwrap())
                            .collect())
                        .collect())
                    .collect()
            },
            tensor::Data::Tensor(data) => data.clone(),
            _ => panic!("Invalid data type"),
        };
        let input_data = match &input.data {
            tensor::Data::Tensor(data) => data,
            _ => panic!("Gradient data is not a tensor."),
        };

        let (in_channels, in_height, in_width) = (input_data.len(), input_data[0].len(), input_data[0][0].len());
        let (out_channels, out_height, out_width) = (gradient_data.len(), gradient_data[0].len(), gradient_data[0][0].len());

        let (filters, height, width) = match self.kernels[0].shape {
            tensor::Shape::Convolution(c, h, w) => (c, h, w),
            _ => panic!("Expected a convolutional kernel shape."),
        };

        // Initialize gradients
        let mut kernel_gradient = vec![vec![vec![vec![0.0; width]; height]; in_channels]; filters];
        let mut input_gradient = vec![vec![vec![0.0; in_width]; in_height]; in_channels];

        // Padding for full convolution
        let pad_height = height - 1;
        let pad_width = width - 1;
        let mut padded_gradient = vec![vec![vec![0.0; out_width + 2 * pad_width]; out_height + 2 * pad_height]; out_channels];

        // Fill padded gradient
        for c in 0..out_channels {
            for h in 0..out_height {
                for w in 0..out_width {
                    padded_gradient[c][h + pad_height][w + pad_width] = gradient_data[c][h][w];
                }
            }
        }

        // Calculate kernel gradients and input gradients
        for oc in 0..out_channels {
            for ic in 0..in_channels {
                for kh in 0..height {
                    for kw in 0..width {
                        for h in 0..out_height {
                            for w in 0..out_width {
                                kernel_gradient[oc][ic][kh][kw] += input_data[ic][h + kh][w + kw] * gradient_data[oc][h][w];
                            }
                        }
                    }
                }
            }
        }

        for ic in 0..in_channels {
            for h in 0..in_height {
                for w in 0..in_width {
                    for kernel in self.kernels.iter() {
                        let data = match &kernel.data {
                            tensor::Data::Tensor(data) => data,
                            _ => panic!("Expected a tensor data."),
                        };
                        for fi in 0..filters {
                            for kh in 0..height {
                                for kw in 0..width {
                                    input_gradient[ic][h][w] += data[fi][kh][kw] * padded_gradient[ic][h + kh][w + kw];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Calculate bias gradient if bias exists
        let bias_gradient = self.bias.as_ref().map(|_| {
            let mut bias_grad = vec![0.0; out_channels];
            for oc in 0..out_channels {
                bias_grad[oc] = gradient_data[oc].iter().flatten().sum();
            }
            tensor::Tensor::from_single(bias_grad)
        });

        gradient.data = tensor::Data::Tensor(input_gradient);

        (
            gradient,
            tensor::Tensor::gradient(kernel_gradient),
            bias_gradient
        )
    }
}
