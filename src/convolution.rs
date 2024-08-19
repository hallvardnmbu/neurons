// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::activation;
use crate::algebra::{
    add3d, convolve_3d, mul3d, mul3d_scalar, mul_scalar, sub_inplace, sub_inplace_tensor,
};
use crate::assert_eq_shape;
use crate::random;
use crate::tensor;

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

    pub(crate) loops: f32,
}

impl std::fmt::Display for Convolution {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Convolution{}({} -> {}, kernel: {}x({}), stride: {:?}, padding: {:?}, bias: {}, loops: {})",
            self.activation,
            self.inputs,
            self.outputs,
            self.kernels.len(),
            self.kernels[0].shape,
            self.stride,
            self.padding,
            self.bias.is_some(),
            self.loops
        )
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
        padding: &(usize, usize),
    ) -> tensor::Shape {
        let input: &(usize, usize, usize) = match input {
            tensor::Shape::Vector(shape) => {
                let root = (*shape as f32).sqrt() as usize;
                &(1, root, root)
            }
            tensor::Shape::Tensor(ch, he, wi) => &(*ch, *he, *wi),
            _ => unimplemented!("Expected a dense or convolutional input shape."),
        };

        let height = (input.1 + 2 * padding.0 - kernel.0) / stride.0 + 1;
        let width = (input.2 + 2 * padding.1 - kernel.1) / stride.1 + 1;

        tensor::Shape::Tensor(*channels, height, width)
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
                &input, &channels, &kernel, &stride, &padding,
            ),

            kernels: {
                let in_channels = match input {
                    tensor::Shape::Vector(size) => size,
                    tensor::Shape::Tensor(ch, _, _) => ch,
                    _ => unimplemented!("Expected a dense or convolutional input shape."),
                };
                (0..channels)
                    .map(|_| {
                        tensor::Tensor::from(
                            (0..in_channels)
                                .map(|_| {
                                    (0..kernel.0)
                                        .map(|_| {
                                            (0..kernel.1)
                                                .map(|_| generator.generate(-1.0, 1.0))
                                                .collect()
                                        })
                                        .collect()
                                })
                                .collect(),
                        )
                    })
                    .collect()
            },
            bias: match bias {
                true => Some(
                    (0..channels)
                        .map(|_| generator.generate(-1.0, 1.0))
                        .collect(),
                ),
                false => None,
            },
            activation: activation::Function::create(&activation),
            dropout,
            stride,
            padding,
            training: false,
            flatten_output: false,
            loops: 1.0,
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
    /// The pre-activation and post-activation Tensors of the convolved input wrt. the kernels.
    pub fn forward(&self, x: &tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {
        assert_eq_shape!(x.shape, self.inputs);

        // Extracting the input data and dimensions.
        let (mut x, ic, ih, iw) = match &x.data {
            tensor::Data::Vector(vector) => {
                let root = (vector.len() as f32).sqrt() as usize;
                (
                    vec![vector.chunks(root).map(|v| v.to_vec()).collect()],
                    1,
                    root,
                    root,
                )
            }
            tensor::Data::Tensor(tensor) => (
                tensor.clone(),
                tensor.len(),
                tensor[0].len(),
                tensor[0][0].len(),
            ),
            _ => panic!("Expected `Vector` or `Tensor` input data."),
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
            tensor::Shape::Tensor(c, h, w) => (c, h, w),
            _ => panic!("The kernel should be a `Convolution`-shaped (3D) Tensor."),
        };
        assert_eq!(
            ic, kc,
            "The number of input channels should match the kernel channels."
        );

        // Defining the output dimensions and Tensor.
        let oh = (ph - kh) / self.stride.0 + 1;
        let ow = (pw - kw) / self.stride.1 + 1;
        let mut y = vec![vec![vec![0.0; ow]; oh]; oc];

        // Convolving the input with the kernels.
        for filter in 0..oc {
            let kernel = match &self.kernels[filter].data {
                tensor::Data::Tensor(kernel) => kernel,
                _ => panic!("The kernel data should be of type `Tensor` (3D)."),
            };

            for height in 0..oh {
                for width in 0..ow {
                    let mut sum = 0.0;
                    for c in 0..kc {
                        for h in 0..kh {
                            for w in 0..kw {
                                let _h = height * self.stride.0 + h;
                                let _w = width * self.stride.1 + w;
                                sum += kernel[c][h][w] * x[c][_h][_w];
                            }
                        }
                    }

                    if let Some(bias) = &self.bias {
                        sum += bias[filter];
                    }

                    y[filter][height][width] = sum;
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

        assert_eq_shape!(pre.shape, self.outputs);
        if self.flatten_output {
            post = post.flatten();
            assert_eq_shape!(post.shape, tensor::Shape::Vector(oc * oh * ow));
        } else {
            assert_eq_shape!(post.shape, self.outputs);
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
    /// The input-, weight- and bias gradient of the layer.
    pub fn backward(
        &self,
        gradient: &tensor::Tensor,
        input: &tensor::Tensor,
        output: &tensor::Tensor,
    ) -> (tensor::Tensor, tensor::Tensor, Option<tensor::Tensor>) {
        assert_eq_shape!(input.shape, self.inputs);
        assert_eq_shape!(output.shape, self.outputs);

        // println!("Input: {}", input.shape);
        // println!("Output: {}", output.shape);
        // println!("Gradients in: {}", gradient.shape);

        let gradient = gradient.get_data(&self.outputs);
        let derivative = self.activation.backward(&output).get_data(&self.outputs);
        let delta = mul3d_scalar(&mul3d(&gradient, &derivative), self.loops);

        let input = input.get_data(&self.inputs);
        let output = output.get_data(&self.outputs);

        // println!(
        //     "Gradients processed: {}x{}x{}",
        //     gradient.len(),
        //     gradient[0].len(),
        //     gradient[0][0].len()
        // );
        // println!(
        //     "Derivative: {}x{}x{}",
        //     derivative.len(),
        //     derivative[0].len(),
        //     derivative[0][0].len()
        // );
        // println!(
        //     "Delta: {}x{}x{}",
        //     delta.len(),
        //     delta[0].len(),
        //     delta[0][0].len()
        // );

        // Extracting the input- and output dimensions.
        let (ic, ih, iw) = match &self.inputs {
            tensor::Shape::Tensor(c, h, w) => (*c, *h, *w),
            _ => panic!("Expected input to be a three-dimensional tensor."),
        };
        let (oc, oh, ow) = match &self.outputs {
            tensor::Shape::Tensor(c, h, w) => (*c, *h, *w),
            _ => panic!("Expected output to be a three-dimensional tensor."),
        };

        // Extracting the kernel dimensions.
        let kf = self.kernels.len(); // Number of filters.
        let (kc, kh, kw) = match self.kernels[0].shape {
            tensor::Shape::Tensor(c, h, w) => (c, h, w),
            _ => panic!("Expected individual kernels to be three-dimensional."),
        };

        // Asserting that shapes match.
        assert_eq!(
            ic, kc,
            "The number of input channels should match the kernel channels."
        );
        assert_eq!(
            oc, kf,
            "The number of output channels should match the kernel filters."
        );

        // Initialize input- and kernel gradients; to be filled.
        let mut igradient = vec![vec![vec![0.0; iw]; ih]; ic];
        let mut kgradient = vec![vec![vec![vec![0.0; kw]; kh]; kc]; kf];

        for filter in 0..kf {
            let kernel = match &self.kernels[filter].data {
                tensor::Data::Tensor(kernel) => kernel,
                _ => panic!("The kernels should be `Tensor`s."),
            };

            // Source:
            // https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf

            // Calculate kernel gradient.
            kgradient[filter] = convolve_3d(&input, &delta, &self.stride, &(0, 0));

            // println!(
            //     "Kernel gradient {}x{}x{}",
            //     kgradient[filter].len(),
            //     kgradient[filter][0].len(),
            //     kgradient[filter][0][0].len()
            // );

            // Calculate input gradient.
            igradient = add3d(
                &igradient,
                &convolve_3d(&delta, &kernel, &self.stride, &self.padding),
            );

            // println!(
            //     "Input gradient {}x{}x{}",
            //     igradient.len(),
            //     igradient[0].len(),
            //     igradient[0][0].len()
            // );
        }

        // Calculate bias gradient if bias exists.
        let bias_gradient = self.bias.as_ref().map(|_| {
            let mut bias_grad = vec![0.0; oc];
            for c in 0..oc {
                bias_grad[oc] = gradient[c].iter().flatten().sum();
            }
            tensor::Tensor::from_single(bias_grad)
        });

        (
            tensor::Tensor::from(igradient),
            tensor::Tensor::gradient(kgradient),
            bias_gradient,
        )
    }

    pub fn update(
        &mut self,
        stepnr: i32,
        kernel_gradient: &tensor::Tensor,
        bias_gradient: &Option<tensor::Tensor>,
        lr: f32,
    ) {
        let kernel_gradient = match &kernel_gradient.data {
            tensor::Data::Gradient(data) => data,
            _ => panic!("Expected a Tensor as the kernel (weight) gradient."),
        };

        for (filter, kernel) in self.kernels.iter_mut().enumerate() {
            let data = match &mut kernel.data {
                tensor::Data::Tensor(data) => data,
                _ => panic!("Expected a tensor data."),
            };
            let gradient = &kernel_gradient[filter];

            // TODO: Update wrt. optimizer.

            sub_inplace_tensor(data, &mul3d_scalar(gradient, lr));

            // TODO: Bias update wrt. optimizer.

            if let Some(bias) = &mut self.bias {
                let bias_gradient = match bias_gradient {
                    Some(bias) => match &bias.data {
                        tensor::Data::Vector(data) => data,
                        _ => panic!("Expected a Vector as the bias gradient."),
                    },
                    None => panic!("Bias gradient is missing."),
                };
                sub_inplace(bias, &mul_scalar(bias_gradient, lr));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_eq_data;

    #[test]
    fn test_calculate_output_size() {
        let input = tensor::Shape::Tensor(1, 5, 5);
        let channels = 1;
        let kernel = (3, 3);
        let stride = (1, 1);
        let padding = (0, 0);

        let output_size =
            Convolution::calculate_output_size(&input, &channels, &kernel, &stride, &padding);

        assert_eq!(output_size, tensor::Shape::Tensor(1, 3, 3));
    }

    #[test]
    fn test_create() {
        let conv = Convolution::create(
            tensor::Shape::Tensor(1, 5, 5),
            1,
            &activation::Activation::Linear,
            false,
            (3, 3),
            (1, 1),
            (0, 0),
            None,
        );

        assert_eq!(conv.inputs, tensor::Shape::Tensor(1, 5, 5));
        assert_eq!(conv.outputs, tensor::Shape::Tensor(1, 3, 3));
        assert_eq!(conv.kernels.len(), 1);
        assert_eq!(conv.bias, None);
        assert_eq!(conv.dropout, None);
        assert_eq!(conv.stride, (1, 1));
        assert_eq!(conv.padding, (0, 0));
        assert_eq!(conv.training, false);
        assert_eq!(conv.flatten_output, false);
    }

    #[test]
    fn test_forward() {
        // Test forward function with a simple input and identity kernel
        // The output should be the same as the input
        let mut conv = Convolution::create(
            tensor::Shape::Tensor(1, 3, 3),
            1,
            &activation::Activation::Linear,
            false,
            (3, 3),
            (1, 1),
            (1, 1),
            None,
        );
        conv.kernels[0] = tensor::Tensor::from(vec![vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]]);

        let input = tensor::Tensor::from(vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]]);

        let (output, _) = conv.forward(&input);
        assert_eq_data!(output.data, input.data);
    }

    // #[test]
    // fn test_backward() {
    //     // Test backward function with a simple input, output, and gradient
    //     // The output should be the same as the input
    //     let mut conv = Convolution::create(
    //         tensor::Shape::Tensor(1, 3, 3),
    //         1,
    //         &activation::Activation::Linear,
    //         false,
    //         (3, 3),
    //         (1, 1),
    //         (1, 1),
    //         None,
    //     );
    //     conv.kernels[0] = tensor::Tensor::from(vec![vec![
    //         vec![0.0, 0.0, 0.0],
    //         vec![0.0, 1.0, 0.0],
    //         vec![0.0, 0.0, 0.0],
    //     ]]);
    //
    //     let input = tensor::Tensor::from(vec![vec![
    //         vec![1.0, 2.0, 3.0],
    //         vec![4.0, 5.0, 6.0],
    //         vec![7.0, 8.0, 9.0],
    //     ]]);
    //
    //     let output = tensor::Tensor::from(vec![vec![
    //         vec![1.0, 2.0, 3.0],
    //         vec![4.0, 5.0, 6.0],
    //         vec![7.0, 8.0, 9.0],
    //     ]]);
    //
    //     let gradient = tensor::Tensor::from(vec![vec![
    //         vec![1.0, 1.0, 1.0],
    //         vec![1.0, 1.0, 1.0],
    //         vec![1.0, 1.0, 1.0],
    //     ]]);
    //
    //     let (input_gradient, kernel_gradient, bias_gradient) = conv.backward(&gradient, &input, &output);
    //
    //     assert_eq_data!(input_gradient.data, input.data);
    //     assert_eq_data!(kernel_gradient.data, conv.kernels[0].data);
    // }

    #[test]
    fn test_update() {
        // Test update function with a simple input, output, and gradient
        // The output should be the same as the input
        let mut conv = Convolution::create(
            tensor::Shape::Tensor(1, 3, 3),
            1,
            &activation::Activation::Linear,
            true,
            (3, 3),
            (1, 1),
            (1, 1),
            None,
        );
        conv.kernels[0] = tensor::Tensor::from(vec![vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]]);
        conv.bias = Some(vec![1.0]);

        let kernel_gradient = tensor::Tensor::gradient(vec![vec![vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]]]);

        let bias_gradient = tensor::Tensor::from_single(vec![1.0]);

        conv.update(1, &kernel_gradient, &Some(bias_gradient), 0.1);

        assert_eq_data!(
            conv.kernels[0].data,
            tensor::Tensor::from(vec![vec![
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.9, 0.0],
                vec![0.0, 0.0, 0.0],
            ]])
            .data
        );
        assert_eq!(conv.bias, Some(vec![0.9]));
    }

    #[test]
    fn test_identity_convolution() {
        let mut conv = Convolution::create(
            tensor::Shape::Tensor(1, 3, 3),
            1,
            &activation::Activation::Linear,
            false,
            (3, 3),
            (1, 1),
            (1, 1),
            None,
        );
        conv.kernels[0] = tensor::Tensor::from(vec![vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]]);

        let input = tensor::Tensor::from(vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]]);

        let (output, _) = conv.forward(&input);
        assert_eq_data!(output.data, input.data);
    }

    #[test]
    fn test_edge_detection_convolution() {
        let mut conv = Convolution::create(
            tensor::Shape::Tensor(1, 5, 5),
            1,
            &activation::Activation::Linear,
            false,
            (3, 3),
            (1, 1),
            (0, 0),
            None,
        );
        conv.kernels[0] = tensor::Tensor::from(vec![vec![
            vec![-1.0, -1.0, -1.0],
            vec![-1.0, 8.0, -1.0],
            vec![-1.0, -1.0, -1.0],
        ]]);

        let input = tensor::Tensor::from(vec![vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 2.0, 2.0, 1.0],
            vec![1.0, 2.0, 3.0, 2.0, 1.0],
            vec![1.0, 2.0, 2.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ]]);

        let (output, _) = conv.forward(&input);

        // The central pixel should have the highest value
        let central_value = output.as_tensor()[0][1][1];
        assert!(central_value > 0.0);

        // The edges should be detected
        for i in 0..3 {
            for j in 0..3 {
                if i != 1 || j != 1 {
                    assert!(output.as_tensor()[0][i][j] < central_value);
                }
            }
        }
    }

    #[test]
    fn test_stride_and_padding() {
        let mut conv = Convolution::create(
            tensor::Shape::Tensor(1, 5, 5),
            1,
            &activation::Activation::Linear,
            false,
            (3, 3),
            (2, 2),
            (1, 1),
            None,
        );
        conv.kernels[0] = tensor::Tensor::from(vec![vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ]]);

        let input = tensor::Tensor::from(vec![vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0, 1.0, 3.0],
        ]]);

        let (output, _) = conv.forward(&input);

        // Check output dimensions
        assert_eq!(output.as_tensor()[0].len(), 3);
        assert_eq!(output.as_tensor()[0][0].len(), 3);

        // Check some values
        assert_eq!(output.as_tensor()[0][0][0], 16.0); // Top-left
        assert_eq!(output.as_tensor()[0][2][2], 7.0); // Bottom-right
    }
}
