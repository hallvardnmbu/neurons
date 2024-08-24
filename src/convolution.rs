// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::{activation, algebra, random, tensor};

/// A convolutional layer.
///
/// # Attributes
///
/// * `inputs` - The `tensor::Shape` of the input to the layer.
/// * `outputs` - The `tensor::Shape` of the output from the layer.
/// * `loops` - The number of loops to run the layer.
/// * `kernels` - The kernels of the layer.
/// * `stride` - The stride of the filter.
/// * `padding` - The padding applied to the input before convolving.
/// * `activation` - The `activation::Function` of the layer.
/// * `dropout` - The dropout rate of the layer (when training).
/// * `flatten_output` - Whether the output should be flattened.
/// * `training` - Whether the layer is training.
pub struct Convolution {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,
    pub(crate) loops: f32,

    pub kernels: Vec<tensor::Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),

    pub(crate) activation: activation::Function,

    dropout: Option<f32>,
    pub flatten_output: bool,
    pub training: bool,
}

impl std::fmt::Display for Convolution {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Convolution{}({} -> {}, kernel: {}x({}), stride: {:?}, padding: {:?}, loops: {})",
            self.activation,
            self.inputs,
            self.outputs,
            self.kernels.len(),
            self.kernels[0].shape,
            self.stride,
            self.padding,
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
    /// The `tensor::Shape` of the output from the layer.
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

    /// Creates a new convolutional layer with randomized kernel weights.
    ///
    /// # Arguments
    ///
    /// * `input` - The `tensor::Shape` of the input to the layer.
    /// * `filters` - The number of output channels from the layer.
    /// * `activation` - The `activation::Activation` function of the layer.
    /// * `kernel` - The size of each filter.
    /// * `stride` - The stride of the filter.
    /// * `padding` - The padding applied to the input before convolving.
    /// * `dropout` - The dropout rate of the layer (when training).
    ///
    /// # Returns
    ///
    /// A new layer with random weights with the given dimensions.
    pub fn create(
        input: tensor::Shape,
        filters: usize,
        activation: &activation::Activation,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dropout: Option<f32>,
    ) -> Self {
        let mut generator = random::Generator::create(12345);
        Convolution {
            inputs: input.clone(),
            outputs: Convolution::calculate_output_size(
                &input, &filters, &kernel, &stride, &padding,
            ),

            kernels: {
                let in_channels = match input {
                    tensor::Shape::Vector(size) => size,
                    tensor::Shape::Tensor(ch, _, _) => ch,
                    _ => unimplemented!("Expected a dense or convolutional input shape."),
                };
                (0..filters)
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
            // bias: match bias {
            //     true => Some(
            //         (0..filters)
            //             .map(|_| generator.generate(-1.0, 1.0))
            //             .collect(),
            //     ),
            //     false => None,
            // },
            activation: activation::Function::create(&activation),
            dropout,
            stride,
            padding,
            training: false,
            flatten_output: false,
            loops: 1.0,
        }
    }

    /// Extract the number of parameters in the layer.
    pub fn parameters(&self) -> usize {
        self.kernels.len()
            * match self.kernels[0].data {
                tensor::Data::Tensor(ref tensor) => {
                    tensor.len() * tensor[0].len() * tensor[0][0].len()
                }
                _ => 0,
            }
    }

    /// Convolves `x` with the given `kernels`.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor to convolve.
    /// * `kernels` - The kernels to convolve the input with.
    ///
    /// # Returns
    ///
    /// The output vector after convolving the input with the kernels.
    fn convolve(
        &self,
        x: &Vec<Vec<Vec<f32>>>,
        kernels: &Vec<&Vec<Vec<Vec<f32>>>>,
    ) -> Vec<Vec<Vec<f32>>> {
        let (ic, ih, iw) = (x.len(), x[0].len(), x[0][0].len());
        let (kf, kc, kh, kw) = (
            kernels.len(),
            kernels[0].len(),
            kernels[0][0].len(),
            kernels[0][0][0].len(),
        );

        assert_eq!(
            ic, kc,
            "The number of input channels should match the kernel channels."
        );

        // Defining the output dimensions and vector.
        let oh = (ih - kh) / self.stride.0 + 1;
        let ow = (iw - kw) / self.stride.1 + 1;
        let mut y = vec![vec![vec![0.0; ow]; oh]; kf];

        // Convolving the input with the kernels.
        for filter in 0..kf {
            for height in 0..oh {
                for width in 0..ow {
                    let mut sum = 0.0;
                    for c in 0..kc {
                        for h in 0..kh {
                            for w in 0..kw {
                                let _h = height * self.stride.0 + h;
                                let _w = width * self.stride.1 + w;
                                sum += kernels[filter][c][h][w] * x[c][_h][_w];
                            }
                        }
                    }

                    // if let Some(bias) = &self.bias {
                    //     sum += bias[filter];
                    // }

                    y[filter][height][width] = sum;
                }
            }
        }

        y
    }

    /// Convolves the two gradients.
    ///
    /// # Arguments
    ///
    /// * `a` - The first gradient tensor.
    /// * `b` - The second gradient tensor.
    ///
    /// # Returns
    ///
    /// The convolved gradients.
    ///
    /// # Notes
    ///
    /// The outputted gradient tensor will have the shape:
    ///
    /// * `[a_channels, b_channels, height, width]`
    ///
    /// Where `height` and `width` are calculated as:
    ///
    /// * `height = (a_height - b_height) / stride.0 + 1`
    /// * `width = (a_width - b_width) / stride.1 + 1`
    fn convolve_gradients(
        &self,
        a: &Vec<Vec<Vec<f32>>>,
        b: &Vec<Vec<Vec<f32>>>,
    ) -> Vec<Vec<Vec<Vec<f32>>>> {
        let (ac, ah, aw) = (a.len(), a[0].len(), a[0][0].len());
        let (bc, bh, bw) = (b.len(), b[0].len(), b[0][0].len());

        // Defining the output shapes and vector.
        let oh = ((ah - bh) / self.stride.0) + 1;
        let ow = ((aw - bw) / self.stride.1) + 1;
        let mut y = vec![vec![vec![vec![0.0; ow]; oh]; ac]; bc];

        // Convolving `a` with `b`.
        for i in 0..bc {
            for j in 0..ac {
                for k in 0..oh {
                    for l in 0..ow {
                        let mut sum = 0.0;
                        for m in 0..bh {
                            for n in 0..bw {
                                let _h = m * self.stride.0 + k;
                                let _w = n * self.stride.1 + l;
                                sum += a[j][_h][_w] * b[i][m][n];
                            }
                        }
                        y[i][j][k][l] = sum;
                    }
                }
            }
        }

        y
    }

    /// Applies the forward pass (convolution) to the input `tensor::Tensor`.
    /// Assumes `x` to match `self.inputs`, and for performance reasons does not check.
    ///
    /// # Arguments
    ///
    /// * `x` - The input `tensor::Tensor` to the layer.
    ///
    /// # Returns
    ///
    /// The pre-activation and post-activation `tensor::Tensor`s of the convolved input wrt. the kernels.
    pub fn forward(&self, x: &tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {
        // Extracting the data from the input Tensor.
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

        // Extracting the weights from the kernels.
        let kernels: Vec<&Vec<Vec<Vec<f32>>>> = self
            .kernels
            .iter()
            .map(|ref k| match k.data {
                tensor::Data::Tensor(ref kernel) => kernel,
                _ => panic!("Expected `Tensor` kernel data."),
            })
            .collect();

        // Convolving the input with the kernels.
        let y = self.convolve(&x, &kernels);

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

    /// Applies the backward pass of the layer to the gradient `tensor::Tensor`.
    /// Assumes `input` to match `self.inputs`, and for performance reasons does not check.
    /// Assumes `output` to match `self.outputs`, and for performance reasons does not check.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient `tensor::Tensor` to the layer.
    /// * `input` - The input `tensor::Tensor` to the layer.
    /// * `output` - The output `tensor::Tensor` of the layer.
    ///
    /// # Returns
    ///
    /// The input-, weight- and bias gradient of the layer.
    ///
    /// # Notes
    ///
    /// [Source](https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf)
    pub fn backward(
        &self,
        gradient: &tensor::Tensor,
        input: &tensor::Tensor,
        output: &tensor::Tensor,
    ) -> (tensor::Tensor, tensor::Tensor, Option<tensor::Tensor>) {
        let gradient = gradient.get_data(&self.outputs);
        let derivative = self.activation.backward(&output).get_data(&self.outputs);
        let delta = algebra::mul3d_scalar(&algebra::mul3d(&gradient, &derivative), self.loops);

        // Extracting the kernel dimensions.
        let kf = self.kernels.len(); // Number of filters.
        let (kc, kh, kw) = match self.kernels[0].shape {
            tensor::Shape::Tensor(c, h, w) => (c, h, w),
            _ => panic!("Expected individual kernels to be three-dimensional."),
        };

        // Extracting the input and its dimensions.
        let input = input.get_data(&self.inputs);
        let (ih, iw) = (input[0].len(), input[0][0].len());

        // Pad the input tensor to provide the kernel gradient when convolving the delta.
        // Based on the formula for the convolutional output, we can derive the formula for the padding:
        let ph = ih * self.stride.0 + delta[0].len() - self.stride.0;
        let pw = iw * self.stride.1 + delta[0][0].len() - self.stride.1;
        let input = algebra::pad3d(&input, (ph, pw));

        // dL/dF = Conv(X, dL/dY)
        let kgradient = self.convolve_gradients(&input, &delta);

        // Flipping the kernels.
        let kernels: Vec<Vec<Vec<Vec<f32>>>> = self
            .kernels
            .iter()
            .map(|k| match &k.data {
                tensor::Data::Tensor(ref kernel) => {
                    let mut flipped_kernel = kernel.clone();
                    flipped_kernel.iter_mut().for_each(|channel| {
                        channel.iter_mut().for_each(|row| {
                            row.reverse();
                        });
                        channel.reverse();
                    });
                    flipped_kernel
                }
                _ => panic!("Expected `Tensor` kernel data."),
            })
            .collect();

        // Rearrange from FxCxHxW to CxFxHxW
        let kernels: Vec<Vec<Vec<Vec<f32>>>> = {
            let mut rearranged = vec![vec![vec![vec![0.0; kw]; kh]; kf]; kc];
            for c in 0..kc {
                for f in 0..kf {
                    for h in 0..kh {
                        for w in 0..kw {
                            rearranged[c][f][h][w] = kernels[f][c][h][w];
                        }
                    }
                }
            }
            rearranged
        };

        // Pad the delta tensor to provide a full convolution.
        // Based on the formula for the convolutional output, we can derive the formula for the padding:
        let ph = ih * self.stride.0 + kh - self.stride.0;
        let pw = iw * self.stride.1 + kw - self.stride.1;
        let delta = algebra::pad3d(&delta, (ph, pw));

        // dL/dX = FullConv(dL/dY, flip(F))
        let igradient = self.convolve(&delta, &kernels.iter().map(|k| k.as_ref()).collect());

        // // Calculate bias gradient if bias exists.
        // let bias_gradient = self.bias.as_ref().map(|_| {
        //     let mut bias_grad = vec![0.0; kf];
        //     for c in 0..kf {
        //         bias_grad[kf] = gradient[c].iter().flatten().sum();
        //     }
        //     tensor::Tensor::from_single(bias_grad)
        // });

        (
            tensor::Tensor::from(igradient),
            tensor::Tensor::gradient(kgradient),
            None,
        )
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
            (3, 3),
            (1, 1),
            (0, 0),
            None,
        );

        assert_eq!(conv.inputs, tensor::Shape::Tensor(1, 5, 5));
        assert_eq!(conv.outputs, tensor::Shape::Tensor(1, 3, 3));
        assert_eq!(conv.kernels.len(), 1);
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
    fn test_backward() {
        // Test backward function with a simple input, output, and gradient
        // The output should be the same as the input
        let mut conv = Convolution::create(
            tensor::Shape::Tensor(1, 3, 3),
            1,
            &activation::Activation::Linear,
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

        let output = tensor::Tensor::from(vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]]);

        // Double-check to ensure the forward pass is correct.
        let (pre, post) = conv.forward(&input);
        assert_eq_data!(pre.data, output.data);
        assert_eq_data!(post.data, output.data);

        let gradient = tensor::Tensor::from(vec![vec![vec![1.0; 3]; 3]]);

        let (input_gradient, kernel_gradient, _) = conv.backward(&gradient, &input, &output);

        let _input_gradient = tensor::Tensor::from(vec![vec![vec![1.0; 3]; 3]]);
        let _kernel_gradient = tensor::Tensor::gradient(vec![vec![vec![
            vec![12.0, 21.0, 16.0],
            vec![27.0, 45.0, 33.0],
            vec![24.0, 39.0, 28.0],
        ]]]);

        assert_eq_data!(input_gradient.data, _input_gradient.data);
        assert_eq_data!(kernel_gradient.data, _kernel_gradient.data);
    }

    #[test]
    fn test_identity_convolution() {
        let mut conv = Convolution::create(
            tensor::Shape::Tensor(1, 3, 3),
            1,
            &activation::Activation::Linear,
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
