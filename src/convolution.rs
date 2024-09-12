// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::{activation, tensor};

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
/// * `flatten` - Whether the output should be flattened.
/// * `training` - Whether the layer is training.
#[derive(Clone)]
pub struct Convolution {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,
    pub(crate) loops: f32,

    pub(crate) kernels: Vec<tensor::Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),

    pub(crate) activation: activation::Function,

    dropout: Option<f32>,
    pub(crate) flatten: bool,
    pub(crate) training: bool,
}

impl std::fmt::Display for Convolution {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Convolution{}(\n", self.activation)?;
        write!(f, "\t\t\t{} -> {}\n", self.inputs, self.outputs)?;
        write!(
            f,
            "\t\t\tkernel: {}x({})\n",
            self.kernels.len(),
            self.kernels[0].shape
        )?;
        write!(f, "\t\t\tstride: {:?}\n", self.stride)?;
        write!(f, "\t\t\tpadding: {:?}\n", self.padding)?;
        write!(
            f,
            "\t\t\tdropout: {}\n",
            if self.dropout.is_some() {
                self.dropout.unwrap().to_string()
            } else {
                "false".to_string()
            }
        )?;
        write!(f, "\t\t\tloops: {}\n", self.loops)?;
        write!(f, "\t\t)")?;
        Ok(())
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
            tensor::Shape::Single(size) => {
                let root = (*size as f32).sqrt() as usize;
                &(1, root, root)
            }
            tensor::Shape::Triple(ch, he, wi) => &(*ch, *he, *wi),
            _ => panic!("Incorrect input shape."),
        };

        let height = (input.1 + 2 * padding.0 - kernel.0) / stride.0 + 1;
        let width = (input.2 + 2 * padding.1 - kernel.1) / stride.1 + 1;

        tensor::Shape::Triple(*channels, height, width)
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
        inputs: tensor::Shape,
        filters: usize,
        activation: &activation::Activation,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dropout: Option<f32>,
    ) -> Self {
        let (inputs, ic) = match &inputs {
            tensor::Shape::Single(size) => {
                let root = (*size as f32).sqrt() as usize;
                if size % root == 0 {
                    (tensor::Shape::Triple(1, root, root), 1)
                } else {
                    panic!("> When adding a convolutional layer after a dense layer, the dense layer must have a square output.\n> Currently, the layer has {} outputs, which cannot cannot be reshaped to a (1, root[{}], root[{}]) tensor.\n> Try using {} or {} outputs for the preceding dense layer.", size, size, size, root*root, (root+1)*(root+1));
                }
            }
            tensor::Shape::Triple(ic, _, _) => (inputs.clone(), *ic),
            _ => unimplemented!("Expected a `tensor::Tensor` input shape."),
        };
        let outputs =
            Convolution::calculate_output_size(&inputs, &filters, &kernel, &stride, &padding);
        Convolution {
            inputs,
            outputs,
            kernels: (0..filters)
                .map(|_| {
                    tensor::Tensor::random(tensor::Shape::Triple(ic, kernel.0, kernel.1), -1.0, 1.0)
                })
                .collect(),
            activation: activation::Function::create(&activation),
            dropout,
            stride,
            padding,
            training: false,
            flatten: false,
            loops: 1.0,
        }
    }

    /// Extract the number of parameters in the layer.
    pub fn parameters(&self) -> usize {
        self.kernels.len()
            * match self.kernels[0].data {
                tensor::Data::Triple(ref tensor) => {
                    tensor.len() * tensor[0].len() * tensor[0][0].len()
                }
                _ => 0,
            }
    }

    /// Convolves `x` with the given `kernels`.
    /// Assumes that the input and kernel shapes are valid and correct, for speed.
    ///
    /// # Arguments
    ///
    /// * `x` - The input vector to convolve.
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
        let (ih, iw) = (x[0].len(), x[0][0].len());
        let (kf, kc, kh, kw) = (
            kernels.len(),
            kernels[0].len(),
            kernels[0][0].len(),
            kernels[0][0][0].len(),
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
                    y[filter][height][width] = sum;
                }
            }
        }

        y
    }

    /// Convolves two three-dimensional vectors producing a four-dimensional vector.
    ///
    /// # Arguments
    ///
    /// * `a` - The first three-dimensional vector.
    /// * `b` - The second three-dimensional vector.
    ///
    /// # Returns
    ///
    /// The convolved result.
    ///
    /// # Notes
    ///
    /// The outputted vector will have the shape:
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
    /// The pre- and post-activation `tensor::Tensor`s of the convolved input wrt. the kernels.
    pub fn forward(&self, x: &tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {
        // Extracting the data from the input `tensor::Tensor`.
        let (mut x, ih, iw) = match &x.data {
            tensor::Data::Single(vector) => {
                let (h, w) = match &self.inputs {
                    tensor::Shape::Triple(_, h, w) => (*h, *w),
                    _ => panic!("Convolutional layers should have `tensor::Shape::Triple` input."),
                };
                (
                    vector
                        .chunks_exact(h * w)
                        .map(|channel| channel.chunks_exact(w).map(|row| row.to_vec()).collect())
                        .collect(),
                    h,
                    w,
                )
            }
            tensor::Data::Triple(tensor) => (tensor.clone(), tensor[0].len(), tensor[0][0].len()),
            _ => panic!("Unexpected input data type."),
        };

        // Padding the input wrt. `self.padding`.
        let ph = ih + 2 * self.padding.0;
        let pw = iw + 2 * self.padding.1;
        x = tensor::pad3d(&x, (ph, pw));

        // Extracting the weights from the kernels.
        let kernels: Vec<&Vec<Vec<Vec<f32>>>> = self
            .kernels
            .iter()
            .map(|ref k| match k.data {
                tensor::Data::Triple(ref kernel) => kernel,
                _ => panic!("Expected `tensor::Shape::Triple` kernel shape."),
            })
            .collect();

        // Convolving the input with the kernels.
        let y = self.convolve(&x, &kernels);

        let pre = tensor::Tensor::triple(y);
        let mut post = self.activation.forward(&pre);

        // Apply dropout if the network is training.
        if self.training {
            if let Some(dropout) = self.dropout {
                post.dropout(dropout);
            }
        }

        if self.flatten {
            post = post.flatten();
        }

        (pre, post)
    }

    /// Applies the backward pass of the layer to the gradient `tensor::Tensor`.
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
        let gradient = gradient.get_triple(&self.outputs);
        let derivative = self.activation.backward(&output).get_triple(&self.outputs);
        let delta = tensor::hadamard3d(&gradient, &derivative, 1.0 / self.loops);

        // Extracting the kernel dimensions.
        let (kh, kw) = match self.kernels[0].shape {
            tensor::Shape::Triple(_, h, w) => (h, w),
            _ => panic!("Expected individual kernels to be three-dimensional."),
        };

        // Extracting the input and its dimensions.
        let input = input.get_triple(&self.inputs);
        let (ih, iw) = (input[0].len(), input[0][0].len());

        // Pad the input vector to provide the kernel gradient when convolving the delta.
        // Based on the formula for the convolutional output, we can derive the formula for the padding.
        // `o = (i - k) / s + 1 => i = o + k * s - s`
        let ph = delta[0].len() + kh * self.stride.0 - self.stride.0;
        let pw = delta[0][0].len() + kw * self.stride.1 - self.stride.1;
        let input = tensor::pad3d(&input, (ph, pw));

        // dL/dF = Conv(X, dL/dY)
        let kgradient = self.convolve_gradients(&input, &delta);

        // Flipping the kernels.
        let mut kernels: Vec<Vec<Vec<Vec<f32>>>> = self
            .kernels
            .iter()
            .map(|k| match &k.data {
                tensor::Data::Triple(ref kernel) => self.rotate(kernel.clone()),
                _ => panic!("Expected `Tensor` kernel data."),
            })
            .collect();

        // Rearrange from FxCxHxW to CxFxHxW
        kernels = self.rearrange(&kernels);

        // Pad the delta vector to provide a full convolution.
        // Based on the formula for the convolutional output, we can derive the formula for the padding.
        // `o = (i - k + 2 * p) / s + 1 => i = o * s + k - s - 2 * p`
        let ph = ih * self.stride.0 + kh - self.stride.0;
        let pw = iw * self.stride.1 + kw - self.stride.1;
        let delta = tensor::pad3d(&delta, (ph, pw));

        // dL/dX = FullConv(dL/dY, flip(F))
        let igradient = self.convolve(&delta, &kernels.iter().map(|k| k.as_ref()).collect());

        (
            tensor::Tensor::triple(igradient),
            tensor::Tensor::quadruple(kgradient),
            None,
        )
    }

    /// Flips the kernel by 180 degrees.
    fn rotate(&self, mut kernel: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        kernel.iter_mut().for_each(|channel| {
            channel.iter_mut().for_each(|row| {
                row.reverse();
            });
            channel.reverse();
        });
        kernel
    }

    /// Rearrange from FxCxHxW to CxFxHxW
    fn rearrange(&self, kernels: &Vec<Vec<Vec<Vec<f32>>>>) -> Vec<Vec<Vec<Vec<f32>>>> {
        let (kf, kc, kh, kw) = (
            kernels.len(),
            kernels[0].len(),
            kernels[0][0].len(),
            kernels[0][0][0].len(),
        );
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_eq_data;

    #[test]
    fn test_calculate_output_size() {
        let input = tensor::Shape::Triple(1, 5, 5);
        let channels = 1;
        let kernel = (3, 3);
        let stride = (1, 1);
        let padding = (0, 0);

        let output_size =
            Convolution::calculate_output_size(&input, &channels, &kernel, &stride, &padding);

        assert_eq!(output_size, tensor::Shape::Triple(1, 3, 3));
    }

    #[test]
    fn test_create() {
        let conv = Convolution::create(
            tensor::Shape::Triple(1, 5, 5),
            1,
            &activation::Activation::Linear,
            (3, 3),
            (1, 1),
            (0, 0),
            None,
        );

        assert_eq!(conv.inputs, tensor::Shape::Triple(1, 5, 5));
        assert_eq!(conv.outputs, tensor::Shape::Triple(1, 3, 3));
        assert_eq!(conv.kernels.len(), 1);
        assert_eq!(conv.dropout, None);
        assert_eq!(conv.stride, (1, 1));
        assert_eq!(conv.padding, (0, 0));
        assert_eq!(conv.training, false);
        assert_eq!(conv.flatten, false);
    }

    #[test]
    fn test_forward() {
        // Test forward function with a simple input and identity kernel
        // The output should be the same as the input
        let mut conv = Convolution::create(
            tensor::Shape::Triple(1, 3, 3),
            1,
            &activation::Activation::Linear,
            (3, 3),
            (1, 1),
            (1, 1),
            None,
        );
        conv.kernels[0] = tensor::Tensor::triple(vec![vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]]);

        let input = tensor::Tensor::triple(vec![vec![
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
            tensor::Shape::Triple(2, 4, 4),
            3,
            &activation::Activation::Linear,
            (2, 2),
            (1, 1),
            (0, 0),
            None,
        );
        conv.kernels[0] = tensor::Tensor::triple(vec![
            vec![vec![1.0, 1.0], vec![2.0, 2.0]],
            vec![vec![1.0, 2.0], vec![1.0, 2.0]],
        ]);
        conv.kernels[1] = tensor::Tensor::triple(vec![
            vec![vec![2.0, 2.0], vec![1.0, 1.0]],
            vec![vec![2.0, 1.0], vec![2.0, 1.0]],
        ]);
        conv.kernels[2] = tensor::Tensor::triple(vec![
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        ]);

        let input = tensor::Tensor::triple(vec![
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 2.0, 0.0],
                vec![0.0, 3.0, 4.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ],
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 4.0, 3.0, 0.0],
                vec![0.0, 2.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ],
        ]);

        let output = tensor::Tensor::triple(vec![
            vec![
                vec![10.0, 16.0, 7.0],
                vec![19.0, 31.0, 14.0],
                vec![7.0, 11.0, 5.0],
            ],
            vec![
                vec![5.0, 14.0, 8.0],
                vec![11.0, 29.0, 16.0],
                vec![8.0, 19.0, 10.0],
            ],
            vec![
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ],
        ]);

        // Double-check to ensure the forward pass is correct.
        let (pre, post) = conv.forward(&input);
        assert_eq_data!(pre.data, output.data);
        assert_eq_data!(post.data, output.data);

        let gradient = tensor::Tensor::triple(vec![vec![vec![1.0; 3]; 3]; 3]);

        let (input_gradient, kernel_gradient, _) = conv.backward(&gradient, &input, &output);

        let _input_gradient = tensor::Tensor::triple(vec![
            vec![
                vec![3.0, 6.0, 6.0, 3.0],
                vec![6.0, 12.0, 12.0, 6.0],
                vec![6.0, 12.0, 12.0, 6.0],
                vec![3.0, 6.0, 6.0, 3.0],
            ];
            2
        ]);
        let _kernel_gradient = tensor::Tensor::quadruple(vec![vec![vec![vec![10.0; 2]; 2]; 2]; 3]);

        assert_eq_data!(input_gradient.data, _input_gradient.data);
        assert_eq_data!(kernel_gradient.data, _kernel_gradient.data);
    }

    #[test]
    fn test_rotate() {
        let conv = Convolution::create(
            tensor::Shape::Triple(1, 3, 3),
            1,
            &activation::Activation::Linear,
            (3, 3),
            (1, 1),
            (1, 1),
            None,
        );

        let data = vec![
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
        ];

        let rotated = conv.rotate(data);

        let expected = vec![
            vec![
                vec![9.0, 8.0, 7.0],
                vec![6.0, 5.0, 4.0],
                vec![3.0, 2.0, 1.0],
            ],
            vec![
                vec![9.0, 8.0, 7.0],
                vec![6.0, 5.0, 4.0],
                vec![3.0, 2.0, 1.0],
            ],
        ];

        assert_eq!(rotated, expected);
    }

    #[test]
    fn test_identity_convolution() {
        let mut conv = Convolution::create(
            tensor::Shape::Triple(1, 3, 3),
            1,
            &activation::Activation::Linear,
            (3, 3),
            (1, 1),
            (1, 1),
            None,
        );
        conv.kernels[0] = tensor::Tensor::triple(vec![vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]]);

        let input = tensor::Tensor::triple(vec![vec![
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
            tensor::Shape::Triple(1, 5, 5),
            1,
            &activation::Activation::Linear,
            (3, 3),
            (1, 1),
            (0, 0),
            None,
        );
        conv.kernels[0] = tensor::Tensor::triple(vec![vec![
            vec![-1.0, -1.0, -1.0],
            vec![-1.0, 8.0, -1.0],
            vec![-1.0, -1.0, -1.0],
        ]]);

        let input = tensor::Tensor::triple(vec![vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 2.0, 2.0, 1.0],
            vec![1.0, 2.0, 3.0, 2.0, 1.0],
            vec![1.0, 2.0, 2.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ]]);

        let (output, _) = conv.forward(&input);

        // The central pixel should have the highest value
        let central_value = output.as_triple()[0][1][1];
        assert!(central_value > 0.0);

        // The edges should be detected
        for i in 0..3 {
            for j in 0..3 {
                if i != 1 || j != 1 {
                    assert!(output.as_triple()[0][i][j] < central_value);
                }
            }
        }
    }

    #[test]
    fn test_stride_and_padding() {
        let mut conv = Convolution::create(
            tensor::Shape::Triple(1, 5, 5),
            1,
            &activation::Activation::Linear,
            (3, 3),
            (2, 2),
            (1, 1),
            None,
        );
        conv.kernels[0] = tensor::Tensor::triple(vec![vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ]]);

        let input = tensor::Tensor::triple(vec![vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0, 1.0, 3.0],
        ]]);

        let (output, _) = conv.forward(&input);

        // Check output dimensions
        assert_eq!(output.as_triple()[0].len(), 3);
        assert_eq!(output.as_triple()[0][0].len(), 3);

        // Check some values
        assert_eq!(output.as_triple()[0][0][0], 16.0); // Top-left
        assert_eq!(output.as_triple()[0][2][2], 7.0); // Bottom-right
    }
}
