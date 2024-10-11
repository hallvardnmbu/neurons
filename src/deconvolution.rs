// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::{activation, tensor};

use std::sync::Arc;

/// A deconvolutional layer.
///
/// # Attributes
///
/// * `inputs` - The `tensor::Shape` of the input to the layer.
/// * `outputs` - The `tensor::Shape` of the output from the layer.
/// * `loops` - The number of loops to run the layer.
/// * `scale` - The scaling function of the loops. Default is `1.0 / x`.
/// * `kernels` - The kernels of the layer.
/// * `stride` - The stride of the filter.
/// * `padding` - The padding applied to the input before deconvolving.
/// * `activation` - The `activation::Function` of the layer.
/// * `dropout` - The dropout rate of the layer (when training).
/// * `flatten` - Whether the output should be flattened.
/// * `training` - Whether the layer is training.
#[derive(Clone)]
pub struct Deconvolution {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,

    pub(crate) loops: f32,
    pub(crate) scale: tensor::Scale,

    pub(crate) kernels: Vec<tensor::Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),

    pub(crate) activation: activation::Function,

    dropout: Option<f32>,
    pub(crate) flatten: bool,
    pub(crate) training: bool,
}

impl std::fmt::Display for Deconvolution {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Deconvolution{} (\n", self.activation)?;
        write!(f, "\t\t\t{} -> {}\n", self.inputs, self.outputs)?;
        write!(
            f,
            "\t\t\tkernel: {}x({})\n",
            self.kernels.len(),
            self.kernels[0].shape
        )?;
        write!(f, "\t\t\tstride: {:?}\n", self.stride)?;
        write!(f, "\t\t\tpadding: {:?}\n", self.padding)?;
        if self.dropout.is_some() {
            write!(f, "\t\t\tdropout: {}\n", self.dropout.unwrap().to_string())?;
        }
        if self.loops > 1.0 {
            write!(
                f,
                "\t\t\tloops: {} (scaling factor: {})\n",
                self.loops,
                (self.scale)(self.loops)
            )?;
        }
        write!(f, "\t\t)")?;
        Ok(())
    }
}

impl Deconvolution {
    /// Calculates the output size of the deconvolutional layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The `tensor::Shape` of the input to the layer.
    /// * `channels` - The number of output channels from the layer (i.e., number of filters).
    /// * `kernel` - The size of each filter.
    /// * `stride` - The stride of the filter.
    /// * `padding` - The padding applied to the input before deconvolving.
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

        let height = (input.1 - 1) * stride.0 + kernel.0 - 2 * padding.0;
        let width = (input.2 - 1) * stride.1 + kernel.1 - 2 * padding.1;

        tensor::Shape::Triple(*channels, height, width)
    }

    /// Creates a new deconvolutional layer with randomized kernel weights.
    ///
    /// # Arguments
    ///
    /// * `input` - The `tensor::Shape` of the input to the layer.
    /// * `filters` - The number of output channels from the layer.
    /// * `activation` - The `activation::Activation` function of the layer.
    /// * `kernel` - The size of each filter.
    /// * `stride` - The stride of the filter.
    /// * `padding` - The padding applied to the input before deconvolving.
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
                    panic!("> When adding a deconvolutional layer after a dense layer, the dense layer must have a square output.\n> Currently, the layer has {} outputs, which cannot cannot be reshaped to a (1, root[{}], root[{}]) tensor.\n> Try using {} or {} outputs for the preceding dense layer.", size, size, size, root*root, (root+1)*(root+1));
                }
            }
            tensor::Shape::Triple(ic, _, _) => (inputs.clone(), *ic),
            _ => unimplemented!("Expected a `tensor::Tensor` input shape."),
        };
        let outputs = Self::calculate_output_size(&inputs, &filters, &kernel, &stride, &padding);
        Self {
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
            scale: Arc::new(|x| 1.0 / x),
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

    /// Applies the forward pass (deconvolution) to the input `tensor::Tensor`.
    /// Assumes `x` to match `self.inputs`, and for performance reasons does not check.
    ///
    /// # Arguments
    ///
    /// * `x` - The input `tensor::Tensor` to the layer.
    ///
    /// # Returns
    ///
    /// The pre- and post-activation `tensor::Tensor`s of the deconvolved input wrt. the kernels.
    pub fn forward(&self, x: &tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {
        // Extracting the data from the input `tensor::Tensor`.
        let x = match &x.data {
            tensor::Data::Single(vector) => {
                let (h, w) = match &self.inputs {
                    tensor::Shape::Triple(_, h, w) => (*h, *w),
                    _ => panic!("Convolutional layers should have `tensor::Shape::Triple` input."),
                };
                vector
                    .chunks_exact(h * w)
                    .map(|channel| channel.chunks_exact(w).map(|row| row.to_vec()).collect())
                    .collect()
            }
            tensor::Data::Triple(tensor) => tensor.clone(),
            _ => panic!("Unexpected input data type."),
        };

        // Extracting the weights from the kernels.
        let kernels: Vec<&Vec<Vec<Vec<f32>>>> = self
            .kernels
            .iter()
            .map(|ref k| match k.data {
                tensor::Data::Triple(ref kernel) => kernel,
                _ => panic!("Expected `tensor::Shape::Triple` kernel shape."),
            })
            .collect();

        let (ih, iw) = (x[0].len(), x[0][0].len());
        let (kf, kc, kh, kw) = (
            kernels.len(),
            kernels[0].len(),
            kernels[0][0].len(),
            kernels[0][0][0].len(),
        );

        // Defining the output dimensions and vector.
        let oh = (ih - 1) * self.stride.0 - 2 * self.padding.0 + kh;
        let ow = (iw - 1) * self.stride.1 - 2 * self.padding.1 + kw;
        let mut y = vec![vec![vec![0.0; ow]; oh]; kf];

        // Deconvolving the input with the kernels.
        for k in 0..kf {
            for c in 0..kc {
                for i in 0..ih {
                    for j in 0..iw {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let oi = i * self.stride.0 + ki;
                                let oi = match oi.checked_sub(self.padding.0) {
                                    Some(value) => value,
                                    None => {
                                        continue;
                                    }
                                };

                                let oj = j * self.stride.1 + kj;
                                let oj = match oj.checked_sub(self.padding.1) {
                                    Some(value) => value,
                                    None => {
                                        continue;
                                    }
                                };

                                if oi < oh && oj < ow {
                                    y[k][oi][oj] += x[c][i][j] * kernels[k][c][ki][kj];
                                }
                            }
                        }
                    }
                }
            }
        }

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
    pub fn backward(
        &self,
        gradient: &tensor::Tensor,
        input: &tensor::Tensor,
        output: &tensor::Tensor,
    ) -> (tensor::Tensor, tensor::Tensor, Option<tensor::Tensor>) {
        let gradient = gradient.get_triple(&self.outputs);
        let derivative = self.activation.backward(&output).get_triple(&self.outputs);
        let delta = tensor::hadamard3d(&gradient, &derivative, (self.scale)(self.loops));

        // Extracting the input and its dimensions.
        let input = input.get_triple(&self.inputs);
        let (ih, iw) = (input[0].len(), input[0][0].len());
        let (oh, ow) = (delta[0].len(), delta[0][0].len());

        // Extracting the kernel and its dimensions.
        let kernels: Vec<Vec<Vec<Vec<f32>>>> = self
            .kernels
            .iter()
            .map(|k| match &k.data {
                tensor::Data::Triple(ref kernel) => kernel.clone(),
                _ => panic!("Expected `Tensor` kernel data."),
            })
            .collect();

        let (kf, kc, kh, kw) = (
            kernels.len(),
            kernels[0].len(),
            kernels[0][0].len(),
            kernels[0][0][0].len(),
        );

        let mut kgradient = vec![vec![vec![vec![0.0; kw]; kh]; kc]; kf];
        let mut igradient = vec![vec![vec![0.0; iw]; ih]; kc];

        for f in 0..kf {
            for c in 0..kc {
                for h in 0..ih {
                    for w in 0..iw {
                        for i in 0..kh {
                            for j in 0..kw {
                                let oi = h * self.stride.0 + i;
                                let oi = match oi.checked_sub(self.padding.0) {
                                    Some(value) => value,
                                    None => {
                                        continue;
                                    }
                                };

                                let oj = w * self.stride.1 + j;
                                let oj = match oj.checked_sub(self.padding.1) {
                                    Some(value) => value,
                                    None => {
                                        continue;
                                    }
                                };

                                if oi < oh && oj < ow {
                                    igradient[c][h][w] += delta[f][oi][oj] * kernels[f][c][i][j];
                                    kgradient[f][c][i][j] += delta[f][oi][oj] * input[c][h][w];
                                }
                            }
                        }
                    }
                }
            }
        }

        (
            tensor::Tensor::triple(igradient),
            tensor::Tensor::quadruple(kgradient),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::tensor::{Shape, Tensor};
    use crate::{assert_eq_data, assert_eq_shape};

    #[test]
    fn test_create_deconvolution_layer() {
        let inputs = Shape::Triple(1, 2, 2);
        let filters = 2;
        let activation = Activation::ReLU;
        let kernel = (2, 2);
        let stride = (2, 2);
        let padding = (0, 0);
        let dropout = Some(0.5);

        let layer = Deconvolution::create(
            inputs.clone(),
            filters,
            &activation,
            kernel,
            stride,
            padding,
            dropout,
        );

        assert_eq!(layer.inputs, inputs);
        assert_eq!(layer.outputs, Shape::Triple(filters, 4, 4));
        assert_eq!(layer.kernels.len(), filters);
        assert_eq!(layer.kernels[0].shape, Shape::Triple(1, 2, 2));
        assert_eq!(layer.dropout, dropout);
        assert_eq!(layer.stride, stride);
        assert_eq!(layer.padding, padding);
    }

    #[test]
    fn test_forward_pass() {
        let inputs = Shape::Triple(1, 2, 2);
        let filters = 1;
        let activation = Activation::ReLU;
        let kernel = (2, 2);
        let stride = (2, 2);
        let padding = (0, 0);
        let dropout = Some(0.5);

        let mut layer = Deconvolution::create(
            inputs.clone(),
            filters,
            &activation,
            kernel,
            stride,
            padding,
            dropout,
        );

        layer.kernels = vec![Tensor::triple(vec![vec![vec![1.0, 4.0], vec![2.0, 3.0]]])];

        let input = Tensor::triple(vec![vec![vec![0.0, 1.0], vec![2.0, 3.0]]]);

        let (pre, post) = layer.forward(&input);

        assert_eq_shape!(pre.shape, layer.outputs);

        assert_eq!(pre.shape, Shape::Triple(1, 4, 4));
        assert_eq!(post.shape, Shape::Triple(1, 4, 4));

        let output = Tensor::triple(vec![vec![
            vec![0.0, 0.0, 1.0, 4.0],
            vec![0.0, 0.0, 2.0, 3.0],
            vec![2.0, 8.0, 3.0, 12.0],
            vec![4.0, 6.0, 6.0, 9.0],
        ]]);

        assert_eq_data!(post.data, output.data);
    }

    #[test]
    fn test_backward_pass() {
        let input_shape = Shape::Triple(1, 4, 4);
        let filters = 1;
        let activation = Activation::Linear;
        let kernel = (3, 3);
        let stride = (2, 2);
        let padding = (1, 1);
        let dropout = None;

        let layer = Deconvolution::create(
            input_shape.clone(),
            filters,
            &activation,
            kernel,
            stride,
            padding,
            dropout,
        );

        let input = Tensor::triple(vec![vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ]]);

        let grad = Tensor::triple(vec![vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            vec![0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            vec![1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3],
            vec![2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
            vec![3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9],
            vec![4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7],
            vec![4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
        ]]);

        let (_pre, post) = layer.forward(&input);
        let (igrad, wgrad, _) = layer.backward(&grad, &input, &post);

        assert_eq!(igrad.shape, input_shape);
        assert_eq!(wgrad.shape, Shape::Quadruple(1, 1, 3, 3));

        let kgrad: Vec<Vec<Vec<f32>>> = vec![vec![
            vec![316.8000, 407.0000, 291.6000],
            vec![400.0000, 512.8000, 366.4000],
            vec![216.0000, 272.6000, 190.8000],
        ]];
        let wdata = match wgrad.data {
            crate::tensor::Data::Quadruple(data) => data[0].clone(),
            _ => panic!("Invalid data type"),
        };

        for (k, w) in kgrad.iter().zip(wdata.iter()) {
            for (k, w) in k.iter().zip(w.iter()) {
                for (k, w) in k.iter().zip(w.iter()) {
                    assert!((k - w).abs() < 1e-4);
                }
            }
        }
    }
}
