// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::tensor;

/// A maxpool layer.
///
/// # Attributes
///
/// * `inputs` - The `tensor::Shape` of the input to the layer.
/// * `outputs` - The `tensor::Shape` of the output from the layer.
/// * `loops` - The number of loops to run the layer.
/// * `kernel` - The shape of the filter.
/// * `stride` - The stride of the filter.
/// * `flatten_output` - Whether the output should be flattened.
pub struct Maxpool {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,
    pub(crate) loops: f32,

    kernel: (usize, usize),
    stride: (usize, usize),

    pub flatten_output: bool,
}

impl std::fmt::Display for Maxpool {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Maxpool({} -> {}, kernel: {:?}, stride: {:?}, loops: {})",
            self.inputs, self.outputs, self.kernel, self.stride, self.loops
        )
    }
}

impl Maxpool {
    /// Calculates the output size of the maxpool layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The `tensor::Shape` of the input to the layer.
    /// * `kernel` - The shape of the filter.
    /// * `stride` - The stride of the filter.
    ///
    /// # Returns
    ///
    /// The `tensor::Shape` of the output from the layer.
    fn calculate_output_size(
        input: &tensor::Shape,
        kernel: &(usize, usize),
        stride: &(usize, usize),
    ) -> tensor::Shape {
        let input: &(usize, usize, usize) = match input {
            tensor::Shape::Tensor(ch, he, wi) => &(*ch, *he, *wi),
            _ => unimplemented!("Expected a `tensor::Tensor` input shape."),
        };

        let height = (input.1 - kernel.0) / stride.0 + 1;
        let width = (input.2 - kernel.1) / stride.1 + 1;

        tensor::Shape::Tensor(input.0, height, width)
    }

    /// Creates a new maxpool layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The `tensor::Shape` of the input to the layer.
    /// * `kernel` - The shape of the filter.
    /// * `stride` - The stride of the filter.
    ///
    /// # Returns
    ///
    /// The new maxpool layer.
    pub fn create(input: tensor::Shape, kernel: (usize, usize), stride: (usize, usize)) -> Self {
        Maxpool {
            inputs: input.clone(),
            outputs: Maxpool::calculate_output_size(&input, &kernel, &stride),
            kernel,
            stride,
            flatten_output: false,
            loops: 1.0,
        }
    }

    /// Applies the forward pass (maxpool) to the input `tensor::Tensor`.
    /// Assumes `x` to match `self.inputs`, and for performance reasons does not check.
    ///
    /// # Arguments
    ///
    /// * `x` - The input `tensor::Tensor` to the layer.
    ///
    /// # Returns
    ///
    /// The pre-activation and post-activation `tensor::Tensor`s of the convolved input wrt. the kernels.
    pub fn forward(
        &self,
        x: &tensor::Tensor,
    ) -> (
        tensor::Tensor,
        tensor::Tensor,
        Vec<Vec<Vec<(usize, usize)>>>,
    ) {
        // Extracting the data from the input `tensor::Tensor`.
        let (x, ih, iw) = match &x.data {
            tensor::Data::Vector(vector) => {
                let root = (vector.len() as f32).sqrt() as usize;
                (
                    vec![vector.chunks(root).map(|v| v.to_vec()).collect()],
                    root,
                    root,
                )
            }
            tensor::Data::Tensor(tensor) => (tensor.clone(), tensor[0].len(), tensor[0][0].len()),
            _ => panic!("Expected `Vector` or `Tensor` input data."),
        };
        let (oc, oh, ow) = match &self.outputs {
            tensor::Shape::Tensor(channels, height, width) => (*channels, *height, *width),
            _ => panic!("Expected `Tensor` output shape."),
        };

        let mut y = vec![vec![vec![0.0; ow]; oh]; oc];
        let mut max = vec![vec![vec![(0, 0); ow]; oh]; oc];
        for c in 0..oc {
            for h in (0..ih - self.kernel.0 + 1).step_by(self.stride.0) {
                for w in (0..iw - self.kernel.1 + 1).step_by(self.stride.1) {
                    let mut value = f32::MIN;
                    let mut index = (0, 0);
                    for k in 0..self.kernel.0 {
                        for l in 0..self.kernel.1 {
                            let _dh = h + k;
                            let _dw = w + l;
                            if _dh < ih && _dw < iw {
                                let x = x[c][_dh][_dw];
                                if x > value {
                                    value = x;
                                    index = (_dh, _dw);
                                }
                            }
                        }
                    }
                    let h = h / self.stride.0;
                    let w = w / self.stride.1;
                    y[c][h][w] = value;
                    max[c][h][w] = index;
                }
            }
        }

        let pre = tensor::Tensor::from(y);
        let mut post = pre.clone();

        if self.flatten_output {
            post = post.flatten();
        }

        (pre, post, max)
    }

    /// Applies the backward pass of the layer to the gradient `tensor::Tensor`.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient `tensor::Tensor` to the layer.
    /// * `max` - The indices of the max values in the forward pass.
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
        max: &Vec<Vec<Vec<(usize, usize)>>>,
    ) -> tensor::Tensor {
        let (ic, ih, iw) = match &self.inputs {
            tensor::Shape::Tensor(channels, height, width) => (*channels, *height, *width),
            _ => panic!("Expected `Tensor` input shape."),
        };

        let ogradient = match &gradient.data {
            tensor::Data::Tensor(tensor) => tensor.clone(),
            _ => panic!("Expected `Tensor` gradient data."),
        };
        let mut igradient = vec![vec![vec![0.0; iw]; ih]; ic];

        let (oh, ow) = (ogradient[0].len(), ogradient[0][0].len());
        for c in 0..ic {
            for h in 0..oh {
                for w in 0..ow {
                    let (mh, mw) = max[c][h][w];
                    igradient[c][mh][mw] += ogradient[c][h][w];
                }
            }
        }

        tensor::Tensor::from(igradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Data;
    use crate::{assert_eq_data, assert_eq_shape};

    #[test]
    fn test_calculate_output_size() {
        let input = tensor::Shape::Tensor(1, 4, 4);
        let kernel = (2, 2);
        let stride = (2, 2);

        let output = Maxpool::calculate_output_size(&input, &kernel, &stride);

        assert_eq_shape!(output, tensor::Shape::Tensor(1, 2, 2));
    }

    #[test]
    fn test_create() {
        let input = tensor::Shape::Tensor(1, 4, 4);
        let kernel = (2, 2);
        let stride = (2, 2);

        let maxpool = Maxpool::create(input.clone(), kernel, stride);

        assert_eq_shape!(maxpool.inputs, input);
        assert_eq_shape!(maxpool.outputs, tensor::Shape::Tensor(1, 2, 2));
        assert_eq!(maxpool.kernel, kernel);
        assert_eq!(maxpool.stride, stride);
        assert_eq!(maxpool.flatten_output, false);
        assert_eq!(maxpool.loops, 1.0);
    }

    #[test]
    fn test_forward() {
        let input = tensor::Shape::Tensor(1, 4, 4);
        let kernel = (2, 2);
        let stride = (2, 2);
        let maxpool = Maxpool::create(input.clone(), kernel, stride);

        let x = tensor::Tensor {
            data: Data::Tensor(vec![vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![9.0, 10.0, 11.0, 12.0],
                vec![13.0, 14.0, 15.0, 16.0],
            ]]),
            shape: input,
        };

        let (pre, post, max) = maxpool.forward(&x);

        assert_eq_data!(
            pre.data,
            Data::Tensor(vec![vec![vec![6.0, 8.0], vec![14.0, 16.0]]])
        );
        assert_eq_data!(
            post.data,
            Data::Tensor(vec![vec![vec![6.0, 8.0], vec![14.0, 16.0]]])
        );
        assert_eq!(max, vec![vec![vec![(1, 1), (1, 3)], vec![(3, 1), (3, 3)]]]);
    }

    #[test]
    fn test_backward() {
        let input = tensor::Shape::Tensor(1, 4, 4);
        let kernel = (2, 2);
        let stride = (2, 2);
        let maxpool = Maxpool::create(input.clone(), kernel, stride);

        let gradient = tensor::Tensor {
            data: Data::Tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]),
            shape: tensor::Shape::Tensor(1, 2, 2),
        };

        let max = vec![vec![vec![(1, 1), (1, 3)], vec![(3, 1), (3, 3)]]];

        let igradient = maxpool.backward(&gradient, &max);

        assert_eq_data!(
            igradient.data,
            Data::Tensor(vec![vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 2.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 3.0, 0.0, 4.0],
            ],])
        );
    }
}
