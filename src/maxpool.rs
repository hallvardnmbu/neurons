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
/// * `flatten` - Whether the output should be flattened.
pub struct Maxpool {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,
    pub(crate) loops: f32,

    kernel: (usize, usize),
    stride: (usize, usize),

    pub(crate) flatten: bool,
}

impl std::fmt::Display for Maxpool {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Maxpool(\n")?;
        write!(f, "\t\t\t{} -> {}\n", self.inputs, self.outputs)?;
        write!(f, "\t\t\tkernel: {:?}\n", self.kernel)?;
        write!(f, "\t\t\tstride: {:?}\n", self.stride)?;
        write!(f, "\t\t\tloops: {}\n", self.loops)?;
        write!(f, "\t\t)")?;
        Ok(())
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
            tensor::Shape::Single(size) => {
                let root = (*size as f32).sqrt() as usize;
                &(1, root, root)
            }
            tensor::Shape::Triple(ch, he, wi) => &(*ch, *he, *wi),
            _ => unimplemented!("Expected a `tensor::Tensor` input shape."),
        };

        let height = (input.1 - kernel.0) / stride.0 + 1;
        let width = (input.2 - kernel.1) / stride.1 + 1;

        tensor::Shape::Triple(input.0, height, width)
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
    pub fn create(inputs: tensor::Shape, kernel: (usize, usize), stride: (usize, usize)) -> Self {
        let inputs = match &inputs {
            tensor::Shape::Single(size) => {
                let root = (*size as f32).sqrt() as usize;
                if size % root == 0 {
                    tensor::Shape::Triple(1, root, root)
                } else {
                    panic!("> When adding a maxpool layer after a dense layer, the dense layer must have a square output.\n> Currently, the layer has {} outputs, which cannot cannot be reshaped to a (1, root[{}], root[{}]) tensor.\n> Try using {} or {} outputs for the preceding dense layer.", size, size, size, root*root, (root+1)*(root+1));
                }
            }
            tensor::Shape::Triple(_, _, _) => inputs,
            _ => unimplemented!("Expected a `tensor::Tensor` input shape."),
        };
        let outputs = Maxpool::calculate_output_size(&inputs, &kernel, &stride);
        Maxpool {
            inputs,
            outputs,
            kernel,
            stride,
            flatten: false,
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
            tensor::Data::Single(vector) => {
                let (h, w) = match &self.inputs {
                    tensor::Shape::Triple(_, h, w) => (*h, *w),
                    _ => panic!("Maxpool layers should have `tensor::Shape::Triple` input."),
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
        let (oc, oh, ow) = match &self.outputs {
            tensor::Shape::Triple(oc, oh, ow) => (*oc, *oh, *ow),
            _ => panic!("Expected `tensor::Shape::Triple` output shape."),
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
                                let _x = x[c][_dh][_dw];
                                if _x > value {
                                    value = _x;
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

        let pre = tensor::Tensor::triple(y);
        let mut post = pre.clone();

        if self.flatten {
            post = post.flatten();
        }

        (pre, post, max)
    }

    /// Applies the backward pass of the layer to the gradient `tensor::Tensor`.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient `tensor::Tensor` to the layer.
    /// * `max` - The indices of the max values from the forward pass.
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
            tensor::Shape::Triple(ic, ih, iw) => (*ic, *ih, *iw),
            _ => panic!("Expected `tensor::Shape::Triple` input shape."),
        };

        let ogradient = gradient.get_triple(&self.outputs);
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

        tensor::Tensor::triple(igradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Data;
    use crate::{assert_eq_data, assert_eq_shape};

    #[test]
    fn test_calculate_output_size() {
        let input = tensor::Shape::Triple(1, 4, 4);
        let kernel = (2, 2);
        let stride = (2, 2);

        let output = Maxpool::calculate_output_size(&input, &kernel, &stride);

        assert_eq_shape!(output, tensor::Shape::Triple(1, 2, 2));
    }

    #[test]
    fn test_create() {
        let input = tensor::Shape::Triple(1, 4, 4);
        let kernel = (2, 2);
        let stride = (2, 2);

        let maxpool = Maxpool::create(input.clone(), kernel, stride);

        assert_eq_shape!(maxpool.inputs, input);
        assert_eq_shape!(maxpool.outputs, tensor::Shape::Triple(1, 2, 2));
        assert_eq!(maxpool.kernel, kernel);
        assert_eq!(maxpool.stride, stride);
        assert_eq!(maxpool.flatten, false);
        assert_eq!(maxpool.loops, 1.0);
    }

    #[test]
    fn test_forward() {
        let input = tensor::Shape::Triple(1, 4, 4);
        let kernel = (2, 2);
        let stride = (2, 2);
        let maxpool = Maxpool::create(input.clone(), kernel, stride);

        let x = tensor::Tensor {
            data: Data::Triple(vec![vec![
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
            Data::Triple(vec![vec![vec![6.0, 8.0], vec![14.0, 16.0]]])
        );
        assert_eq_data!(
            post.data,
            Data::Triple(vec![vec![vec![6.0, 8.0], vec![14.0, 16.0]]])
        );
        assert_eq!(max, vec![vec![vec![(1, 1), (1, 3)], vec![(3, 1), (3, 3)]]]);
    }

    #[test]
    fn test_backward() {
        let input = tensor::Shape::Triple(1, 4, 4);
        let kernel = (2, 2);
        let stride = (2, 2);
        let maxpool = Maxpool::create(input.clone(), kernel, stride);

        let gradient = tensor::Tensor {
            data: Data::Triple(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]),
            shape: tensor::Shape::Triple(1, 2, 2),
        };

        let max = vec![vec![vec![(1, 1), (1, 3)], vec![(3, 1), (3, 3)]]];

        let igradient = maxpool.backward(&gradient, &max);

        assert_eq_data!(
            igradient.data,
            Data::Triple(vec![vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 2.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 3.0, 0.0, 4.0],
            ],])
        );
    }
}
