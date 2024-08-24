// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::tensor;

/// A maxpool layer.
///
/// # Attributes
///
/// * `inputs` - The `tensor::Shape` of the input to the layer.
/// * `outputs` - The `tensor::Shape` of the output from the layer.
/// * `loops` - The number of loops to run the layer.
/// * `shape` - The shape of the filter.
/// * `stride` - The stride of the filter.
/// * `flatten_output` - Whether the output should be flattened.
pub struct Maxpool {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,
    pub(crate) loops: f32,

    shape: (usize, usize),
    stride: (usize, usize),

    pub flatten_output: bool,
}

impl std::fmt::Display for Maxpool {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Maxpool({} -> {}, shape: {:?}, stride: {:?}, loops: {})",
            self.inputs, self.outputs, self.shape, self.stride, self.loops
        )
    }
}

impl Maxpool {
    /// Calculates the output size of the maxpool layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The `tensor::Shape` of the input to the layer.
    /// * `shape` - The shape of the filter.
    /// * `stride` - The stride of the filter.
    ///
    /// # Returns
    ///
    /// The `tensor::Shape` of the output from the layer.
    fn calculate_output_size(
        input: &tensor::Shape,
        shape: &(usize, usize),
        stride: &(usize, usize),
    ) -> tensor::Shape {
        let input: &(usize, usize, usize) = match input {
            tensor::Shape::Vector(shape) => {
                let root = (*shape as f32).sqrt() as usize;
                &(1, root, root)
            }
            tensor::Shape::Tensor(ch, he, wi) => &(*ch, *he, *wi),
            _ => unimplemented!("Expected a dense or maxpool input shape."),
        };

        let height = (input.1 - shape.0) / stride.0 + 1;
        let width = (input.2 - shape.1) / stride.1 + 1;

        tensor::Shape::Tensor(input.0, height, width)
    }

    /// Creates a new maxpool layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The `tensor::Shape` of the input to the layer.
    /// * `shape` - The shape of the filter.
    /// * `stride` - The stride of the filter.
    ///
    /// # Returns
    ///
    /// The new maxpool layer.
    pub fn create(input: tensor::Shape, shape: (usize, usize), stride: (usize, usize)) -> Self {
        Maxpool {
            inputs: input.clone(),
            outputs: Maxpool::calculate_output_size(&input, &shape, &stride),
            shape,
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
        // Extracting the data from the input Tensor.
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
            for h in (0..ih - self.shape.0 + 1).step_by(self.stride.0) {
                for w in (0..iw - self.shape.1 + 1).step_by(self.stride.1) {
                    let mut value = f32::MIN;
                    let mut index = (0, 0);
                    for k in 0..self.shape.0 {
                        for l in 0..self.shape.1 {
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
                    let _h = h * self.stride.0 + mh;
                    let _w = w * self.stride.1 + mw;
                    igradient[c][_h][_w] += ogradient[c][h][w];
                }
            }
        }

        tensor::Tensor::from(igradient)
    }
}
