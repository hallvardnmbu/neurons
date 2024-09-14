// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::{activation, tensor};

use std::sync::Arc;

/// A dense layer.
///
/// # Attributes
///
/// * `inputs` - The number of inputs to the layer.
/// * `outputs` - The number of outputs from the layer.
/// * `loops` - The number of loops, i.e., feedback connections.
/// * `scale` - The scaling function of the loops. Default is `1.0 / x`.
/// * `weights` - The weights of the layer.
/// * `bias` - The bias of the layer.
/// * `activation` - The `activation::Function` of the layer.
/// * `dropout` - The dropout rate of the layer (when training).
/// * `training` - Whether the network is training.
#[derive(Clone)]
pub struct Dense {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,

    pub(crate) loops: f32,
    pub(crate) scale: tensor::Scale,

    pub(crate) weights: tensor::Tensor,
    pub(crate) bias: Option<tensor::Tensor>,

    pub(crate) activation: activation::Function,

    dropout: Option<f32>,
    pub(crate) training: bool,
}

impl std::fmt::Display for Dense {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Dense{}(\n", self.activation)?;
        write!(f, "\t\t\t{} -> {}\n", self.inputs, self.outputs)?;
        write!(f, "\t\t\tbias: {}\n", self.bias.is_some())?;
        write!(
            f,
            "\t\t\tdropout: {}\n",
            if self.dropout.is_some() {
                self.dropout.unwrap().to_string()
            } else {
                "false".to_string()
            }
        )?;
        write!(
            f,
            "\t\t\tloops: {} (scaling factor: {})\n",
            self.loops,
            (self.scale)(self.loops)
        )?;
        write!(f, "\t\t)")?;
        Ok(())
    }
}

impl Dense {
    /// Creates a new dense layer with randomized weights and bias.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The number of inputs to the layer.
    /// * `outputs` - The number of outputs from the layer.
    /// * `activation` - The `activation::Activation` function of the layer.
    /// * `bias` - Whether the layer should have a bias.
    /// * `dropout` - The dropout rate of the layer (when training).
    ///
    /// # Returns
    ///
    /// A new layer with random weights and bias with the given dimensions.
    pub fn create(
        inputs: tensor::Shape,
        outputs: tensor::Shape,
        activation: &activation::Activation,
        bias: bool,
        dropout: Option<f32>,
    ) -> Self {
        let (input, output) = match (&inputs, &outputs) {
            (tensor::Shape::Single(input), tensor::Shape::Single(output)) => (*input, *output),
            _ => panic!("Invalid input- and output shape."),
        };
        Dense {
            inputs,
            outputs,
            loops: 1.0,
            scale: Arc::new(|x| 1.0 / x),
            weights: tensor::Tensor::random(tensor::Shape::Double(output, input), -1.0, 1.0),
            bias: match bias {
                true => Some(tensor::Tensor::random(
                    tensor::Shape::Single(output),
                    -1.0,
                    1.0,
                )),
                false => None,
            },
            activation: activation::Function::create(&activation),
            dropout,
            training: false,
        }
    }

    /// Extract the number of parameters in the layer.
    pub fn parameters(&self) -> usize {
        let (input, output) = match (&self.inputs, &self.outputs) {
            (tensor::Shape::Single(input), tensor::Shape::Single(output)) => (*input, *output),
            _ => panic!("Invalid input- and output shape."),
        };
        input * output + if self.bias.is_some() { output } else { 0 }
    }

    /// Applies the forward pass of the layer to the input `tensor::Tensor`.
    ///
    /// # Arguments
    ///
    /// * `x` - The input `tensor::Tensor` to the layer.
    ///
    /// # Returns
    ///
    /// The pre-activation and post-activation `tensor::Tensor`s of the layer.
    pub fn forward(&self, x: &tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {
        let mut pre = self.weights.dot(x);
        if let Some(bias) = &self.bias {
            pre.add_inplace(bias);
        }

        let mut post = self.activation.forward(&pre);

        // Apply dropout if the network is training.
        if self.training {
            if let Some(dropout) = self.dropout {
                post.dropout(dropout);
            }
        }

        (pre, post)
    }

    /// Applies the backward pass of the layer to the gradient vector.
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
        let gradient = match &gradient.shape {
            tensor::Shape::Single(_) => gradient,
            tensor::Shape::Triple(_, _, _) => &gradient.flatten(),
            _ => panic!("Invalid gradient shape."),
        };
        let derivative = self.activation.backward(output);
        let delta = derivative.hadamard(gradient, (self.scale)(self.loops));

        let weight_gradient = delta.product(input);

        let bias_gradient: Option<tensor::Tensor> = match self.bias {
            Some(_) => Some(delta.clone()),
            None => None,
        };

        let input_gradient = self.weights.transpose().dot(&delta);

        (input_gradient, weight_gradient, bias_gradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{activation::Activation, assert_eq_data, assert_eq_shape, tensor};

    #[test]
    fn test_create() {
        let dense = Dense::create(
            tensor::Shape::Single(3),
            tensor::Shape::Single(2),
            &Activation::Linear,
            true,
            Some(0.5),
        );

        assert_eq_shape!(dense.weights.shape, tensor::Shape::Double(2, 3));
        if let Some(bias) = &dense.bias {
            assert_eq_shape!(bias.shape, tensor::Shape::Single(2));
        }
        assert_eq!(dense.dropout, Some(0.5));
    }

    #[test]
    fn test_forward() {
        let mut dense = Dense::create(
            tensor::Shape::Single(3),
            tensor::Shape::Single(2),
            &Activation::Linear,
            true,
            Some(0.5),
        );
        dense.weights = tensor::Tensor::double(vec![vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]]);
        dense.bias = Some(tensor::Tensor::single(vec![0.0, 0.0]));

        let input = tensor::Tensor::single(vec![1.0, 2.0, 3.0]);
        let (pre, post) = dense.forward(&input);

        assert_eq_data!(pre.data, tensor::Data::Single(vec![3.0, 3.0]));
        assert_eq_data!(post.data, tensor::Data::Single(vec![3.0, 3.0]));
    }

    #[test]
    fn test_backward() {
        let mut dense = Dense::create(
            tensor::Shape::Single(3),
            tensor::Shape::Single(2),
            &Activation::Linear,
            true,
            Some(0.5),
        );
        dense.weights = tensor::Tensor::double(vec![vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]]);
        dense.bias = Some(tensor::Tensor::single(vec![0.0, 0.0]));

        let input = tensor::Tensor::single(vec![1.0, 2.0, 3.0]);
        let (_, post) = dense.forward(&input);
        let gradient = tensor::Tensor::single(vec![1.0, 1.0]);

        let (input_gradient, weight_gradient, bias_gradient) =
            dense.backward(&gradient, &input, &post);

        assert_eq_data!(
            input_gradient.data,
            tensor::Data::Single(vec![1.0, 1.0, 1.0])
        );
        assert_eq_data!(
            weight_gradient.data,
            tensor::Data::Double(vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]])
        );
        if let Some(bias_gradient) = bias_gradient {
            assert_eq_data!(bias_gradient.data, tensor::Data::Single(vec![1.0, 1.0]));
        }
    }
}
