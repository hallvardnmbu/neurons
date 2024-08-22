// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::activation;
use crate::algebra::*;
use crate::{random, tensor};

/// A dense layer in a neural network.
///
/// # Attributes
///
/// * `inputs` - The number of inputs to the layer.
/// * `outputs` - The number of outputs from the layer.
/// * `loops` - The number of loops, i.e., feedback connections.
/// * `weights` - The weights of the layer.
/// * `bias` - The bias of the layer.
/// * `activation` - The activation function of the layer.
/// * `dropout` - The dropout rate of the layer (when training).
/// * `training` - Whether the network is training.
pub struct Dense {
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,

    pub(crate) loops: f32,

    pub(crate) weights: Vec<Vec<f32>>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) activation: activation::Function,

    pub(crate) dropout: Option<f32>,

    pub(crate) training: bool,
}

impl std::fmt::Display for Dense {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Dense{}({} -> {}, bias: {}, dropout: {}, loops: {})",
            self.activation,
            self.inputs,
            self.outputs,
            self.bias.is_some(),
            if let Some(dropout) = self.dropout {
                dropout.to_string()
            } else {
                "false".to_string()
            },
            self.loops
        )
    }
}

impl Dense {
    /// Creates a new dense layer with random weights and bias.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The number of inputs to the layer.
    /// * `outputs` - The number of outputs from the layer.
    /// * `activation` - The activation function of the layer.
    /// * `bias` - Whether the layer should have a bias.
    /// * `dropout` - The dropout rate of the layer (when training).
    ///
    /// # Returns
    ///
    /// A new layer with random weights and bias with the given dimensions.
    pub fn create(
        inputs: usize,
        outputs: usize,
        activation: &activation::Activation,
        bias: bool,
        dropout: Option<f32>,
    ) -> Self {
        let mut generator = random::Generator::create(12345);
        Dense {
            inputs,
            outputs,
            loops: 1.0,
            weights: (0..outputs)
                .map(|_| (0..inputs).map(|_| generator.generate(-1.0, 1.0)).collect())
                .collect(),
            bias: match bias {
                true => Some(
                    (0..outputs)
                        .map(|_| generator.generate(-1.0, 1.0))
                        .collect(),
                ),
                false => None,
            },
            activation: activation::Function::create(&activation),
            dropout,
            training: false,
        }
    }

    pub fn parameters(&self) -> usize {
        self.inputs * self.outputs + if self.bias.is_some() { self.outputs } else { 0 }
    }

    /// Applies the forward pass of the layer to the input Tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input Tensor to the layer.
    ///
    /// # Returns
    ///
    /// The pre-activation and post-activation Tensors of the layer.
    pub fn forward(&self, x: &tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {
        let x = x.get_flat();

        let pre = tensor::Tensor::from_single(self.weights.iter().map(|w| dot(&w, &x)).collect());
        let mut post = match &self.bias {
            Some(bias) => self
                .activation
                .forward(&pre)
                .add(&tensor::Tensor::from_single(bias.clone())),
            None => self.activation.forward(&pre),
        };

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
        let derivative: Vec<f32> = self.activation.backward(&output).get_flat();

        // TODO: Should the loops be multiplied as-is? 1/loops? sqrt(loops)? etc.
        // Iris; better with *loops than /loops => Converges faster.
        let delta: Vec<f32> = mul_scalar(&mul(&gradient.get_flat(), &derivative), self.loops);

        let weight_gradient: Vec<Vec<f32>> = delta
            .iter()
            .map(|d: &f32| input.get_flat().iter().map(|i: &f32| i * d).collect())
            .collect();
        let bias_gradient: Option<tensor::Tensor> = match self.bias {
            Some(_) => Some(tensor::Tensor::from_single(delta.clone())),
            None => None,
        };
        let input_gradient: Vec<f32> = (0..input.get_flat().len())
            .map(|i: usize| {
                delta
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(d, w)| d * w[i])
                    .sum::<f32>()
            })
            .collect();

        (
            tensor::Tensor::from_single(input_gradient),
            tensor::Tensor::from(vec![weight_gradient]),
            bias_gradient,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;

    #[test]
    fn test_create() {
        let dense = Dense::create(3, 2, &Activation::Linear, true, Some(0.5));

        assert_eq!(dense.inputs, 3);
        assert_eq!(dense.outputs, 2);
        assert_eq!(dense.weights.len(), 2);
        assert_eq!(dense.weights[0].len(), 3);
        assert_eq!(dense.bias.unwrap().len(), 2);
        assert_eq!(dense.dropout, Some(0.5));
    }

    #[test]
    fn test_forward() {
        let mut dense = Dense::create(3, 2, &Activation::Linear, true, Some(0.5));
        dense.weights = vec![vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]];
        dense.bias = Some(vec![0.0, 0.0]);

        let input = tensor::Tensor::from_single(vec![1.0, 2.0, 3.0]);
        let (pre, post) = dense.forward(&input);

        assert_eq!(pre.get_flat(), vec![3.0, 3.0]);
        assert_eq!(post.get_flat(), vec![3.0, 3.0]);
    }

    #[test]
    fn test_backward() {
        let mut dense = Dense::create(3, 2, &Activation::Linear, true, Some(0.5));
        dense.weights = vec![vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]];
        dense.bias = Some(vec![0.0, 0.0]);

        let input = tensor::Tensor::from_single(vec![1.0, 2.0, 3.0]);
        let (pre, post) = dense.forward(&input);
        let gradient = tensor::Tensor::from_single(vec![1.0, 1.0]);

        let (input_gradient, weight_gradient, bias_gradient) =
            dense.backward(&gradient, &input, &post);

        assert_eq!(input_gradient.get_flat(), vec![1.0, 1.0, 1.0]);
        assert_eq!(
            weight_gradient.get_flat(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(bias_gradient.unwrap().get_flat(), vec![1.0, 1.0]);
    }
}
