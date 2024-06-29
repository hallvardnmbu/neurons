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

use crate::{random, tensor};
use crate::activation;
use crate::algebra::*;

/// A dense layer in a neural network.
///
/// # Attributes
///
/// * `weights` - The weights of the layer.
/// * `bias` - The bias of the layer.
/// * `activation` - The activation function of the layer.
pub struct Dense {
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,

    pub(crate) weights: Vec<Vec<f32>>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) activation: activation::Function,

    dropout: Option<f32>,

    pub(crate) training: bool,
}

impl std::fmt::Display for Dense {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Dense{}({} -> {}, bias: {})",
               self.activation, self.inputs, self.outputs, self.bias.is_some())
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
    pub fn create(inputs: usize,
                  outputs: usize,
                  activation: &activation::Activation,
                  bias: bool,
                  dropout: Option<f32>,
    ) -> Self {
        let mut generator = random::Generator::create(12345);
        Dense {
            inputs, outputs,
            weights: (0..outputs)
                .map(|_|
                    (0..inputs)
                    .map(|_| generator.generate(-1.0, 1.0))
                    .collect())
                .collect(),
            bias: match bias {
                true => Some((0..outputs).map(|_| generator.generate(-1.0, 1.0)).collect()),
                false => None,
            },
            activation: activation::Function::create(&activation),
            dropout,
            training: false,
        }
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

        let pre = tensor::Tensor::from_single(
            self.weights.iter().map(|w| dot(&w, &x)).collect()
        );
        let mut post = match &self.bias {
            Some(bias) => self.activation.forward(&pre).add(
                &tensor::Tensor::from_single(bias.clone())
            ),
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
    /// The weight gradient and the bias gradient of the layer.
    pub fn backward(
        &self, mut gradient: tensor::Tensor, input: &tensor::Tensor, output: &tensor::Tensor
    ) -> (tensor::Tensor, tensor::Tensor, Option<tensor::Tensor>) {
        let derivative: Vec<f32> = self.activation.backward(&output).get_flat();
        let delta: Vec<f32> = mul(&gradient.get_flat(), &derivative);

        let weight_gradient: Vec<Vec<f32>> = delta
            .iter().map(|d: &f32| input.get_flat()
            .iter().map(|i: &f32| i * d)
            .collect())
            .collect();
        let bias_gradient: Option<tensor::Tensor> = match self.bias {
            Some(_) => Some(tensor::Tensor::from_single(delta.clone())),
            None => None,
        };
        let input_gradient: Vec<f32> = (0..input.get_flat().len())
            .map(|i: usize| delta
                .iter().zip(self.weights.iter())
                .map(|(d, w)| d * w[i])
                .sum::<f32>())
            .collect();

        // gradient.data = tensor::Data::Vector(input_gradient.clone());
        gradient = tensor::Tensor::from_single(input_gradient);

        (gradient,
         tensor::Tensor::from(vec![weight_gradient]),
         bias_gradient)
    }
}
