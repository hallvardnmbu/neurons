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

use crate::algebra::dot;
use crate::tensor;

/// Activation functions for neural networks.
pub enum Activation {
    ReLU,
    LeakyReLU,
    Sigmoid,
    Softmax,
    Tanh,
    Linear,
}

/// Wrapper for the different activation functions.
pub enum Function {
    ReLU(ReLU),
    LeakyReLU(LeakyReLU),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
    Tanh(Tanh),
    Linear(Linear),
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Function::ReLU(_) => write!(f, "ReLU"),
            Function::LeakyReLU(_) => write!(f, "LeakyReLU"),
            Function::Sigmoid(_) => write!(f, "Sigmoid"),
            Function::Softmax(_) => write!(f, "Softmax"),
            Function::Tanh(_) => write!(f, "Tanh"),
            Function::Linear(_) => write!(f, "Linear"),
        }
    }
}

/// Wrapper for the creation, forward, and backward methods of the activation functions.
impl Function {

    /// Creates an activation function based on the provided `Activation` variant.
    pub fn create(activation: &Activation) -> Self {
        match activation {
            Activation::ReLU => Function::ReLU(ReLU {}),
            Activation::LeakyReLU => Function::LeakyReLU(LeakyReLU { alpha: 0.01 }),
            Activation::Sigmoid => Function::Sigmoid(Sigmoid {}),
            Activation::Softmax => Function::Softmax(Softmax {}),
            Activation::Tanh => Function::Tanh(Tanh {}),
            Activation::Linear => Function::Linear(Linear {}),
        }
    }

    /// Applies the activation function to the input vector in the forward direction using the
    /// respective formula.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        match self {
            Function::ReLU(act) => act.forward(input),
            Function::LeakyReLU(act) => act.forward(input),
            Function::Sigmoid(act) => act.forward(input),
            Function::Softmax(act) => act.forward(input),
            Function::Tanh(act) => act.forward(input),
            Function::Linear(act) => act.forward(input),
        }
    }

    /// Applies the derivative of the activation function to the input vector in the backward
    /// direction using the derivative of the respective forward function.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        match self {
            Function::ReLU(act) => act.backward(input),
            Function::LeakyReLU(act) => act.backward(input),
            Function::Sigmoid(act) => act.backward(input),
            Function::Softmax(act) => act.backward(input),
            Function::Tanh(act) => act.backward(input),
            Function::Linear(act) => act.backward(input),
        }
    }
}

/// Rectified Linear Unit (ReLU) activation function.
pub struct ReLU {}

impl ReLU {

    /// Forward pass of the ReLU activation function.
    ///
    /// # Function
    ///
    /// * `f(x) = max(0, x)`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Vector(vector) => {
                tensor::Data::Vector(vector.iter().map(|&v| v.max(0.0)).collect())
            },
            tensor::Data::Tensor(vector) => {
                tensor::Data::Tensor(vector.iter()
                    .map(|i| i.iter()
                        .map(|j| j.iter()
                            .map(|&v| v.max(0.0))
                            .collect())
                        .collect())
                    .collect())
            },
            _ => panic!("Invalid data type"),
        };
        tensor::Tensor { shape: input.shape.clone(), data }
    }

    /// Backward pass of the ReLU activation function.
    ///
    /// # Function
    ///
    /// * `f'(x) = 1 if x > 0, 0 otherwise`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Vector(vector) => {
                tensor::Data::Vector(vector.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect())
            },
            tensor::Data::Tensor(vector) => {
                tensor::Data::Tensor(vector.iter()
                    .map(|i| i.iter()
                        .map(|j| j.iter()
                            .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                            .collect())
                        .collect())
                    .collect())
            },
            _ => panic!("Invalid data type"),
        };
        tensor::Tensor { shape: input.shape.clone(), data }
    }
}

/// Leaky Rectified Linear Unit (LeakyReLU) activation function.
///
/// # Attributes
///
/// * `alpha` - The slope of the negative part of the function.
pub struct LeakyReLU {
    alpha: f32,
}

impl LeakyReLU {

    /// Forward pass of the LeakyReLU activation function.
    ///
    /// # Function
    ///
    /// * `f(x) = x if x > 0, alpha * x otherwise`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Vector(vector) => {
                tensor::Data::Vector(vector.iter().map(|&v| if v > 0.0 { v } else { self.alpha * v }).collect())
            },
            tensor::Data::Tensor(vector) => {
                tensor::Data::Tensor(vector.iter()
                    .map(|i| i.iter()
                        .map(|j| j.iter()
                            .map(|&v| if v > 0.0 { v } else { self.alpha * v })
                            .collect())
                        .collect())
                    .collect())
            },
            _ => panic!("Invalid data type"),
        };
        tensor::Tensor { shape: input.shape.clone(), data }
    }

    /// Backward pass of the LeakyReLU activation function.
    ///
    /// # Function
    ///
    /// * `f'(x) = 1 if x > 0, alpha otherwise`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Vector(vector) => {
                tensor::Data::Vector(vector.iter().map(|&v| if v > 0.0 { 1.0 } else { self.alpha }).collect())
            },
            tensor::Data::Tensor(vector) => {
                tensor::Data::Tensor(vector.iter()
                    .map(|i| i.iter()
                        .map(|j| j.iter()
                            .map(|&v| if v > 0.0 { 1.0 } else { self.alpha })
                            .collect())
                        .collect())
                    .collect())
            },
            _ => panic!("Invalid data type"),
        };
        tensor::Tensor { shape: input.shape.clone(), data }
    }
}

/// Sigmoid activation function.
pub struct Sigmoid {}

impl Sigmoid {

    /// Forward pass of the Sigmoid activation function.
    ///
    /// # Function
    ///
    /// * `f(x) = 1 / (1 + e^(-x))`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Vector(vector) => {
                tensor::Data::Vector(vector.iter().map(|&v| 1.0 / (1.0 + f32::exp(-v))).collect())
            },
            tensor::Data::Tensor(vector) => {
                tensor::Data::Tensor(vector.iter()
                    .map(|i| i.iter()
                        .map(|j| j.iter()
                            .map(|&v| 1.0 / (1.0 + f32::exp(-v)))
                            .collect())
                        .collect())
                    .collect())
            },
            _ => panic!("Invalid data type"),
        };
        tensor::Tensor { shape: input.shape.clone(), data }
    }

    /// Backward pass of the Sigmoid activation function.
    ///
    /// # Function
    ///
    /// * `f'(x) = f(x) * (1 - f(x))`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Vector(vector) => {
                tensor::Data::Vector(vector.iter().map(|&v| {
                    let y = 1.0 / (1.0 + f32::exp(-v));
                    y * (1.0 - y)
                }).collect())
            },
            tensor::Data::Tensor(vector) => {
                tensor::Data::Tensor(vector.iter()
                    .map(|i| i.iter()
                        .map(|j| j.iter()
                            .map(|&v| {
                                let y = 1.0 / (1.0 + f32::exp(-v));
                                y * (1.0 - y)
                            })
                            .collect())
                        .collect())
                    .collect())
            },
            _ => panic!("Invalid data type"),
        };
        tensor::Tensor { shape: input.shape.clone(), data }
    }
}

/// Softmax activation function.
pub struct Softmax {}

impl Softmax {

    /// Forward pass of the Softmax activation function.
    ///
    /// # Function
    ///
    /// * `f(x) = e^(x - max(x)) / sum(e^(x - max(x)))`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let x = input.get_flat();

        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        let y = exps.iter().map(|v| v / sum).collect();

        tensor::Tensor::from_single(y).reshape(input.shape.clone())
    }

    /// Backward pass of the Softmax activation function ([source](https://e2eml.school/softmax)).
    ///
    /// # Function
    ///
    /// * `f'(x_i) = f(x_i) * (1 - f(x_i)) - f(x) @ f(x) - sum_j(f(x_i) * f(x_j) - f(x) @ f(x))`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of the derivatives.
    pub fn backward(&self, logits: &tensor::Tensor) -> tensor::Tensor {
        let probability = self.forward(logits).get_flat();
        let scalar = dot(&probability, &probability);

        let mut derivative = vec![0.0f32; probability.len()];

        for i in 0..probability.len() {
            for j in 0..probability.len() {
                if i == j {
                    derivative[i] += probability[i] * (1.0 - probability[i]) - scalar;
                } else {
                    derivative[i] -= probability[i] * probability[j] - scalar;
                }
            }
        }

        tensor::Tensor::from_single(derivative).reshape(logits.shape.clone())
    }
}

/// Hyperbolic Tangent (Tanh) activation function.
pub struct Tanh {}

impl Tanh {

    /// Forward pass of the Tanh activation function.
    ///
    /// # Function
    ///
    /// * `f(x) = x.tanh()`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Vector(vector) => {
                tensor::Data::Vector(vector.iter().map(|&v| v.tanh()).collect())
            },
            tensor::Data::Tensor(vector) => {
                tensor::Data::Tensor(vector.iter()
                    .map(|i| i.iter()
                        .map(|j| j.iter()
                            .map(|&v| v.tanh())
                            .collect())
                        .collect())
                    .collect())
            },
            _ => panic!("Invalid data type"),
        };
        tensor::Tensor { shape: input.shape.clone(), data }
    }

    /// Backward pass of the Tanh activation function.
    ///
    /// # Function
    ///
    /// * `f'(x) = 1 / cosh(x)^2`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Vector(vector) => {
                tensor::Data::Vector(vector.iter().map(|&v| 1.0 / (v.cosh().powi(2))).collect())
            },
            tensor::Data::Tensor(vector) => {
                tensor::Data::Tensor(vector.iter()
                    .map(|i| i.iter()
                        .map(|j| j.iter()
                            .map(|&v| 1.0 / (v.cosh().powi(2)))
                            .collect())
                        .collect())
                    .collect())
            },
            _ => panic!("Invalid data type"),
        };
        tensor::Tensor { shape: input.shape.clone(), data }
    }
}

/// Linear activation function.
pub struct Linear {}

impl Linear {

    /// Forward pass of the Tanh activation function.
    ///
    /// # Function
    ///
    /// * `f(x) = x`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        input.clone()
    }

    /// Backward pass of the Tanh activation function.
    ///
    /// # Function
    ///
    /// * `f'(x) = 1`
    ///
    /// # Arguments
    ///
    /// * `input` - A vector of input values.
    ///
    /// # Returns
    ///
    /// * A vector of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        tensor::Tensor::ones(input.shape.clone())
    }
}
