// Copyright (C) 2024 Hallvard Høyland Lavik

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
#[derive(Clone)]
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

    /// Applies the activation function to the input `tensor::Tensor` in the forward direction using the respective formula.
    ///
    /// # Arguments
    ///
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of the output values.
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

    /// Applies the derivative of the activation function to the input `tensor::Tensor` in the backward direction using the derivative of the respective forward function.
    ///
    /// # Arguments
    ///
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of the output values.
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
#[derive(Clone)]
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
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Single(data) => {
                tensor::Data::Single(data.iter().map(|&v| v.max(0.0)).collect())
            }
            tensor::Data::Triple(data) => tensor::Data::Triple(
                data.iter()
                    .map(|i| {
                        i.iter()
                            .map(|j| j.iter().map(|&v| v.max(0.0)).collect())
                            .collect()
                    })
                    .collect(),
            ),
            _ => panic!("Invalid data type."),
        };
        tensor::Tensor {
            shape: input.shape.clone(),
            data,
        }
    }

    /// Backward pass of the ReLU activation function.
    ///
    /// # Function
    ///
    /// * `f'(x) = 1 if x > 0, 0 otherwise`
    ///
    /// # Arguments
    ///
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Single(data) => tensor::Data::Single(
                data.iter()
                    .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                    .collect(),
            ),
            tensor::Data::Triple(data) => tensor::Data::Triple(
                data.iter()
                    .map(|i| {
                        i.iter()
                            .map(|j| j.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect())
                            .collect()
                    })
                    .collect(),
            ),
            _ => panic!("Invalid data type."),
        };
        tensor::Tensor {
            shape: input.shape.clone(),
            data,
        }
    }
}

/// Leaky Rectified Linear Unit (LeakyReLU) activation function.
///
/// # Attributes
///
/// * `alpha` - The slope of the negative part of the function.
#[derive(Clone)]
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
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Single(data) => tensor::Data::Single(
                data.iter()
                    .map(|&v| if v > 0.0 { v } else { self.alpha * v })
                    .collect(),
            ),
            tensor::Data::Triple(data) => tensor::Data::Triple(
                data.iter()
                    .map(|i| {
                        i.iter()
                            .map(|j| {
                                j.iter()
                                    .map(|&v| if v > 0.0 { v } else { self.alpha * v })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
            ),
            _ => panic!("Invalid data type."),
        };
        tensor::Tensor {
            shape: input.shape.clone(),
            data,
        }
    }

    /// Backward pass of the LeakyReLU activation function.
    ///
    /// # Function
    ///
    /// * `f'(x) = 1 if x > 0, alpha otherwise`
    ///
    /// # Arguments
    ///
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Single(data) => tensor::Data::Single(
                data.iter()
                    .map(|&v| if v > 0.0 { 1.0 } else { self.alpha })
                    .collect(),
            ),
            tensor::Data::Triple(data) => tensor::Data::Triple(
                data.iter()
                    .map(|i| {
                        i.iter()
                            .map(|j| {
                                j.iter()
                                    .map(|&v| if v > 0.0 { 1.0 } else { self.alpha })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
            ),
            _ => panic!("Invalid data type."),
        };
        tensor::Tensor {
            shape: input.shape.clone(),
            data,
        }
    }
}

/// Sigmoid activation function.
#[derive(Clone)]
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
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Single(data) => {
                tensor::Data::Single(data.iter().map(|&v| 1.0 / (1.0 + f32::exp(-v))).collect())
            }
            tensor::Data::Triple(data) => tensor::Data::Triple(
                data.iter()
                    .map(|i| {
                        i.iter()
                            .map(|j| j.iter().map(|&v| 1.0 / (1.0 + f32::exp(-v))).collect())
                            .collect()
                    })
                    .collect(),
            ),
            _ => panic!("Invalid data type."),
        };
        tensor::Tensor {
            shape: input.shape.clone(),
            data,
        }
    }

    /// Backward pass of the Sigmoid activation function.
    ///
    /// # Function
    ///
    /// * `f'(x) = f(x) * (1 - f(x))`
    ///
    /// # Arguments
    ///
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Single(data) => tensor::Data::Single(
                data.iter()
                    .map(|&v| {
                        let y = 1.0 / (1.0 + f32::exp(-v));
                        y * (1.0 - y)
                    })
                    .collect(),
            ),
            tensor::Data::Triple(data) => tensor::Data::Triple(
                data.iter()
                    .map(|i| {
                        i.iter()
                            .map(|j| {
                                j.iter()
                                    .map(|&v| {
                                        let y = 1.0 / (1.0 + f32::exp(-v));
                                        y * (1.0 - y)
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
            ),
            _ => panic!("Invalid data type."),
        };
        tensor::Tensor {
            shape: input.shape.clone(),
            data,
        }
    }
}

/// Softmax activation function.
#[derive(Clone)]
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
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let x = input.get_flat();

        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        let y = exps.iter().map(|v| v / sum).collect();

        tensor::Tensor::single(y).reshape(input.shape.clone())
    }

    /// Backward pass of the Softmax activation function ([source](https://e2eml.school/softmax)).
    ///
    /// # Function
    ///
    /// * `f'(x_i) = f(x_i) * (1 - f(x_i)) - f(x) @ f(x) - sum_j(f(x_i) * f(x_j) - f(x) @ f(x))`
    ///
    /// # Arguments
    ///
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of the derivatives.
    pub fn backward(&self, logits: &tensor::Tensor) -> tensor::Tensor {
        let probability = self.forward(logits).get_flat();
        let scalar: f32 = probability
            .iter()
            .zip(probability.iter())
            .map(|(a, b)| a * b)
            .sum();

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

        tensor::Tensor::single(derivative).reshape(logits.shape.clone())
    }
}

/// Hyperbolic Tangent (Tanh) activation function.
#[derive(Clone)]
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
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of output values.
    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Single(data) => {
                tensor::Data::Single(data.iter().map(|&v| v.tanh()).collect())
            }
            tensor::Data::Triple(data) => tensor::Data::Triple(
                data.iter()
                    .map(|i| {
                        i.iter()
                            .map(|j| j.iter().map(|&v| v.tanh()).collect())
                            .collect()
                    })
                    .collect(),
            ),
            _ => panic!("Invalid data type."),
        };
        tensor::Tensor {
            shape: input.shape.clone(),
            data,
        }
    }

    /// Backward pass of the Tanh activation function.
    ///
    /// # Function
    ///
    /// * `f'(x) = 1 / cosh(x)^2`
    ///
    /// # Arguments
    ///
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let data = match &input.data {
            tensor::Data::Single(data) => {
                tensor::Data::Single(data.iter().map(|&v| 1.0 / (v.cosh().powi(2))).collect())
            }
            tensor::Data::Triple(data) => tensor::Data::Triple(
                data.iter()
                    .map(|i| {
                        i.iter()
                            .map(|j| j.iter().map(|&v| 1.0 / (v.cosh().powi(2))).collect())
                            .collect()
                    })
                    .collect(),
            ),
            _ => panic!("Invalid data type."),
        };
        tensor::Tensor {
            shape: input.shape.clone(),
            data,
        }
    }
}

/// Linear activation function.
#[derive(Clone)]
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
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of output values.
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
    /// * `input` - A `tensor::Tensor` of input values.
    ///
    /// # Returns
    ///
    /// * A `tensor::Tensor` of the derivatives.
    pub fn backward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        tensor::Tensor::ones(input.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn create_test_tensor() -> Tensor {
        Tensor::single(vec![-2.0, -1.0, 0.0, 1.0, 2.0])
    }

    #[test]
    fn test_relu() {
        let relu = Function::create(&Activation::ReLU);
        let input = create_test_tensor();

        // Test forward
        let output = relu.forward(&input);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        assert_eq!(output.get_flat(), expected);

        // Test backward
        let grad_output = relu.backward(&input);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        assert_eq!(grad_output.get_flat(), expected);
    }

    #[test]
    fn test_leaky_relu() {
        let leaky_relu = Function::create(&Activation::LeakyReLU);
        let input = create_test_tensor();

        // Test forward
        let output = leaky_relu.forward(&input);
        let expected = vec![-0.02, -0.01, 0.0, 1.0, 2.0];
        for (a, b) in output.get_flat().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // Test backward
        let grad_output = leaky_relu.backward(&input);
        let expected = vec![0.01, 0.01, 0.01, 1.0, 1.0];
        for (a, b) in grad_output.get_flat().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid() {
        let sigmoid = Function::create(&Activation::Sigmoid);
        let input = create_test_tensor();

        // Test forward
        let output = sigmoid.forward(&input);
        let expected = vec![
            0.11920292202211755,
            0.2689414213699951,
            0.5,
            0.7310585786300049,
            0.8807970779778823,
        ];
        for (a, b) in output.get_flat().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // Test backward
        let grad_output = sigmoid.backward(&input);
        let expected = vec![
            0.1049935854035065,
            0.19661193324148185,
            0.25,
            0.19661193324148185,
            0.10499358540350662,
        ];
        for (a, b) in grad_output.get_flat().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax() {
        let softmax = Function::create(&Activation::Softmax);
        let input = create_test_tensor();

        // Test forward
        let output = softmax.forward(&input);
        let expected = vec![
            0.011656230956039605,
            0.03168492079612427,
            0.0861285444362687,
            0.23412165725273662,
            0.6364086465588308,
        ];
        for (a, b) in output.get_flat().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // Test backward
        let grad_output = softmax.backward(&input);
        let expected = vec![1.4051605, 1.4051604, 1.4051604, 1.4051604, 1.4051605];
        for (a, b) in grad_output.get_flat().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tanh() {
        let tanh = Function::create(&Activation::Tanh);
        let input = create_test_tensor();

        // Test forward
        let output = tanh.forward(&input);
        let expected = vec![
            -0.9640275800758169,
            -0.7615941559557649,
            0.0,
            0.7615941559557649,
            0.9640275800758169,
        ];
        for (a, b) in output.get_flat().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // Test backward
        let grad_output = tanh.backward(&input);
        let expected = vec![
            0.07065082485316447,
            0.4199743416140261,
            1.0,
            0.4199743416140261,
            0.07065082485316447,
        ];
        for (a, b) in grad_output.get_flat().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_linear() {
        let linear = Function::create(&Activation::Linear);
        let input = create_test_tensor();

        // Test forward
        let output = linear.forward(&input);
        assert_eq!(output.get_flat(), input.get_flat());

        // Test backward
        let grad_output = linear.backward(&input);
        let expected = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(grad_output.get_flat(), expected);
    }

    #[test]
    fn test_function_display() {
        assert_eq!(format!("{}", Function::create(&Activation::ReLU)), "ReLU");
        assert_eq!(
            format!("{}", Function::create(&Activation::LeakyReLU)),
            "LeakyReLU"
        );
        assert_eq!(
            format!("{}", Function::create(&Activation::Sigmoid)),
            "Sigmoid"
        );
        assert_eq!(
            format!("{}", Function::create(&Activation::Softmax)),
            "Softmax"
        );
        assert_eq!(format!("{}", Function::create(&Activation::Tanh)), "Tanh");
        assert_eq!(
            format!("{}", Function::create(&Activation::Linear)),
            "Linear"
        );
    }
}
