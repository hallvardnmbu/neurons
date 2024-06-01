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

use crate::activation;
use crate::layer;
use crate::optimizer;
use crate::objective;

/// A neural network.
///
/// # Attributes
///
/// * `layers` - The layers of the network.
/// * `optimizer` - The optimizer function of the network.
/// * `objective` - The objective function of the network.
/// * `training` - Whether the network is currently training.
pub struct Network {
    pub(crate) layers: Vec<layer::Layer>,
    pub(crate) optimizer: optimizer::Optimizer,
    pub(crate) objective: objective::Function,
}

impl std::fmt::Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Network (\n")?;

        write!(f, "\toptimizer: (\n{}\n", self.optimizer)?;
        write!(f, "\tobjective: (\n\t\t{}\n\t)\n", self.objective)?;

        write!(f, " \tlayers: (\n")?;
        for (i, layer) in self.layers.iter().enumerate() {
            write!(f, "\t\t{}: {}\n", i, layer)?;
        }
        write!(f, "\t)\n)")?;
        Ok(())
    }
}

impl Network {

    /// Creates a new neural network with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `nodes` - The number of nodes in each layer.
    /// * `biases` - Whether each layer should have a bias.
    /// * `activations` - The activation functions of each layer.
    /// * `optimizer` - The optimizer function of the network.
    /// * `objective` - The objective function of the network.
    ///
    /// # Returns
    ///
    /// A new neural network with the given parameters.
    ///
    /// # Panics
    ///
    /// * If the number of activations is not equal to the number of nodes minus one.
    /// * If the number of biases is not equal to the number of nodes minus one.
    pub fn create(
        nodes: Vec<u16>,
        biases: Vec<bool>,
        activations: Vec<activation::Activation>,
        optimizer: optimizer::Optimizer,
        objective: objective::Objective,
    ) -> Self {
        assert_eq!(nodes.len(), activations.len() + 1, "Invalid number of activations");
        assert_eq!(nodes.len(), biases.len() + 1, "Invalid number of biases");

        let mut layers = Vec::new();
        for i in 0..nodes.len() - 1 {
            layers.push(layer::Layer::create(nodes[i], nodes[i + 1], &activations[i], biases[i], None));
        }

        Network {
            layers,
            optimizer,
            objective: objective::Function::create(objective, None),
        }
    }

    /// Creates a new (empty) neural network.
    ///
    /// Generates a new neural network with no layers, with a standard optimizer and objective,
    /// respectively:
    ///
    /// * Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.1.
    /// * Objective: Mean Squared Error (MSE).
    ///
    /// # Returns
    ///
    /// An empty neural network, with no layers.
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            optimizer: optimizer::Optimizer::SGD(
                optimizer::SGD {
                    learning_rate: 0.1,
                    decay: None,
                }
            ),
            objective: objective::Function::create(objective::Objective::MSE, None),
        }
    }

    /// Adds a new layer to the network.
    ///
    /// The layer is added to the end of the network, and the number of inputs to the layer must
    /// be equal to the number of outputs from the previous layer. The activation function of the
    /// layer is set to the given activation function, and the layer may have a bias if specified.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The number of inputs to the layer.
    /// * `outputs` - The number of outputs from the layer.
    /// * `activation` - The activation function of the layer.
    /// * `bias` - Whether the layer should have a bias.
    ///
    /// # Panics
    ///
    /// * If the number of inputs to the layer is not equal to the number of outputs from the
    /// previous layer.
    pub fn add_layer(
        &mut self, inputs: u16, outputs: u16, activation: activation::Activation,
        bias: bool, dropout: Option<f32>
    ) {
        if self.layers.is_empty() {
            self.layers.push(layer::Layer::create(inputs, outputs, &activation, bias, dropout));
            return;
        }
        let previous = match self.layers.last() {
            Some(layer) => layer.weights.len() as u16,
            None => inputs,
        };
        assert_eq!(previous, inputs,
                   "Invalid number of inputs. Last layer has {} inputs.", previous);
        self.layers.push(layer::Layer::create(inputs, outputs, &activation, bias, dropout));
    }

    /// Set the activation function of a layer.
    ///
    /// # Arguments
    ///
    /// * `layer` - The index of the layer (in the `self.layers` vector).
    /// * `activation` - The new activation function to be used.
    ///
    /// # Panics
    ///
    /// * If the layer index is out of bounds.
    pub fn set_activation(&mut self, layer: usize, activation: activation::Activation) {
        if layer >= self.layers.len() {
            panic!("Invalid layer index");
        }
        self.layers[layer].activation = activation::Function::create(&activation);
    }

    /// Set the optimizer function of the network.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The new optimizer function to be used.
    pub fn set_optimizer(&mut self, mut optimizer: optimizer::Optimizer) {
        match optimizer {
            optimizer::Optimizer::SGD(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.1;
                }
            },
            optimizer::Optimizer::SGDM(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.1;
                }
                if params.momentum == 0.0 {
                    params.momentum = 0.9;
                }
                params.velocity = self.layers.iter().rev().map(|layer| {
                    vec![vec![0.0; layer.weights[0].len()];
                         layer.weights.len() + if layer.bias.is_some() { 1 } else { 0 }]
                }).collect();
            },
            optimizer::Optimizer::Adam(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.001;
                }
                if params.beta1 == 0.0 {
                    params.beta1 = 0.9;
                }
                if params.beta2 == 0.0 {
                    params.beta2 = 0.999;
                }
                if params.epsilon == 0.0 {
                    params.epsilon = 1e-8;
                }

                params.velocity = self.layers.iter().rev().map(|layer| {
                    vec![vec![0.0; layer.weights[0].len()];
                         layer.weights.len() + if layer.bias.is_some() { 1 } else { 0 }]
                }).collect();
                params.momentum = params.velocity.clone();
            },
            optimizer::Optimizer::AdamW(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.001;
                }
                if params.beta1 == 0.0 {
                    params.beta1 = 0.9;
                }
                if params.beta2 == 0.0 {
                    params.beta2 = 0.999;
                }
                if params.epsilon == 0.0 {
                    params.epsilon = 1e-8;
                }

                params.velocity = self.layers.iter().rev().map(|layer| {
                    vec![vec![0.0; layer.weights[0].len()];
                         layer.weights.len() + if layer.bias.is_some() { 1 } else { 0 }]
                }).collect();
                params.momentum = params.velocity.clone();
            },
            optimizer::Optimizer::RMSprop(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.01;
                }
                if params.alpha == 0.0 {
                    params.alpha = 0.99;
                }
                if params.epsilon == 0.0 {
                    params.epsilon = 1e-8;
                }

                params.velocity = self.layers.iter().rev().map(|layer| {
                    vec![vec![0.0; layer.weights[0].len()];
                         layer.weights.len() + if layer.bias.is_some() { 1 } else { 0 }]
                }).collect();
                params.gradient = params.velocity.clone();
                params.buffer = params.velocity.clone();
            },
        };
        self.optimizer = optimizer;
    }

    /// Set the objective function of the network.
    ///
    /// # Arguments
    ///
    /// * `objective` - The new objective function to be used.
    /// * `clamp` - The clamp values for the objective function.
    pub fn set_objective(&mut self, objective: objective::Objective, clamp: Option<(f32, f32)>) {
        self.objective = objective::Function::create(objective, clamp);
    }

    /// Train the network on the given inputs and targets.
    ///
    /// Computes the forward and backward pass of the network for the given number of epochs,
    /// with respect to the given inputs and targets. The loss and gradient of the network is
    /// computed for each sample in the input data, and the weights and biases of the network are
    /// updated accordingly.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The input data (x).
    /// * `targets` - The targets of the given inputs (y).
    /// * `epochs` - The number of epochs to train the network for.
    ///
    /// # Returns
    ///
    /// A vector of the average loss of the network per epoch.
    pub fn learn(
        &mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, epochs: i32
    ) -> Vec<f32> {
        for layer in &mut self.layers {
            layer.training = true;
        }

        let checkpoint = epochs / 10;
        let mut losses = Vec::new();
        for epoch in 1..epochs+1 {
            let mut _losses = 0.0f32;
            for (input, target) in inputs
                .iter()
                .zip(targets.iter()) {

                let (unactivated, activated) = self.forward(input);
                let (loss, gradient) = self.loss(&activated.last().unwrap(), target);
                _losses += loss;

                // TODO: Backward pass on batch instead of single input.
                self.backward(epoch, gradient, unactivated, activated);
            }
            losses.push(_losses / inputs.len() as f32);

            if epoch % checkpoint == 0 && epoch > 0 {
                println!("Epoch: {} Loss: {}",
                         epoch, losses[(epoch as usize)-(checkpoint as usize)..(epoch as usize)]
                             .iter().sum::<f32>() / checkpoint as f32);
            }
        }
        for layer in &mut self.layers {
            layer.training = false;
        }

        losses
    }

    /// Validate the network on the given inputs and targets.
    ///
    /// Computes the forward pass of the network for the given inputs, and compares the output to
    /// the targets. The accuracy and loss of the network is computed for each sample in the
    /// input.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The input data (x).
    /// * `targets` - The targets of the given inputs (y).
    /// * `tol` - The tolerance for the accuracy, see `self.accuracy`.
    ///
    /// # Returns
    ///
    /// A tuple containing the total accuracy and loss of the network.
    pub fn validate(
        &mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, tol: f32
    ) -> (f32, f32) {

        let mut losses = Vec::new();
        let mut accuracy = Vec::new();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict(input);
            let (loss, _) = self.loss(&prediction, target);

            losses.push(loss);

            match self.layers.last().unwrap().activation {
                activation::Function::Softmax(_) => {
                    let predicted = prediction.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index).unwrap() as f32;
                    let actual = target.iter().position(|&v| v == 1.0).unwrap() as f32;
                    accuracy.push(if (predicted - actual).abs() < tol { 1.0 } else { 0.0 });
                },
                _ => {
                    if target.len() == 1 {
                        accuracy.push(if (prediction[0] - target[0]).abs() < tol { 1.0 } else { 0.0 });
                    } else {
                        target.iter().zip(prediction.iter()).for_each(
                            |(t, p)| accuracy.push(if (t - p).abs() < tol { 1.0 } else { 0.0 })
                        );
                    }
                },
            };
        }

        (accuracy.iter().sum::<f32>() / accuracy.len() as f32,
         losses.iter().sum::<f32>() / inputs.len() as f32)
    }

    /// Compute the accuracy of the network on the given inputs and targets.
    ///
    /// The accuracy is computed with respect to the given tolerance. I.e., if the difference
    /// between the prediction and target is less than the tolerance, it's assumed to be
    /// correctly predicted.
    ///
    /// # Arguments
    ///
    /// * `predictions` - The predictions of the network.
    /// * `targets` - The targets of the given inputs.
    /// * `tol` - The tolerance for the accuracy, see above.
    ///
    /// # Returns
    ///
    /// The accuracy of the network on the given inputs and targets.
    pub fn accuracy(&self, predictions: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, tol: f32) -> f32 {
        let mut accuracy: Vec<f32> = Vec::new();

        for (prediction, target) in predictions.iter().zip(targets.iter()) {
            match self.layers.last().unwrap().activation {
                activation::Function::Softmax(_) => {
                    let predicted = prediction.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index).unwrap() as f32;
                    let actual = target.iter().position(|&v| v == 1.0).unwrap() as f32;
                    accuracy.push(if (predicted - actual).abs() < tol { 1.0 } else { 0.0 });
                },
                _ => {
                    if target.len() == 1 {
                        accuracy.push(if (prediction[0] - target[0]).abs() < tol { 1.0 } else { 0.0 });
                    } else {
                        target.iter().zip(prediction.iter()).for_each(
                            |(t, p)| accuracy.push(if (t - p).abs() < tol { 1.0 } else { 0.0 })
                        );
                    }
                },
            }
        }

        accuracy.iter().sum::<f32>() / accuracy.len() as f32
    }

    /// Predict the output of the network for the given input.
    ///
    /// Computes the forward pass of the network for the given input, and returns the output.
    /// That is, the output of the last layer of the network only.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data (x).
    ///
    /// # Returns
    ///
    /// The output of the network for the given input.
    pub fn predict(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = input.clone();
        for layer in &self.layers {
            let (_, out) = layer.forward(&output);
            output = out;
        }
        output
    }

    /// Predict the output of the network for the given two-dimensional inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The input data (x).
    ///
    /// # Returns
    ///
    /// The output of the network for each of the given inputs.
    pub fn predict_batch(&self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        inputs.iter().map(|input| self.predict(input)).collect()
    }

    /// Compute the forward pass of the network for the given input, including all intermediate
    /// pre- and post-activation values.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data (x).
    ///
    /// # Returns
    ///
    /// A tuple containing the pre-activation and post-activation values of each layer.
    pub fn forward(&mut self, input: &Vec<f32>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut unactivated: Vec<Vec<f32>> = Vec::new();
        let mut activated: Vec<Vec<f32>> = vec![input.clone()];

        for layer in &self.layers {
            let (pre, post): (Vec<f32>, Vec<f32>) = layer.forward(&activated.last().unwrap());

            unactivated.push(pre);
            activated.push(post);
        }

        (unactivated, activated)
    }

    /// Compute the loss and gradient of the network for the given prediction and target.
    ///
    /// # Arguments
    ///
    /// * `prediction` - The prediction of the network.
    /// * `target` - The target of the given input.
    ///
    /// # Returns
    ///
    /// A tuple containing the loss and gradient of the network for the given prediction and target.
    fn loss(&mut self, prediction: &Vec<f32>, target: &Vec<f32>) -> (f32, Vec<f32>) {
        self.objective.loss(prediction, target)
    }

    /// Compute the backward pass of the network for the given gradient, and update the weights
    /// and biases of the network accordingly.
    ///
    /// # Arguments
    ///
    /// * `stepnr` - The current step number (i.e., epoch number).
    /// * `gradient` - The gradient of the output.
    /// * `unactivated` - The pre-activation values of each layer.
    /// * `activated` - The post-activation values of each layer.
    fn backward(
        &mut self, stepnr: i32,
        mut gradient: Vec<f32>, unactivated: Vec<Vec<f32>>, activated: Vec<Vec<f32>>
    ) {

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            let input: &Vec<f32> = &activated[activated.len() - i - 2];
            let output: &Vec<f32> = &unactivated[unactivated.len() - i - 1];

            let (mut weight_gradient, bias_gradient, input_gradient) =
                layer.backward(&gradient, input, output);
            gradient = input_gradient;

            // Weight update.
            for (j, (weights, gradients)) in layer
                .weights.iter_mut()
                .zip(weight_gradient.iter_mut())
                .enumerate() {
                self.optimizer.update(i, j, stepnr, weights, gradients);
            }

            // Bias update.
            if let Some(ref mut bias) = layer.bias {
                // Using `layer.weights.len()` as the bias' momentum/velocity is stored therein.
                self.optimizer.update(i, layer.weights.len(), stepnr,
                                      bias, &mut bias_gradient.unwrap());
            }
        }
    }
}
