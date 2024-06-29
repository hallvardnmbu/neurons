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

use crate::tensor;
use crate::activation;
use crate::optimizer;
use crate::objective;

use crate::dense;
use crate::convolution;

pub enum Layer {
    Dense(dense::Dense),
    Convolution(convolution::Convolution),
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Layer::Dense(layer) => write!(f, "{}", layer),
            Layer::Convolution(layer) => write!(f, "{}", layer),
        }
    }
}

/// A feedforward neural network.
///
/// # Attributes
///
/// * `image` - The input dimensions of the network, for image inputs.
/// * `layers` - The layers of the network.
/// * `optimizer` - The optimizer function of the network.
/// * `objective` - The objective function of the network.
pub struct Feedforward {
    pub input: tensor::Shape,

    pub(crate) layers: Vec<Layer>,
    pub(crate) optimizer: optimizer::Optimizer,
    pub(crate) objective: objective::Function,
}

impl std::fmt::Display for Feedforward {
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

impl Feedforward {

    /// Creates a new (empty) feedforward neural network.
    ///
    /// Generates a new neural network with no layers, with a standard optimizer and objective,
    /// respectively:
    ///
    /// * Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.1.
    /// * Objective: Mean Squared Error (MSE).
    ///
    /// # Arguments
    ///
    /// * `input` - The input dimensions of the network.
    ///     Either `tensor::Shape::Dense` or `tensor::Shape::Convolution`.
    ///
    /// # Returns
    ///
    /// An empty neural network, with no layers.
    pub fn new(input: tensor::Shape) -> Self {
        Feedforward {
            input,
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

    /// Adds a new dense layer to the network.
    ///
    /// The layer is added to the end of the network, and the number of inputs to the layer must
    /// be equal to the number of outputs from the previous layer. The activation function of the
    /// layer is set to the given activation function, and the layer may have a bias if specified.
    ///
    /// # Arguments
    ///
    /// * `outputs` - The number of outputs from the layer.
    /// * `activation` - The activation function of the layer.
    /// * `bias` - Whether the layer should have a bias.
    /// * `dropout` - The dropout rate of the layer.
    ///
    /// # Panics
    ///
    /// * If the number of inputs to the layer is not equal to the number of outputs from the
    /// previous layer.
    pub fn add_dense(
        &mut self,
        outputs: usize,
        activation: activation::Activation,
        bias: bool,
        dropout: Option<f32>
    ) {
        if self.layers.is_empty() {
            let inputs = match self.input {
                tensor::Shape::Dense(inputs) => inputs,
                _ => panic!("Network is configured for image inputs; the first layer must be Convolutional"),
            };
            self.layers.push(
                Layer::Dense(dense::Dense::create(inputs, outputs, &activation, bias, dropout))
            );
            return;
        }
        let inputs = match &mut self.layers.last_mut().unwrap() {
            Layer::Dense(layer) => layer.outputs,
            Layer::Convolution(layer) => {
                // Make sure the output of the convolutional layer is flattened.
                layer.flatten_output = true;
                match layer.outputs {
                    tensor::Shape::Convolution(ch, he, wi) => ch * he * wi,
                    _ => panic!("Expected Convolution shape"),
                }
            },
        };

        self.layers.push(
            Layer::Dense(dense::Dense::create(inputs, outputs, &activation, bias, dropout))
        );
    }

    /// Adds a new convolutional layer to the network.
    ///
    /// The layer is added to the end of the network, and the number of inputs to the layer must
    /// be equal to the number of outputs from the previous layer. The activation function of the
    /// layer is set to the given activation function, and the layer may have a bias if specified.
    ///
    /// # Arguments
    ///
    /// * `channels` - The number of filters of the layer.
    /// * `kernel` - The size of the kernel.
    /// * `stride` - The stride of the kernel.
    /// * `padding` - The padding of the input.
    /// * `activation` - The activation function of the layer.
    /// * `bias` - Whether the layer should have a bias.
    /// * `dropout` - The dropout rate of the layer.
    ///
    /// # Panics
    ///
    /// * If the number of inputs to the layer is not equal to the number of outputs from the
    /// previous layer.
    pub fn add_convolution(
        &mut self,
        channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        activation: activation::Activation,
        bias: bool,
        dropout: Option<f32>,
    ) {
        if self.layers.is_empty() {
            self.layers.push(
                Layer::Convolution(convolution::Convolution::create(
                    self.input.clone(),
                    channels, &activation, bias,
                    kernel, stride, padding, dropout
                ))
            );
            return;
        }
        self.layers.push(
            Layer::Convolution(convolution::Convolution::create(
                match self.layers.last().unwrap() {
                    Layer::Dense(layer) => tensor::Shape::Dense(layer.outputs),
                    Layer::Convolution(layer) => layer.outputs.clone(),
                },
                channels, &activation, bias,
                kernel, stride, padding, dropout
            ))
        );
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
        match self.layers[layer] {
            Layer::Dense(ref mut layer) => layer.activation = activation::Function::create(&activation),
            Layer::Convolution(ref mut layer) => layer.activation = activation::Function::create(&activation),
        }
    }

    /// Set the optimizer function of the network.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The new optimizer function to be used.
    pub fn set_optimizer(&mut self, mut optimizer: optimizer::Optimizer) {

        // Create the placeholder vector used for various optimizer functions.
        // See the `match optimizer` below.
        let vector: Vec<Vec<Vec<f32>>> = self.layers.iter().rev().map(|layer| {
            let (inputs, outputs, bias) = match layer {
                Layer::Dense(layer) => (layer.inputs, layer.outputs, layer.bias.is_some()),
                Layer::Convolution(layer) => {
                    let (chi, hei, wii) = match layer.inputs {
                        tensor::Shape::Convolution(ch, he, wi) => (ch, he, wi),
                        _ => panic!("Expected Convolution shape"),
                    };
                    let (cho, heo, wio) = match layer.outputs {
                        tensor::Shape::Convolution(ch, he, wi) => (ch, he, wi),
                        _ => panic!("Expected Convolution shape"),
                    };
                    (chi * hei * wii, cho * heo * wio, layer.bias.is_some())
                },
            };
            vec![vec![0.0; inputs]; outputs + if bias { 1 } else { 0 }]
        }).collect();

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
                params.velocity = vector.clone();
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

                params.velocity = vector.clone();
                params.momentum = vector.clone();
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

                params.velocity = vector.clone();
                params.momentum = vector.clone();
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

                params.velocity = vector.clone();
                params.gradient = vector.clone();
                params.buffer = vector.clone();
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
        &mut self, inputs: &Vec<tensor::Tensor>, targets: &Vec<tensor::Tensor>, epochs: i32
    ) -> Vec<f32> {
        for layer in &mut self.layers {
            match layer {
                Layer::Dense(layer) => layer.training = true,
                Layer::Convolution(layer) => layer.training = true,
            }
        }

        let checkpoint = epochs / 10;
        let mut losses = Vec::new();
        for epoch in 1..epochs+1 {
            let mut _losses = 0.0f32;
            for (input, target) in inputs
                .iter()
                .zip(targets.iter()) {

                let (unactivated, activated) = self.forward(input);
                let (loss, mut gradient) = self.loss(&activated.last().unwrap(), target);
                _losses += loss;

                // TODO: Backward pass on batch instead of single input.
                self.backward(epoch, gradient, &unactivated, &activated);
            }
            losses.push(_losses / inputs.len() as f32);

            if epoch % checkpoint == 0 && epoch > 0 {
                println!("Epoch: {} Loss: {}",
                         epoch, losses[(epoch as usize)-(checkpoint as usize)..(epoch as usize)]
                             .iter().sum::<f32>() / checkpoint as f32);
            }
        }
        for layer in &mut self.layers {
            match layer {
                Layer::Dense(layer) => layer.training = false,
                Layer::Convolution(layer) => layer.training = false,
            }
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
        &mut self, inputs: &Vec<tensor::Tensor>, targets: &Vec<tensor::Tensor>, tol: f32
    ) -> (f32, f32) {

        let mut losses = Vec::new();
        let mut accuracy = Vec::new();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict(input);
            let (loss, _) = self.loss(&prediction, target);

            losses.push(loss);

            let target = target.get_flat();
            let prediction = prediction.get_flat();

            match self.layers.last().unwrap() {
                Layer::Dense(layer) => {
                    match layer.activation {
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
                },
                Layer::Convolution(layer) => {
                    unimplemented!("Convolutional layer not implemented for validation")
                }
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
    pub fn accuracy(&self, predictions: &Vec<tensor::Tensor>, targets: &Vec<tensor::Tensor>, tol: f32) -> f32 {
        let mut accuracy: Vec<f32> = Vec::new();

        let predictions: Vec<Vec<f32>> = predictions.iter().map(|t| t.get_flat()).collect();
        let targets: Vec<Vec<f32>> = targets.iter().map(|t| t.get_flat()).collect();

        for (prediction, target) in predictions.iter().zip(targets.iter()) {

            match self.layers.last().unwrap() {
                Layer::Dense(layer) => {
                    match layer.activation {
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
                },
                Layer::Convolution(layer) => {
                    unimplemented!("Convolutional layer not implemented for validation")
                }
            };
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
    pub fn predict(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let mut output = input.clone();
        for layer in &self.layers {
            match layer {
                Layer::Dense(layer) => {
                    let (_, out) = layer.forward(&output);
                    output = out;
                },
                Layer::Convolution(layer) => {
                    unimplemented!("Convolutional layer not implemented for prediction");
                },
            }
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
    pub fn predict_batch(&self, inputs: &Vec<tensor::Tensor>) -> Vec<tensor::Tensor> {
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
    pub fn forward(&mut self, input: &tensor::Tensor) -> (Vec<tensor::Tensor>, Vec<tensor::Tensor>) {
        let mut unactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];

        for layer in &self.layers {
            let (pre, post): (tensor::Tensor, tensor::Tensor) = match layer {
                Layer::Dense(layer) => {
                    layer.forward(activated.last().unwrap())
                },
                Layer::Convolution(layer) => {
                    layer.forward(activated.last().unwrap())
                },
            };

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
    fn loss(&mut self, prediction: &tensor::Tensor, target: &tensor::Tensor) -> (f32, tensor::Tensor) {
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
        &mut self,
        stepnr: i32,
        mut gradient: tensor::Tensor,
        unactivated: &Vec<tensor::Tensor>,
        activated: &Vec<tensor::Tensor>
    ) {
        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            let input: &tensor::Tensor = &activated[activated.len() - i - 2];
            let output: &tensor::Tensor = &unactivated[unactivated.len() - i - 1];

            // println!("Iteration {} {} -> {}", i, input.shape, output.shape);
            // println!("{}", layer);

            let (_gradient, mut weight_gradient, bias_gradient) = match layer {
                Layer::Dense(layer) => layer.backward(gradient, input, output),
                Layer::Convolution(layer) => layer.backward(gradient, input, output),
            };
            gradient = _gradient;

            match layer {
                Layer::Dense(layer) => {

                    // Weight update.
                    for (j, (weights, gradients)) in layer
                        .weights.iter_mut()
                        .zip(match weight_gradient.data {
                            tensor::Data::Tensor(data) => { data },
                            _ => panic!("Expected a tensor, but got one-dimensional data."),
                        }[0].iter_mut())
                        .enumerate() {
                        self.optimizer.update(i, j, stepnr, weights, gradients);
                    }

                    // Bias update.
                    if let Some(ref mut bias) = layer.bias {
                        // Using `layer.weights.len()` as the bias' momentum/velocity is stored therein.
                        self.optimizer.update(i, layer.weights.len(), stepnr,
                                              bias, &mut bias_gradient.unwrap().get_flat());
                    }
                },
                Layer::Convolution(layer) => {
                    unimplemented!("Convolutional layer not implemented for backpropagation");
                },
            }
        }
    }
}
