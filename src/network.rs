// Copyright (C) 2024 Hallvard Høyland Lavik

use std::collections::HashMap;

use crate::{activation, convolution, dense, objective, optimizer, tensor};

/// Layer types of the network.
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

impl Layer {
    /// Extracts the number of parameters in the layer.
    fn parameters(&self) -> usize {
        match self {
            Layer::Dense(layer) => layer.parameters(),
            Layer::Convolution(layer) => layer.parameters(),
        }
    }
}

/// A feedforward neural network.
///
/// # Attributes
///
/// * `input` - The input `tensor::Shape` of the network.
/// * `layers` - The `Layer`s of the network.
/// * `feedbacks` - The feedback connections of the network.
/// * `optimizer` - The `optimizer::Optimizer` function of the network.
/// * `objective` - The `objective::Function` of the network.
pub struct Network {
    pub(crate) input: tensor::Shape,

    pub(crate) layers: Vec<Layer>,
    pub(crate) feedbacks: HashMap<usize, usize>,

    pub(crate) optimizer: optimizer::Optimizer,
    pub(crate) objective: objective::Function,
}

impl std::fmt::Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Network (\n")?;

        write!(f, "\toptimizer: (\n{}\n", self.optimizer)?;
        write!(f, "\tobjective: (\n\t\t{}\n\t)\n", self.objective)?;

        write!(f, "\tlayers: (\n")?;
        for (i, layer) in self.layers.iter().enumerate() {
            write!(f, "\t\t{}: {}\n", i, layer)?;
        }
        write!(f, "\t)\n")?;
        write!(f, "\tparameters: {}\n)", self.parameters())?;
        Ok(())
    }
}

impl Network {
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
        Network {
            input,
            layers: Vec::new(),
            feedbacks: HashMap::new(),
            optimizer: optimizer::Optimizer::SGD(optimizer::SGD {
                learning_rate: 0.1,
                decay: None,
            }),
            objective: objective::Function::create(objective::Objective::MSE, None),
        }
    }

    /// Add a dense layer to the network.
    ///
    /// The layer is added to the end of the network, and the number of inputs to the layer must
    /// be equal to the number of outputs from the previous layer. The activation function of the
    /// layer is set to the given activation function, and the layer may have a bias if specified.
    ///
    /// # Arguments
    ///
    /// * `outputs` - The number of outputs from the layer.
    /// * `activation` - The `activation::Activation` function of the layer.
    /// * `bias` - Whether the layer should contain a bias.
    /// * `dropout` - The dropout rate of the layer (applied during training only).
    ///
    /// # Panics
    ///
    /// * If the network is configured for image inputs, and the first layer is not convolutional.
    /// * If the number of inputs to the layer is not equal to the number of outputs from the
    /// previous layer.
    pub fn dense(
        &mut self,
        outputs: usize,
        activation: activation::Activation,
        bias: bool,
        dropout: Option<f32>,
    ) {
        if self.layers.is_empty() {
            let inputs = match self.input {
                tensor::Shape::Vector(inputs) => inputs,
                _ => panic!(
                    "Network is configured for image inputs; the first layer must be Convolutional"
                ),
            };
            self.layers.push(Layer::Dense(dense::Dense::create(
                inputs,
                outputs,
                &activation,
                bias,
                dropout,
            )));
            return;
        }
        let inputs = match &mut self.layers.last_mut().unwrap() {
            Layer::Dense(layer) => layer.outputs,
            Layer::Convolution(layer) => {
                // Make sure the output of the convolutional layer is flattened.
                layer.flatten_output = true;
                match layer.outputs {
                    tensor::Shape::Tensor(ch, he, wi) => ch * he * wi,
                    _ => panic!("Expected Convolution shape"),
                }
            }
        };

        self.layers.push(Layer::Dense(dense::Dense::create(
            inputs,
            outputs,
            &activation,
            bias,
            dropout,
        )));
    }

    /// Adds a new convolutional layer to the network.
    ///
    /// The layer is added to the end of the network, and the number of inputs to the layer must
    /// be equal to the number of outputs from the previous layer. The activation function of the
    /// layer is set to the given activation function, and the layer may have a bias if specified.
    ///
    /// # Arguments
    ///
    /// * `filters` - The number of filters of the layer.
    /// * `kernel` - The size of the kernel.
    /// * `stride` - The stride of the kernel.
    /// * `padding` - The padding of the input.
    /// * `activation` - The `activation::Activation` function of the layer.
    /// * `dropout` - The dropout rate of the layer (applied during training).
    pub fn convolution(
        &mut self,
        filters: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        activation: activation::Activation,
        dropout: Option<f32>,
    ) {
        if self.layers.is_empty() {
            self.layers
                .push(Layer::Convolution(convolution::Convolution::create(
                    self.input.clone(),
                    filters,
                    &activation,
                    kernel,
                    stride,
                    padding,
                    dropout,
                )));
            return;
        }
        self.layers
            .push(Layer::Convolution(convolution::Convolution::create(
                match self.layers.last().unwrap() {
                    Layer::Dense(layer) => tensor::Shape::Vector(layer.outputs),
                    Layer::Convolution(layer) => layer.outputs.clone(),
                },
                filters,
                &activation,
                kernel,
                stride,
                padding,
                dropout,
            )));
    }

    /// Add a feedback connection between two layers.
    ///
    /// INCOMPLETE: Currently only supports feedback connections between dense layers.
    ///
    /// # Arguments
    ///
    /// * `from` - The index of the layer to connect from.
    /// * `to` - The index of the layer to connect to.
    pub fn feedback(&mut self, from: usize, to: usize) {
        if from > self.layers.len() || to >= self.layers.len() || from <= to {
            panic!("Invalid layer indices for feedback connection.");
        }

        // TODO: Add support for convolutional layers.
        // TODO: Add support for mismatched input/output sizes.
        let inputs = match &self.layers[from] {
            Layer::Dense(layer) => &layer.inputs,
            _ => unimplemented!("Feedback connections for convolutional layers."),
        };
        let outputs = match &self.layers[to] {
            Layer::Dense(layer) => &layer.outputs,
            _ => unimplemented!("Feedback connections for convolutional layers."),
        };
        if inputs != outputs {
            panic!("Incompatible number of values for feedback connection.");
        }

        // Loop through layers to -> from and add +1 to its loopback count.
        for k in to..from {
            match &mut self.layers[k] {
                Layer::Dense(layer) => layer.loops += 1.0,
                _ => unimplemented!("Feedback connections for convolutional layers."),
            }
        }

        // Store the feedback connection for use in the forward pass.
        if self.feedbacks.contains_key(&from) {
            panic!("Feedback connection already exists for layer {}", from);
        }
        self.feedbacks.insert(from, to);
    }

    /// Extract the total number of parameters in the network.
    fn parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameters()).sum()
    }

    /// Set the `activation::Activation` function of a layer.
    ///
    /// # Arguments
    ///
    /// * `layer` - The index of the layer (in the `self.layers` vector).
    /// * `activation` - The new `activation::Activation` function to be used.
    ///
    /// # Panics
    ///
    /// * If the layer index is out of bounds.
    pub fn set_activation(&mut self, layer: usize, activation: activation::Activation) {
        if layer >= self.layers.len() {
            panic!("Invalid layer index");
        }
        match self.layers[layer] {
            Layer::Dense(ref mut layer) => {
                layer.activation = activation::Function::create(&activation)
            }
            Layer::Convolution(ref mut layer) => {
                layer.activation = activation::Function::create(&activation)
            }
        }
    }

    /// Set the `optimizer::Optimizer` function of the network.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The new `optimizer::Optimizer` function to be used.
    pub fn set_optimizer(&mut self, mut optimizer: optimizer::Optimizer) {
        // Create the placeholder vector used for various optimizer functions.
        // See the `match optimizer` below.
        let vector: Vec<Vec<Vec<Vec<Vec<f32>>>>> = self
            .layers
            .iter()
            .rev()
            .map(|layer| match layer {
                Layer::Dense(layer) => {
                    vec![vec![vec![
                        vec![0.0; layer.inputs];
                        layer.outputs
                            + if layer.bias.is_some() { 1 } else { 0 }
                    ]]]
                }
                Layer::Convolution(layer) => {
                    let (ch, kh, kw) = match layer.kernels[0].shape {
                        tensor::Shape::Tensor(ch, he, wi) => (ch, he, wi),
                        _ => panic!("Expected Convolution shape"),
                    };
                    vec![vec![vec![vec![0.0; kw]; kh]; ch]; layer.kernels.len()]
                }
            })
            .collect();

        match optimizer {
            optimizer::Optimizer::SGD(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.1;
                }
            }
            optimizer::Optimizer::SGDM(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.1;
                }
                if params.momentum == 0.0 {
                    params.momentum = 0.9;
                }
                params.velocity = vector.clone();
            }
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
            }
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
            }
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
            }
        };
        self.optimizer = optimizer;
    }

    /// Set the `objective::Objective` function of the network.
    ///
    /// # Arguments
    ///
    /// * `objective` - The new `objective::Objective` function to be used.
    /// * `clamp` - The clamp values for the objective function.
    pub fn set_objective(&mut self, objective: objective::Objective, clamp: Option<(f32, f32)>) {
        self.objective = objective::Function::create(objective, clamp);
    }

    /// Train the network on the given inputs and targets for the given number of epochs.
    ///
    /// Computes the forward and backward pass of the network for the given number of epochs,
    /// with respect to the given inputs and targets. The loss and gradient of the network is
    /// computed for each sample in the input data, and the weights and biases of the network are
    /// updated accordingly.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The individual inputs (x) stored in a vector.
    /// * `targets` - The respective individual (y) targets stored in a vector.
    /// * `epochs` - The number of epochs to train the network for.
    ///
    /// # Returns
    ///
    /// A vector of the average loss of the network per epoch.
    pub fn learn(
        &mut self,
        inputs: &Vec<tensor::Tensor>,
        targets: &Vec<tensor::Tensor>,
        epochs: i32,
    ) -> Vec<f32> {
        for layer in &mut self.layers {
            match layer {
                Layer::Dense(layer) => layer.training = true,
                Layer::Convolution(layer) => layer.training = true,
            }
        }

        let checkpoint = (epochs / 10).max(1);
        let mut losses = Vec::new();
        for epoch in 1..epochs + 1 {
            let mut _losses = 0.0f32;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let (unactivated, activated) = self.forward(input);
                let (loss, gradient) = self.loss(&activated.last().unwrap(), target);
                _losses += loss;

                // TODO: Backward pass on batch instead of single input.
                self.backward(epoch, gradient, &unactivated, &activated);
            }
            losses.push(_losses / inputs.len() as f32);

            if epoch % checkpoint == 0 && epoch > 0 {
                println!(
                    "Epoch: {} Loss: {}",
                    epoch,
                    losses[(epoch as usize) - (checkpoint as usize)..(epoch as usize)]
                        .iter()
                        .sum::<f32>()
                        / checkpoint as f32
                );
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
    pub fn forward(
        &mut self,
        input: &tensor::Tensor,
    ) -> (Vec<tensor::Tensor>, Vec<tensor::Tensor>) {
        let mut unactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];

        for i in 1..self.layers.len() + 1 {
            let (mut pre, mut post) = self._forward(activated.last().unwrap(), i - 1, i);

            if self.feedbacks.contains_key(&i) {
                let (fpre, fpost) = self._forward(post.last().unwrap(), self.feedbacks[&i], i);

                // Adding the forward pass (before feedback) to the unactivated and activated vectors.
                // This is done after calculating the forward pass of the fed-back layers.
                // Due to said pass needing `post.last()`.
                unactivated.append(&mut pre);
                activated.append(&mut post);

                for (idx, j) in (self.feedbacks[&i]..i).enumerate() {
                    // TODO: Handle the case with feedbacks.
                    // The values should be overwritten(?) summed(?) multiplied(?).

                    // // Overwriting.
                    // unactivated[j] = fpre[idx].to_owned();
                    // activated[j + 1] = fpost[idx].to_owned();

                    // Summing.
                    unactivated[j].add_inplace(&fpre[idx]);
                    activated[j + 1].add_inplace(&fpost[idx]);

                    // // Multiplying.
                    // unactivated[j].mul_inplace(&fpre[idx]);
                    // activated[j + 1].mul_inplace(&fpost[idx]);
                }
            } else {
                unactivated.append(&mut pre);
                activated.append(&mut post);
            }
        }

        (unactivated, activated)
    }

    /// Compute the forward pass for the specified range of layers.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data (x).
    /// * `from` - The starting index of the layers to compute the forward pass for.
    /// * `to` - The ending index of the layers to compute the forward pass for.
    ///
    /// # Returns
    ///
    /// A tuple containing the pre-activation and post-activation values of each layer inbetween.
    fn _forward(
        &mut self,
        input: &tensor::Tensor,
        from: usize,
        to: usize,
    ) -> (Vec<tensor::Tensor>, Vec<tensor::Tensor>) {
        let mut unactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];

        for layer in &self.layers[from..to] {
            let (pre, post): (tensor::Tensor, tensor::Tensor) = match layer {
                Layer::Dense(layer) => layer.forward(activated.last().unwrap()),
                Layer::Convolution(layer) => layer.forward(activated.last().unwrap()),
            };

            unactivated.push(pre);
            activated.push(post);
        }

        // Removing the input clone from the activated vector.
        // As this is present in the `forward` function.
        activated.remove(0);

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
    fn loss(
        &mut self,
        prediction: &tensor::Tensor,
        target: &tensor::Tensor,
    ) -> (f32, tensor::Tensor) {
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
        activated: &Vec<tensor::Tensor>,
    ) {
        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            let input: &tensor::Tensor = &activated[activated.len() - i - 2];
            let output: &tensor::Tensor = &unactivated[unactivated.len() - i - 1];

            let (_gradient, weight_gradient, bias_gradient) = match layer {
                Layer::Dense(layer) => layer.backward(&gradient, input, output),
                Layer::Convolution(layer) => layer.backward(&gradient, input, output),
            };
            gradient = _gradient;

            match layer {
                Layer::Dense(layer) => {
                    // Weight update.
                    for (j, (weights, gradients)) in layer
                        .weights
                        .iter_mut()
                        .zip(
                            match weight_gradient.data {
                                tensor::Data::Tensor(data) => data,
                                _ => panic!("Expected a tensor, but got one-dimensional data."),
                            }[0]
                            .iter_mut(),
                        )
                        .enumerate()
                    {
                        self.optimizer
                            .update(i, 0, 0, j, stepnr, weights, gradients);
                    }

                    // Bias update.
                    if let Some(ref mut bias) = layer.bias {
                        // Using `layer.weights.len()` as the bias' momentum/velocity is stored therein.
                        self.optimizer.update(
                            i,
                            0,
                            0,
                            layer.weights.len(),
                            stepnr,
                            bias,
                            &mut bias_gradient.unwrap().get_flat(),
                        );
                    }
                }
                Layer::Convolution(layer) => {
                    let mut weight_gradient = match weight_gradient.data {
                        tensor::Data::Gradient(data) => data,
                        _ => panic!("Expected four-dimensional data."),
                    };
                    for (f, (filter, gradients)) in layer
                        .kernels
                        .iter_mut()
                        .zip(weight_gradient.iter_mut())
                        .enumerate()
                    {
                        let filter = match &mut filter.data {
                            tensor::Data::Tensor(data) => data,
                            _ => panic!("Expected a tensor, but got one-dimensional data."),
                        };
                        for (c, (kernel, gradients)) in
                            filter.iter_mut().zip(gradients.iter_mut()).enumerate()
                        {
                            for (r, (weight, gradient)) in
                                kernel.iter_mut().zip(gradients.iter_mut()).enumerate()
                            {
                                self.optimizer.update(i, f, c, r, stepnr, weight, gradient);
                            }
                        }
                    }
                }
            }
        }
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
        &mut self,
        inputs: &Vec<tensor::Tensor>,
        targets: &Vec<tensor::Tensor>,
        tol: f32,
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
                Layer::Dense(layer) => match layer.activation {
                    activation::Function::Softmax(_) => {
                        let predicted = prediction
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .map(|(index, _)| index)
                            .unwrap() as f32;
                        let actual = target.iter().position(|&v| v == 1.0).unwrap() as f32;
                        accuracy.push(if (predicted - actual).abs() < tol {
                            1.0
                        } else {
                            0.0
                        });
                    }
                    _ => {
                        if target.len() == 1 {
                            accuracy.push(if (prediction[0] - target[0]).abs() < tol {
                                1.0
                            } else {
                                0.0
                            });
                        } else {
                            target.iter().zip(prediction.iter()).for_each(|(t, p)| {
                                accuracy.push(if (t - p).abs() < tol { 1.0 } else { 0.0 })
                            });
                        }
                    }
                },
                Layer::Convolution(_) => {
                    unimplemented!("Image output (target) not supported.")
                }
            };
        }

        (
            accuracy.iter().sum::<f32>() / accuracy.len() as f32,
            losses.iter().sum::<f32>() / inputs.len() as f32,
        )
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
    pub fn accuracy(
        &self,
        predictions: &Vec<tensor::Tensor>,
        targets: &Vec<tensor::Tensor>,
        tol: f32,
    ) -> f32 {
        let mut accuracy: Vec<f32> = Vec::new();

        let predictions: Vec<Vec<f32>> = predictions.iter().map(|t| t.get_flat()).collect();
        let targets: Vec<Vec<f32>> = targets.iter().map(|t| t.get_flat()).collect();

        for (prediction, target) in predictions.iter().zip(targets.iter()) {
            match self.layers.last().unwrap() {
                Layer::Dense(layer) => match layer.activation {
                    activation::Function::Softmax(_) => {
                        let predicted = prediction
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .map(|(index, _)| index)
                            .unwrap() as f32;
                        let actual = target.iter().position(|&v| v == 1.0).unwrap() as f32;
                        accuracy.push(if (predicted - actual).abs() < tol {
                            1.0
                        } else {
                            0.0
                        });
                    }
                    _ => {
                        if target.len() == 1 {
                            accuracy.push(if (prediction[0] - target[0]).abs() < tol {
                                1.0
                            } else {
                                0.0
                            });
                        } else {
                            target.iter().zip(prediction.iter()).for_each(|(t, p)| {
                                accuracy.push(if (t - p).abs() < tol { 1.0 } else { 0.0 })
                            });
                        }
                    }
                },
                Layer::Convolution(_) => {
                    unimplemented!("Image output (target) not supported.")
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
                }
                Layer::Convolution(layer) => {
                    let (_, out) = layer.forward(&output);
                    output = out;
                }
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
}
