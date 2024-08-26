// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::{activation, convolution, dense, maxpool, objective, optimizer, tensor};

use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};

/// Layer types of the network.
pub enum Layer {
    Dense(dense::Dense),
    Convolution(convolution::Convolution),
    Maxpool(maxpool::Maxpool),
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Layer::Dense(layer) => write!(f, "{}", layer),
            Layer::Convolution(layer) => write!(f, "{}", layer),
            Layer::Maxpool(layer) => write!(f, "{}", layer),
        }
    }
}

impl Layer {
    /// Extracts the number of parameters in the layer.
    fn parameters(&self) -> usize {
        match self {
            Layer::Dense(layer) => layer.parameters(),
            Layer::Convolution(layer) => layer.parameters(),
            Layer::Maxpool(_) => 0,
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

    pub layers: Vec<Layer>,
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
                    _ => panic!("Expected `tensor::Tensor` shape"),
                }
            }
            Layer::Maxpool(layer) => {
                // Make sure the output of the maxpool layer is flattened.
                layer.flatten_output = true;
                match layer.outputs {
                    tensor::Shape::Tensor(ch, he, wi) => ch * he * wi,
                    _ => panic!("Expected `tensor::Tensor` shape"),
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
                    Layer::Maxpool(layer) => layer.outputs.clone(),
                },
                filters,
                &activation,
                kernel,
                stride,
                padding,
                dropout,
            )));
    }

    /// Adds a new maxpool layer to the network.
    ///
    /// # Arguments
    ///
    /// * `kernel` - The shape of the filter.
    /// * `stride` - The stride of the filter.
    pub fn maxpool(&mut self, kernel: (usize, usize), stride: (usize, usize)) {
        if self.layers.is_empty() {
            self.layers.push(Layer::Maxpool(maxpool::Maxpool::create(
                self.input.clone(),
                kernel,
                stride,
            )));
            return;
        }
        let input = match self.layers.last().unwrap() {
            Layer::Dense(layer) => tensor::Shape::Vector(layer.outputs),
            Layer::Convolution(layer) => layer.outputs.clone(),
            Layer::Maxpool(layer) => layer.outputs.clone(),
        };
        self.layers.push(Layer::Maxpool(maxpool::Maxpool::create(
            input, kernel, stride,
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
            _ => panic!("Maxpool layers do not use activation functions!"),
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
                Layer::Maxpool(_) => vec![vec![vec![vec![0.0; 0]; 0]; 0]; 0],
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
                params.velocity = vector;
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
                params.momentum = vector;
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
                params.momentum = vector;
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
                params.buffer = vector;
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
    /// * `batch` - The batch size to use when training.
    /// * `epochs` - The number of epochs to train the network for.
    ///
    /// # Returns
    ///
    /// A vector of the average loss of the network per epoch.
    pub fn learn(
        &mut self,
        inputs: &Vec<&tensor::Tensor>,
        targets: &Vec<&tensor::Tensor>,
        batch: usize,
        epochs: i32,
    ) -> Vec<f32> {
        self.layers.iter_mut().for_each(|layer| match layer {
            Layer::Dense(layer) => layer.training = true,
            Layer::Convolution(layer) => layer.training = true,
            _ => (),
        });

        let print = 1; //(epochs / 10).max(1);
        let mut losses = Vec::new();

        let split_at = (0.8 * inputs.len() as f32) as usize; // 80% for training, 20% for validation
        let (train_inputs, val_inputs) = inputs.split_at(split_at);
        let (train_targets, val_targets) = targets.split_at(split_at);

        // Split the input data into batches.
        let batches: Vec<(&[&tensor::Tensor], &[&tensor::Tensor])> = train_inputs
            .par_chunks(batch)
            .zip(train_targets.par_chunks(batch))
            .collect();

        for epoch in 1..epochs + 1 {
            let results: Vec<_> = batches
                .par_iter()
                .map(|(inputs, targets)| {
                    let mut loss = 0.0f32;
                    let mut weight_gradients: Vec<tensor::Tensor> = Vec::new();
                    let mut bias_gradients: Vec<Option<tensor::Tensor>> = Vec::new();

                    for (i, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
                        let (unactivated, activated, maxpools) = self.forward(input);
                        let (_loss, gradient) =
                            self.objective.loss(&activated.last().unwrap(), target);
                        loss += _loss;

                        let (wg, bg) = self.backward(gradient, &unactivated, &activated, maxpools);

                        if i == 0 {
                            weight_gradients = wg;
                            bias_gradients = bg;
                        } else {
                            for (gradient, new) in weight_gradients.iter_mut().zip(wg.iter()) {
                                gradient.add_inplace(new)
                            }

                            for (gradient, new) in bias_gradients.iter_mut().zip(bg.iter()) {
                                match gradient {
                                    Some(gradient) => match new {
                                        Some(new) => gradient.add_inplace(new),
                                        None => panic!("Expected Some, got None."),
                                    },
                                    None => match new {
                                        Some(_) => panic!("Expected None, got Some."),
                                        None => (),
                                    },
                                }
                            }
                        }
                    }

                    let size = inputs.len() as f32;

                    if size > 1.0 {
                        weight_gradients
                            .iter_mut()
                            .for_each(|gradient| gradient.div_scalar_inplace(size));
                        bias_gradients.iter_mut().for_each(|gradient| {
                            if let Some(gradient) = gradient {
                                gradient.div_scalar_inplace(size);
                            }
                        });
                    }

                    (loss / size, weight_gradients, bias_gradients)
                })
                .collect();

            let mut _losses: Vec<f32> = Vec::new();
            for (loss, weight_gradients, bias_gradients) in results {
                self.update(epoch, weight_gradients, bias_gradients);
                _losses.push(loss);
            }
            losses.push(_losses.iter().sum::<f32>() / _losses.len() as f32);

            let val_results: Vec<_> = val_inputs
                .par_iter()
                .zip(val_targets.par_iter())
                .map(|(input, target)| {
                    let (_, activated, _) = self.forward(input);
                    let (loss, _) = self.objective.loss(&activated.last().unwrap(), target);
                    let pred = activated.last().unwrap().argmax();
                    let target = target.argmax();
                    let acc = if pred == target { 1.0 } else { 0.0 };
                    (loss, acc)
                })
                .collect();

            let val_loss: f32 = val_results.iter().map(|(loss, _)| *loss).sum();
            let val_acc: f32 = val_results.iter().map(|(_, acc)| *acc).sum();
            println!(
                "Validation Loss: {}, Validation Accuracy: {}",
                val_loss / val_inputs.len() as f32,
                val_acc / val_inputs.len() as f32
            );

            if epoch % print == 0 && epoch > 0 {
                println!(
                    "Epoch: {} Loss: {}",
                    epoch,
                    losses[(epoch as usize) - (print as usize)..(epoch as usize)]
                        .iter()
                        .sum::<f32>()
                        / print as f32
                );
            }
        }
        for layer in &mut self.layers {
            match layer {
                Layer::Dense(layer) => layer.training = false,
                Layer::Convolution(layer) => layer.training = false,
                _ => (),
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
    /// A tuple containing the pre- and post-activation values and the maxpool indices (if any) of each layer.
    pub fn forward(
        &self,
        input: &tensor::Tensor,
    ) -> (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        VecDeque<Vec<Vec<Vec<(usize, usize)>>>>,
    ) {
        let mut unactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];
        let mut maxpools: VecDeque<Vec<Vec<Vec<(usize, usize)>>>> = VecDeque::new();

        for i in 1..self.layers.len() + 1 {
            let (mut pre, mut post, mut max) = self._forward(activated.last().unwrap(), i - 1, i);

            if self.feedbacks.contains_key(&i) {
                let (fpre, fpost, fmax) =
                    self._forward(post.last().unwrap(), self.feedbacks[&i], i);

                // Adding the forward pass (before feedback) to the unactivated and activated vectors.
                // This is done after calculating the forward pass of the fed-back layers.
                // Due to said pass needing `post.last()`.
                unactivated.append(&mut pre);
                activated.append(&mut post);
                maxpools.append(&mut max);

                for (idx, j) in (self.feedbacks[&i]..i).enumerate() {
                    // TODO: Handle the case with feedbacks.
                    // TODO: Handle maxpool indices.
                    // The values should be overwritten(?) summed(?) multiplied(?).

                    // // Overwriting.
                    // unactivated[j] = fpre[idx].to_owned();
                    // activated[j + 1] = fpost[idx].to_owned();
                    // maxpools[j] = fmax[idx].to_owned();

                    // Summing.
                    unactivated[j].add_inplace(&fpre[idx]);
                    activated[j + 1].add_inplace(&fpost[idx]);
                    // maxpools[j].add_inplace(&fmax[idx]);

                    // // Multiplying.
                    // unactivated[j].mul_inplace(&fpre[idx]);
                    // activated[j + 1].mul_inplace(&fpost[idx]);
                    // maxpools[j].mul_inplace(&fmax[idx]);
                }
            } else {
                unactivated.append(&mut pre);
                activated.append(&mut post);
                maxpools.append(&mut max);
            }
        }

        (unactivated, activated, maxpools)
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
    /// A tuple containing the pre- and post-activation values and the maxpool indices (if any) of each layer inbetween.
    fn _forward(
        &self,
        input: &tensor::Tensor,
        from: usize,
        to: usize,
    ) -> (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        VecDeque<Vec<Vec<Vec<(usize, usize)>>>>,
    ) {
        let mut unactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];
        let mut maxpools: VecDeque<Vec<Vec<Vec<(usize, usize)>>>> = VecDeque::new();

        for layer in &self.layers[from..to] {
            match layer {
                Layer::Dense(layer) => {
                    let (pre, post) = layer.forward(activated.last().unwrap());
                    unactivated.push(pre);
                    activated.push(post);
                }
                Layer::Convolution(layer) => {
                    let (pre, post) = layer.forward(activated.last().unwrap());
                    unactivated.push(pre);
                    activated.push(post);
                }
                Layer::Maxpool(layer) => {
                    let (pre, post, max) = layer.forward(activated.last().unwrap());
                    unactivated.push(pre);
                    activated.push(post);
                    maxpools.push_back(max);
                }
            };
        }

        // Removing the input clone from the activated vector.
        // As this is present in the `forward` function.
        activated.remove(0);

        (unactivated, activated, maxpools)
    }

    /// Compute the backward pass of the network for the given output gradient.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient of the output.
    /// * `unactivated` - The pre-activation values of each layer.
    /// * `activated` - The post-activation values of each layer.
    /// * `maxpools` - The maxpool indices of each maxpool-layer.
    ///
    /// # Returns
    ///
    /// A tuple containing the weight and bias gradients of each layer.
    pub fn backward(
        &self,
        mut gradient: tensor::Tensor,
        unactivated: &Vec<tensor::Tensor>,
        activated: &Vec<tensor::Tensor>,
        mut maxpools: VecDeque<Vec<Vec<Vec<(usize, usize)>>>>,
    ) -> (Vec<tensor::Tensor>, Vec<Option<tensor::Tensor>>) {
        let mut weight_gradient: Vec<tensor::Tensor> = Vec::new();
        let mut bias_gradient: Vec<Option<tensor::Tensor>> = Vec::new();
        self.layers.iter().rev().enumerate().for_each(|(i, layer)| {
            let input: &tensor::Tensor = &activated[activated.len() - i - 2];
            let output: &tensor::Tensor = &unactivated[unactivated.len() - i - 1];

            let (input_gradient, wg, bg) = match layer {
                Layer::Dense(layer) => layer.backward(&gradient, input, output),
                Layer::Convolution(layer) => layer.backward(&gradient, input, output),
                Layer::Maxpool(layer) => (
                    layer.backward(&gradient, &maxpools.pop_front().unwrap()),
                    tensor::Tensor::from_single(vec![0.0; 0]),
                    None,
                ),
            };

            weight_gradient.push(wg);
            bias_gradient.push(bg);

            gradient = input_gradient;
        });
        (weight_gradient, bias_gradient)
    }

    /// Update the weights and biases of the network using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `stepnr` - The current step number (i.e., epoch number).
    /// * `weight_gradients` - The weight gradients of each layer.
    /// * `bias_gradients` - The bias gradients of each layer.
    pub fn update(
        &mut self,
        stepnr: i32,
        weight_gradients: Vec<tensor::Tensor>,
        bias_gradients: Vec<Option<tensor::Tensor>>,
    ) {
        self.layers
            .iter_mut()
            .rev()
            .enumerate()
            .for_each(|(i, layer)| {
                match layer {
                    Layer::Dense(layer) => {
                        // Weight update.
                        let mut weight_gradient = match &weight_gradients[i].data {
                            tensor::Data::Tensor(data) => data.clone(),
                            _ => panic!("Expected four-dimensional data."),
                        };

                        for (j, (weights, gradients)) in layer
                            .weights
                            .iter_mut()
                            .zip(weight_gradient[0].iter_mut())
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
                                &mut bias_gradients[i].as_ref().unwrap().get_flat(),
                            );
                        }
                    }
                    Layer::Convolution(layer) => {
                        let mut weight_gradient = match &weight_gradients[i].data {
                            tensor::Data::Gradient(data) => data.clone(),
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
                    Layer::Maxpool(_) => {}
                }
            });
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
        inputs: &Vec<&tensor::Tensor>,
        targets: &Vec<&tensor::Tensor>,
        tol: f32,
    ) -> (f32, f32) {
        let mut losses = Vec::new();
        let mut accuracy = Vec::new();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict(input);
            let (loss, _) = self.objective.loss(&prediction, target);

            losses.push(loss);

            match self.layers.last().unwrap() {
                Layer::Dense(layer) => match layer.activation {
                    activation::Function::Softmax(_) => {
                        accuracy.push(if target.argmax() == prediction.argmax() {
                            1.0
                        } else {
                            0.0
                        });
                    }
                    _ => {
                        let target = target.get_flat();
                        let prediction = prediction.get_flat();

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
                Layer::Maxpool(_) => {
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
        predictions: &Vec<&tensor::Tensor>,
        targets: &Vec<&tensor::Tensor>,
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
                Layer::Maxpool(_) => {
                    unimplemented!("Maxpool output (target) not supported.")
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
                Layer::Maxpool(layer) => {
                    let (_, out, _) = layer.forward(&output);
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
    pub fn predict_batch(&self, inputs: &Vec<&tensor::Tensor>) -> Vec<tensor::Tensor> {
        inputs.iter().map(|input| self.predict(input)).collect()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::assert_eq_data;

    #[test]
    fn test_forward() {
        let mut network = Network::new(tensor::Shape::Tensor(1, 3, 3));

        network.convolution(
            1,
            (3, 3),
            (1, 1),
            (1, 1),
            activation::Activation::Linear,
            None,
        );

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                conv.kernels[0] = tensor::Tensor::from(vec![vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                ]]);
            }
            _ => (),
        }

        let input = tensor::Tensor::from(vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]]);

        let (pre, post, _) = network.forward(&input);

        assert_eq_data!(pre.last().unwrap().data, input.data);
        assert_eq_data!(post.last().unwrap().data, input.data);
    }

    #[test]
    fn test_backward() {
        // See Python file `validation/test_network_backward.py` for the reference implementation.

        let mut network = Network::new(tensor::Shape::Tensor(2, 4, 4));

        network.convolution(
            3,
            (2, 2),
            (1, 1),
            (0, 0),
            activation::Activation::ReLU,
            None,
        );
        network.dense(5, activation::Activation::ReLU, true, None);

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                conv.kernels[0] = tensor::Tensor::from(vec![
                    vec![vec![1.0, 1.0], vec![2.0, 2.0]],
                    vec![vec![1.0, 2.0], vec![1.0, 2.0]],
                ]);
                conv.kernels[1] = tensor::Tensor::from(vec![
                    vec![vec![2.0, 2.0], vec![1.0, 1.0]],
                    vec![vec![2.0, 1.0], vec![2.0, 1.0]],
                ]);
                conv.kernels[2] = tensor::Tensor::from(vec![
                    vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                    vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                ]);
            }
            _ => (),
        }
        match network.layers[1] {
            Layer::Dense(ref mut dense) => {
                dense.weights = vec![
                    vec![2.5; dense.inputs],
                    vec![-1.2; dense.inputs],
                    vec![0.5; dense.inputs],
                    vec![3.5; dense.inputs],
                    vec![5.2; dense.inputs],
                ];
                dense.bias = Some(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
            }
            _ => (),
        }

        let input = tensor::Tensor::from(vec![
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 2.0, 0.0],
                vec![0.0, 3.0, 4.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ],
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 4.0, 3.0, 0.0],
                vec![0.0, 2.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ],
        ]);

        let output = vec![
            tensor::Tensor::from(vec![
                vec![
                    vec![10.0, 16.0, 7.0],
                    vec![19.0, 31.0, 14.0],
                    vec![7.0, 11.0, 5.0],
                ],
                vec![
                    vec![5.0, 14.0, 8.0],
                    vec![11.0, 29.0, 16.0],
                    vec![8.0, 19.0, 10.0],
                ],
                vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                ],
            ]),
            tensor::Tensor::from_single(vec![
                10.0, 16.0, 7.0, 19.0, 31.0, 14.0, 7.0, 11.0, 5.0, 5.0, 14.0, 8.0, 11.0, 29.0,
                16.0, 8.0, 19.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
            tensor::Tensor::from_single(vec![603.0, -284.00003, 125.0, 846.0, 1255.0]),
            tensor::Tensor::from_single(vec![603.0, 0.0, 125.0, 846.0, 1255.0]),
        ];

        let (pre, post, max) = network.forward(&input);

        assert_eq_data!(pre[0].data, output[0].data);
        assert_eq_data!(post[1].data, output[1].data);

        assert_eq_data!(pre[1].data, output[2].data);
        assert_eq_data!(post[2].data, output[3].data);

        // let gradient = tensor::Tensor::from(vec![vec![vec![1.0; 3]; 3]; 3]);
        let gradient = tensor::Tensor::from_single(vec![1.0; 5]);

        let (weight_gradient, bias_gradient) = network.backward(gradient, &pre, &post, max);

        let _weight_gradient = vec![
            tensor::Tensor::gradient(vec![
                vec![vec![vec![117., 117.]; 2]; 2],
                vec![vec![vec![117., 117.]; 2]; 2],
                vec![vec![vec![0.0, 0.0]; 2]; 2],
            ]),
            tensor::Tensor::from(vec![vec![
                vec![
                    10., 16., 7., 19., 31., 14., 7., 11., 5., 5., 14., 8., 11., 29., 16., 8., 19.,
                    10., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                ],
                vec![
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0.,
                ],
                vec![
                    10., 16., 7., 19., 31., 14., 7., 11., 5., 5., 14., 8., 11., 29., 16., 8., 19.,
                    10., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                ],
                vec![
                    10., 16., 7., 19., 31., 14., 7., 11., 5., 5., 14., 8., 11., 29., 16., 8., 19.,
                    10., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                ],
                vec![
                    10., 16., 7., 19., 31., 14., 7., 11., 5., 5., 14., 8., 11., 29., 16., 8., 19.,
                    10., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                ],
            ]]),
            tensor::Tensor::from_single(vec![1., 0., 1., 1., 1.]),
        ];

        // Kernel gradient(s)
        assert_eq_data!(weight_gradient[1].data, _weight_gradient[0].data);

        // Fully connected layer gradient
        assert_eq_data!(weight_gradient[0].data, _weight_gradient[1].data);
        if let Some(bias) = bias_gradient.last().unwrap() {
            assert_eq_data!(bias.data, _weight_gradient[2].data);
        }
    }

    #[test]
    fn test_update() {
        // See Python file `validation/test_network_update.py` for the reference implementation.

        let mut network = Network::new(tensor::Shape::Tensor(2, 4, 4));

        network.convolution(
            3,
            (2, 2),
            (1, 1),
            (0, 0),
            activation::Activation::ReLU,
            None,
        );
        network.dense(5, activation::Activation::ReLU, true, None);

        network.set_optimizer(optimizer::Optimizer::SGD(optimizer::SGD {
            learning_rate: 0.1,
            decay: None,
        }));

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                conv.kernels[0] = tensor::Tensor::from(vec![
                    vec![vec![1.0, 1.0], vec![2.0, 2.0]],
                    vec![vec![1.0, 2.0], vec![1.0, 2.0]],
                ]);
                conv.kernels[1] = tensor::Tensor::from(vec![
                    vec![vec![2.0, 2.0], vec![1.0, 1.0]],
                    vec![vec![2.0, 1.0], vec![2.0, 1.0]],
                ]);
                conv.kernels[2] = tensor::Tensor::from(vec![
                    vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                    vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                ]);
            }
            _ => (),
        }
        match network.layers[1] {
            Layer::Dense(ref mut dense) => {
                dense.weights = vec![
                    vec![2.5; dense.inputs],
                    vec![-1.2; dense.inputs],
                    vec![0.5; dense.inputs],
                    vec![3.5; dense.inputs],
                    vec![5.2; dense.inputs],
                ];
                dense.bias = Some(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
            }
            _ => (),
        }

        let input = tensor::Tensor::from(vec![
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 2.0, 0.0],
                vec![0.0, 3.0, 4.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ],
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 4.0, 3.0, 0.0],
                vec![0.0, 2.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ],
        ]);

        let (pre, post, max) = network.forward(&input);

        let gradient = tensor::Tensor::from_single(vec![1.0; 5]);

        let (weight_gradients, bias_gradients) = network.backward(gradient, &pre, &post, max);

        network.update(0, weight_gradients, bias_gradients);

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                assert_eq_data!(
                    conv.kernels[0].data,
                    tensor::Tensor::from(vec![
                        vec![vec![-10.7, -10.7], vec![-9.7000, -9.7000]],
                        vec![vec![-10.7, -9.7000], vec![-10.7, -9.7000]],
                    ])
                    .data
                );
                assert_eq_data!(
                    conv.kernels[1].data,
                    tensor::Tensor::from(vec![
                        vec![vec![-9.7000, -9.7000], vec![-10.7, -10.7]],
                        vec![vec![-9.7000, -10.7], vec![-9.7000, -10.7]],
                    ])
                    .data
                );
                assert_eq_data!(
                    conv.kernels[2].data,
                    tensor::Tensor::from(vec![
                        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                    ])
                    .data
                );
            }
            _ => (),
        }
        match network.layers[1] {
            Layer::Dense(ref mut dense) => {
                assert_eq_data!(
                    tensor::Data::Tensor(vec![dense.weights.clone()]),
                    tensor::Data::Tensor(vec![vec![
                        vec![
                            1.5000, 0.9000, 1.8000, 0.6000, -0.6000, 1.1000, 1.8000, 1.4000,
                            2.0000, 2.0000, 1.1000, 1.7000, 1.4000, -0.4000, 0.9000, 1.7000,
                            0.6000, 1.5000, 2.5000, 2.5000, 2.5000, 2.5000, 2.5000, 2.5000, 2.5000,
                            2.5000, 2.5000
                        ],
                        vec![
                            -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000,
                            -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000,
                            -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000,
                            -1.2000, -1.2000, -1.2000
                        ],
                        vec![
                            -0.5000, -1.1000, -0.2000, -1.4000, -2.6000, -0.9000, -0.2000, -0.6000,
                            0.0000, 0.0000, -0.9000, -0.3000, -0.6000, -2.4000, -1.1000, -0.3000,
                            -1.4000, -0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                            0.5000, 0.5000, 0.5000
                        ],
                        vec![
                            2.5000, 1.9000, 2.8000, 1.6000, 0.4000, 2.1000, 2.8000, 2.4000, 3.0000,
                            3.0000, 2.1000, 2.7000, 2.4000, 0.6000, 1.9000, 2.7000, 1.6000, 2.5000,
                            3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000
                        ],
                        vec![
                            4.2000, 3.6000, 4.5000, 3.3000, 2.1000, 3.8000, 4.5000, 4.1000, 4.7000,
                            4.7000, 3.8000, 4.4000, 4.1000, 2.3000, 3.6000, 4.4000, 3.3000, 4.2000,
                            5.2000, 5.2000, 5.2000, 5.2000, 5.2000, 5.2000, 5.2000, 5.2000, 5.2000
                        ],
                    ]])
                );
                assert_eq!(
                    dense.bias,
                    Some(vec![2.9000, 4.0000, 4.9000, 5.9000, 6.9000])
                );
            }
            _ => (),
        }
    }
}
