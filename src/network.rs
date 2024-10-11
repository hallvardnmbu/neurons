// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::{
    activation, assert_eq_shape, convolution, deconvolution, dense, feedback, maxpool, objective,
    optimizer, tensor,
};

use rayon::prelude::*;

use std::collections::HashMap;
use std::sync::Arc;

/// Layer types of the network.
#[derive(Clone)]
pub enum Layer {
    Dense(dense::Dense),
    Convolution(convolution::Convolution),
    Deconvolution(deconvolution::Deconvolution),
    Maxpool(maxpool::Maxpool),
    Feedback(feedback::Feedback),
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Layer::Dense(layer) => write!(f, "{}", layer),
            Layer::Convolution(layer) => write!(f, "{}", layer),
            Layer::Deconvolution(layer) => write!(f, "{}", layer),
            Layer::Maxpool(layer) => write!(f, "{}", layer),
            Layer::Feedback(layer) => write!(f, "{}", layer),
        }
    }
}

impl Layer {
    /// Extracts the number of parameters in the layer.
    fn parameters(&self) -> usize {
        match self {
            Layer::Dense(layer) => layer.parameters(),
            Layer::Convolution(layer) => layer.parameters(),
            Layer::Deconvolution(layer) => layer.parameters(),
            Layer::Feedback(layer) => layer.parameters(),
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
/// * `loopbacks` - The looped connections of the network.
/// * `connect` - The skip connections of the network.
/// * `accumulation` - The accumulation type of the network for looped- and skip connections.
/// * `optimizer` - The `optimizer::Optimizer` function of the network.
/// * `objective` - The `objective::Function` of the network.
pub struct Network {
    input: tensor::Shape,

    layers: Vec<Layer>,
    loopbacks: HashMap<usize, usize>,
    connect: HashMap<usize, usize>,
    accumulation: feedback::Accumulation,

    optimizer: optimizer::Optimizer,
    objective: objective::Function,
}

impl std::fmt::Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Network (\n")?;

        write!(f, "\toptimizer: (\n{}\n\t)\n", self.optimizer)?;
        write!(f, "\tobjective: (\n{}\n\t)\n", self.objective)?;

        write!(f, "\tlayers: (\n")?;
        for (i, layer) in self.layers.iter().enumerate() {
            write!(f, "\t\t{}: {}\n", i, layer)?;
        }
        write!(f, "\t)\n")?;
        if !self.connect.is_empty() {
            write!(f, "\tconnections: (\n")?;
            write!(f, "\t\taccumulation: {}\n", self.accumulation)?;

            let mut entries: Vec<(&usize, &usize)> = self.connect.iter().collect();
            entries.sort_by_key(|&(to, _)| to);
            for (to, from) in entries.iter() {
                write!(f, "\t\t{}.input -> {}.input\n", from, to)?;
            }
            write!(f, "\t)\n")?;
        }
        if !self.loopbacks.is_empty() {
            write!(f, "\tloops: (\n")?;
            write!(f, "\t\taccumulation: {}\n", self.accumulation)?;

            let mut entries: Vec<(&usize, &usize)> = self.loopbacks.iter().collect();
            entries.sort_by_key(|&(to, _)| to);
            for (from, to) in entries.iter() {
                write!(f, "\t\t{}.output -> {}.input\n", from, to)?;
            }
            write!(f, "\t)\n")?;
        }
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
            loopbacks: HashMap::new(),
            connect: HashMap::new(),
            accumulation: feedback::Accumulation::Mean,
            optimizer: optimizer::SGD::create(0.1, None),
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
            match self.input {
                tensor::Shape::Single(_) => (),
                _ => panic!(
                    "Network is configured for image inputs; the first layer cannot be dense. Modify the input shape to `tensor::Shape::Single` or add a convolutional layer first."
                ),
            };
            self.layers.push(Layer::Dense(dense::Dense::create(
                self.input.clone(),
                tensor::Shape::Single(outputs),
                &activation,
                bias,
                dropout,
            )));
            return;
        }
        let inputs = match &mut self.layers.last_mut().unwrap() {
            Layer::Dense(layer) => layer.outputs.clone(),
            // If the previous layer is convolutional or maxpool:
            // * Compute the flattened shape of the output.
            // * Set the `flatten` flag to `true`.
            Layer::Convolution(layer) => {
                layer.flatten = true;
                match layer.outputs {
                    tensor::Shape::Triple(ch, he, wi) => tensor::Shape::Single(ch * he * wi),
                    _ => panic!("Expected `tensor::Tensor` shape."),
                }
            }
            Layer::Deconvolution(layer) => {
                layer.flatten = true;
                match layer.outputs {
                    tensor::Shape::Triple(ch, he, wi) => tensor::Shape::Single(ch * he * wi),
                    _ => panic!("Expected `tensor::Tensor` shape."),
                }
            }
            Layer::Maxpool(layer) => {
                layer.flatten = true;
                match layer.outputs {
                    tensor::Shape::Triple(ch, he, wi) => tensor::Shape::Single(ch * he * wi),
                    _ => panic!("Expected `tensor::Tensor` shape."),
                }
            }
            Layer::Feedback(layer) => match layer.outputs {
                tensor::Shape::Single(_) => layer.outputs.clone(),
                tensor::Shape::Triple(ch, he, wi) => {
                    layer.flatten = true;
                    tensor::Shape::Single(ch * he * wi)
                }
                _ => panic!("Expected `tensor::Tensor` shape."),
            },
        };

        self.layers.push(Layer::Dense(dense::Dense::create(
            inputs,
            tensor::Shape::Single(outputs),
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
    /// * `dilation` - The dilation of the kernel.
    /// * `activation` - The `activation::Activation` function of the layer.
    /// * `dropout` - The dropout rate of the layer (applied during training).
    pub fn convolution(
        &mut self,
        filters: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        activation: activation::Activation,
        dropout: Option<f32>,
    ) {
        if self.layers.is_empty() {
            match self.input {
                tensor::Shape::Triple(_, _, _) => (),
                _ => panic!(
                    "Network is configured for dense inputs; the first layer cannot be convolutional. Modify the input shape to `tensor::Shape::Triple` or add a dense layer first."
                ),
            };
            self.layers
                .push(Layer::Convolution(convolution::Convolution::create(
                    self.input.clone(),
                    filters,
                    &activation,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    dropout,
                )));
            return;
        }
        self.layers
            .push(Layer::Convolution(convolution::Convolution::create(
                match self.layers.last().unwrap() {
                    Layer::Dense(layer) => layer.outputs.clone(),
                    Layer::Convolution(layer) => layer.outputs.clone(),
                    Layer::Deconvolution(layer) => layer.outputs.clone(),
                    Layer::Maxpool(layer) => layer.outputs.clone(),
                    Layer::Feedback(layer) => layer.outputs.clone(),
                },
                filters,
                &activation,
                kernel,
                stride,
                padding,
                dilation,
                dropout,
            )));
    }

    /// Adds a new deconvolutional layer to the network.
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
    pub fn deconvolution(
        &mut self,
        filters: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        activation: activation::Activation,
        dropout: Option<f32>,
    ) {
        if self.layers.is_empty() {
            match self.input {
                    tensor::Shape::Triple(_, _, _) => (),
                    _ => panic!(
                        "Network is configured for dense inputs; the first layer cannot be convolutional. Modify the input shape to `tensor::Shape::Triple` or add a dense layer first."
                    ),
                };
            self.layers
                .push(Layer::Deconvolution(deconvolution::Deconvolution::create(
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
            .push(Layer::Deconvolution(deconvolution::Deconvolution::create(
                match self.layers.last().unwrap() {
                    Layer::Dense(layer) => layer.outputs.clone(),
                    Layer::Convolution(layer) => layer.outputs.clone(),
                    Layer::Deconvolution(layer) => layer.outputs.clone(),
                    Layer::Maxpool(layer) => layer.outputs.clone(),
                    Layer::Feedback(layer) => layer.outputs.clone(),
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
            match self.input {
                tensor::Shape::Triple(_, _, _) => (),
                _ => panic!(
                    "Network is configured for dense inputs; the first layer cannot be maxpool. Modify the input shape to `tensor::Shape::Triple` or add a dense layer first."
                ),
            };
            self.layers.push(Layer::Maxpool(maxpool::Maxpool::create(
                self.input.clone(),
                kernel,
                stride,
            )));
            return;
        }
        self.layers.push(Layer::Maxpool(maxpool::Maxpool::create(
            match self.layers.last().unwrap() {
                Layer::Dense(layer) => layer.outputs.clone(),
                Layer::Convolution(layer) => layer.outputs.clone(),
                Layer::Deconvolution(layer) => layer.outputs.clone(),
                Layer::Maxpool(layer) => layer.outputs.clone(),
                Layer::Feedback(layer) => layer.outputs.clone(),
            },
            kernel,
            stride,
        )));
    }

    /// Adds a new feedback block to the network.
    ///
    /// # Arguments
    ///
    /// * `layers` - The layers of the feedback block.
    /// * `loops` - The number of loops in the feedback block.
    ///
    /// # Notes
    ///
    /// * The feedback block must have at least one layer.
    /// * The input and output shapes of the feedback block must match.
    ///   - To allow for loops.
    pub fn feedback(&mut self, layers: Vec<feedback::Layer>, loops: usize, skips: bool) {
        assert!(
            !layers.is_empty(),
            "Feedback block must have at least one layer."
        );

        // Convert `feedback::Layer` to `Layer`.
        let mut _layers: Vec<Layer> = Vec::new();

        let mut input: tensor::Shape = {
            if self.layers.is_empty() {
                self.input.clone()
            } else {
                match self.layers.last().unwrap() {
                    Layer::Dense(layer) => layer.outputs.clone(),
                    Layer::Convolution(layer) => layer.outputs.clone(),
                    Layer::Deconvolution(layer) => layer.outputs.clone(),
                    Layer::Maxpool(layer) => layer.outputs.clone(),
                    Layer::Feedback(layer) => layer.outputs.clone(),
                }
            }
        };
        for layer in layers.iter() {
            _layers.push(match layer {
                feedback::Layer::Dense(outputs, activation, bias, dropout) => {
                    Layer::Dense(dense::Dense::create(
                        input.clone(),
                        tensor::Shape::Single(*outputs),
                        activation,
                        *bias,
                        *dropout,
                    ))
                }
                feedback::Layer::Convolution(
                    filters,
                    activation,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    dropout,
                ) => Layer::Convolution(convolution::Convolution::create(
                    input.clone(),
                    *filters,
                    activation,
                    *kernel,
                    *stride,
                    *padding,
                    *dilation,
                    *dropout,
                )),
                feedback::Layer::Deconvolution(
                    filters,
                    activation,
                    kernel,
                    stride,
                    padding,
                    dropout,
                ) => Layer::Deconvolution(deconvolution::Deconvolution::create(
                    input.clone(),
                    *filters,
                    activation,
                    *kernel,
                    *stride,
                    *padding,
                    *dropout,
                )),
                feedback::Layer::Maxpool(kernel, stride) => {
                    Layer::Maxpool(maxpool::Maxpool::create(input.clone(), *kernel, *stride))
                }
            });
            input = match _layers.last().unwrap() {
                Layer::Dense(layer) => layer.outputs.clone(),
                Layer::Convolution(layer) => layer.outputs.clone(),
                Layer::Deconvolution(layer) => layer.outputs.clone(),
                Layer::Maxpool(layer) => layer.outputs.clone(),
                Layer::Feedback(layer) => layer.outputs.clone(),
            };
        }

        self.layers.push(Layer::Feedback(feedback::Feedback::create(
            _layers,
            loops,
            skips,
            self.accumulation.clone(),
        )));
    }

    /// Add a loop connection between two layers.
    /// Only supports loop connections for identical shapes.
    ///
    /// # Arguments
    ///
    /// * `outof` - The index of the layer to connect from (output).
    /// * `into` - The index of the layer to connect to (input).
    /// * `scale` - The scaling function of the loop connection wrt. gradients.
    pub fn loopback(&mut self, outof: usize, into: usize, scale: tensor::Scale) {
        if outof > self.layers.len() || into >= self.layers.len() || outof < into {
            panic!("Invalid layer indices for loop connection.");
        } else if self.loopbacks.contains_key(&outof) {
            panic!("Loop connection already exists for layer {}", outof);
        }

        let inputs = match &self.layers[into] {
            Layer::Dense(layer) => &layer.inputs,
            Layer::Convolution(layer) => &layer.inputs,
            Layer::Deconvolution(layer) => &layer.inputs,
            Layer::Maxpool(layer) => &layer.inputs,
            Layer::Feedback(feedback) => &feedback.inputs,
        };
        let outputs = match &self.layers[outof] {
            Layer::Dense(layer) => &layer.outputs,
            Layer::Convolution(layer) => &layer.outputs,
            Layer::Deconvolution(layer) => &layer.outputs,
            Layer::Maxpool(layer) => &layer.outputs,
            Layer::Feedback(feedback) => &feedback.outputs,
        };
        assert_eq_shape!(inputs, outputs);

        // Loop through layers into -> outof and add +1 to its loopback count.
        for k in into..outof + 1 {
            match &mut self.layers[k] {
                Layer::Dense(layer) => {
                    layer.scale = Arc::clone(&scale);
                    layer.loops += 1.0
                }
                Layer::Convolution(layer) => {
                    layer.scale = Arc::clone(&scale);
                    layer.loops += 1.0
                }
                Layer::Deconvolution(layer) => {
                    layer.scale = Arc::clone(&scale);
                    layer.loops += 1.0
                }
                Layer::Maxpool(layer) => layer.loops += 1.0,
                Layer::Feedback(_) => panic!("Loop connection includes feedback block."),
            }
        }

        // Store the loop connection for use in the forward pass.
        self.loopbacks.insert(outof, into);
    }

    /// Add a skip connection between two layers.
    /// Note: The `infrom` and `into` refer to their inputs.
    /// I.e., `infrom = 0` means the input to the network.
    /// Only supports skip connections for identical shapes.
    ///
    /// # Arguments
    ///
    /// * `infrom` - The index of the layer to connect from (input).
    /// * `into` - The index of the layer to connect to (input).
    pub fn connect(&mut self, infrom: usize, into: usize) {
        if infrom > self.layers.len() || into >= self.layers.len() || infrom > into {
            panic!("Invalid layer indices for skip connection.");
        } else if self.connect.contains_key(&infrom) {
            panic!("Skip connection already exists for layer {}", infrom);
        }

        let from = match &self.layers[infrom] {
            Layer::Dense(layer) => &layer.inputs,
            Layer::Convolution(layer) => &layer.inputs,
            Layer::Deconvolution(layer) => &layer.inputs,
            Layer::Maxpool(_) => panic!("Skip connection from Maxpool layer not supported."),
            Layer::Feedback(feedback) => &feedback.inputs,
        };
        let to = match &self.layers[into] {
            Layer::Dense(layer) => &layer.inputs,
            Layer::Convolution(layer) => &layer.inputs,
            Layer::Deconvolution(layer) => &layer.inputs,
            Layer::Maxpool(layer) => &layer.inputs,
            Layer::Feedback(feedback) => &feedback.inputs,
        };
        assert_eq_shape!(from, to);

        // Store the skip connection for use in the propagation.
        self.connect.insert(into, infrom);
    }

    /// Extract the total number of parameters in the network.
    fn parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameters()).sum()
    }

    /// Set the `feedback::Accumulation` function of the network.
    /// Note that this is only relevant for loopback- and skip connections.
    pub fn set_accumulation(&mut self, accumulation: feedback::Accumulation) {
        self.accumulation = accumulation;

        for layer in self.layers.iter_mut() {
            match layer {
                Layer::Feedback(ref mut layer) => layer.accumulation = self.accumulation.clone(),
                _ => (),
            }
        }
    }

    /// Modify the `activation::Activation` function of a layer.
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
            Layer::Deconvolution(ref mut layer) => {
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
        let mut vectors: Vec<Vec<Vec<tensor::Tensor>>> = Vec::new();
        for layer in self.layers.iter().rev() {
            match layer {
                Layer::Dense(layer) => {
                    let (output, input) = match &layer.weights.shape {
                        tensor::Shape::Double(output, input) => (*output, *input),
                        _ => panic!("Expected Dense shape"),
                    };
                    vectors.push(vec![vec![
                        tensor::Tensor::double(vec![vec![0.0; input]; output]),
                        if layer.bias.is_some() {
                            tensor::Tensor::single(vec![0.0; output])
                        } else {
                            tensor::Tensor::single(vec![])
                        },
                    ]]);
                }
                Layer::Convolution(layer) => {
                    let (ch, kh, kw) = match layer.kernels[0].shape {
                        tensor::Shape::Triple(ch, he, wi) => (ch, he, wi),
                        _ => panic!("Expected Convolution shape"),
                    };
                    vectors.push(vec![
                        vec![
                            tensor::Tensor::triple(vec![vec![vec![0.0; kw]; kh]; ch]),
                            // TODO: Add bias term here.
                        ];
                        layer.kernels.len()
                    ]);
                }
                Layer::Deconvolution(layer) => {
                    let (ch, kh, kw) = match layer.kernels[0].shape {
                        tensor::Shape::Triple(ch, he, wi) => (ch, he, wi),
                        _ => panic!("Expected Deconvolution shape"),
                    };
                    vectors.push(vec![
                        vec![
                            tensor::Tensor::triple(vec![vec![vec![0.0; kw]; kh]; ch]),
                            // TODO: Add bias term here.
                        ];
                        layer.kernels.len()
                    ]);
                }
                Layer::Maxpool(_) => vectors.push(vec![vec![tensor::Tensor::single(vec![0.0; 0])]]),
                Layer::Feedback(_) => {
                    vectors.push(vec![vec![tensor::Tensor::single(vec![0.0; 0])]])
                }
            }
        }

        // Validate the optimizers' parameters.
        // Override to default values if wrongly set.
        optimizer.validate(vectors);

        self.optimizer = optimizer;

        for layer in self.layers.iter_mut() {
            match layer {
                Layer::Feedback(block) => block.copy_optimizer(self.optimizer.clone()),
                _ => (),
            }
        }
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
    /// Stops early if the validation loss does not improve for five consecutive epochs.
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
    /// * `validation` - An optional tuple.
    ///     - 1: The validation inputs.
    ///     - 2: The corresponding validation targets.
    ///     - 3: The tolerance for early stopping.
    ///          If the validation loss does not improve for this many epochs, the training stops.
    /// * `batch` - The batch size to use when training.
    /// * `epochs` - The number of epochs to train the network for.
    /// * `print` - The frequency of printing validation metrics to the console.
    ///
    /// # Returns
    ///
    /// A vector of the train- and validation loss of the network per epoch.
    ///
    /// # Panics
    ///
    /// If the loss is NaN.
    pub fn learn(
        &mut self,
        inputs: &Vec<&tensor::Tensor>,
        targets: &Vec<&tensor::Tensor>,
        validation: Option<(&Vec<&tensor::Tensor>, &Vec<&tensor::Tensor>, i32)>,
        batch: usize,
        epochs: i32,
        print: Option<i32>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut threshold: Option<i32> = None;
        if let Some((_, _, limit)) = validation {
            threshold = Some(limit);
        }

        // Print the header of the table.
        if let Some(print) = print {
            if print > epochs as i32 {
                println!("Note: print frequency is higher than the number of epochs. No printouts will be made.");
            } else if let Some(_) = validation {
                println!("{:>5} \t {:<23} \t {:>10}", "EPOCH", "LOSS", "ACCURACY");
                println!(
                    "{:>5} \t {:>10} | {:<10} \t {:>10}",
                    "", "validation", "train", "validation"
                );
            } else {
                println!("{:>5} \t {:>10}", "EPOCH", "TRAIN LOSS");
            }
        }

        self.layers.iter_mut().for_each(|layer| match layer {
            Layer::Dense(layer) => layer.training = true,
            Layer::Convolution(layer) => layer.training = true,
            Layer::Deconvolution(layer) => layer.training = true,
            Layer::Feedback(feedback) => feedback.training(true),
            _ => (),
        });

        let mut train_loss = Vec::new();
        let mut val_loss = Vec::new();
        let mut val_acc = Vec::new();

        // Split the input data into batches.
        let batches: Vec<(&[&tensor::Tensor], &[&tensor::Tensor])> = inputs
            .par_chunks(batch)
            .zip(targets.par_chunks(batch))
            .collect();

        for epoch in 1..epochs + 1 {
            let mut loss_epoch = 0.0;
            for batch in batches.iter() {
                // Parallel iteration over the batch.
                // I.e., parallell forward and backward pass for each sample in the batch.
                let results: Vec<_> = batch
                    .into_par_iter()
                    .map(|(input, target)| {
                        let (preactivated, activated, maxpools, feedbacks) = self.forward(input);
                        let (loss, gradient) =
                            self.objective.loss(&activated.last().unwrap(), target);

                        let (wg, bg) = self.backward(
                            gradient,
                            &preactivated,
                            &activated,
                            &maxpools,
                            &feedbacks,
                        );

                        (wg, bg, loss)
                    })
                    .collect();

                let mut weight_gradients: Vec<tensor::Tensor> = Vec::new();
                let mut bias_gradients: Vec<Option<tensor::Tensor>> = Vec::new();
                let mut losses: Vec<f32> = Vec::new();

                // Collect the results from the parallel iteration, and sum the gradients and loss.
                for (wg, wb, loss) in results {
                    if loss.is_nan() {
                        panic!("Loss is NaN. Aborting.");
                    }
                    losses.push(loss);

                    if weight_gradients.is_empty() {
                        weight_gradients = wg;
                        bias_gradients = wb;
                    } else {
                        for (gradient, new) in weight_gradients.iter_mut().zip(wg.iter()) {
                            gradient.add_inplace(new)
                        }

                        for (gradient, new) in bias_gradients.iter_mut().zip(wb.iter()) {
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
                loss_epoch += losses.iter().sum::<f32>() / losses.len() as f32;

                // Perform the update step wrt. the summed gradients for the batch.
                self.update(epoch, weight_gradients, bias_gradients);
            }
            train_loss.push(loss_epoch / batches.len() as f32);

            if let Some((val_inputs, val_targets, _)) = validation {
                let (_val_loss, _val_acc) = self.validate(val_inputs, val_targets, 1e-6);
                val_loss.push(_val_loss);
                val_acc.push(_val_acc);
            }

            if let Some(print) = print {
                if epoch % print == 0 && !val_acc.is_empty() {
                    println!(
                        "{:>5} \t {:>10.5} | {:<10.5} \t {:>8.2} %",
                        epoch,
                        val_loss.last().unwrap(),
                        train_loss.last().unwrap(),
                        val_acc.last().unwrap() * 100.0
                    );
                } else if epoch % print == 0 {
                    println!("{:>5} \t {:>10.5}", epoch, train_loss.last().unwrap(),);
                }
            }

            // Check if the validation loss has not improved for the last `threshold` epochs.
            // If so, stop training.
            if let Some(threshold) = threshold {
                if epoch > threshold {
                    let history: Vec<&f32> =
                        val_loss.iter().rev().take(threshold as usize).collect();
                    let mut increasing = true;
                    for i in 0..threshold as usize - 1 {
                        if history[i] <= history[i + 1] {
                            increasing = false;
                            break;
                        }
                    }
                    if increasing {
                        println!("Validation loss has increased for the last {} epochs.\nStopping training (at epoch {}).", threshold, epoch);
                        break;
                    }
                }
            }
        }
        for layer in &mut self.layers {
            match layer {
                Layer::Dense(layer) => layer.training = false,
                Layer::Convolution(layer) => layer.training = false,
                Layer::Deconvolution(layer) => layer.training = false,
                Layer::Feedback(feedback) => feedback.training(false),
                _ => (),
            }
        }

        (train_loss, val_loss, val_acc)
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
    /// * A vector of preactivated tensors.
    /// * A vector of activated tensors.
    /// * A vector of maxpool tensors.
    /// * A nested vector of intermediate feedback block tensors.
    pub fn forward(
        &self,
        input: &tensor::Tensor,
    ) -> (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        Vec<Option<tensor::Tensor>>,
        Vec<Vec<tensor::Tensor>>,
    ) {
        let mut preactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];
        let mut maxpools: Vec<Option<tensor::Tensor>> = Vec::new();
        let mut feedbacks: Vec<Vec<tensor::Tensor>> = Vec::new();

        for i in 0..self.layers.len() {
            let mut x = activated.last().unwrap().clone();

            // Check if the layer should account for a skip connection.
            if self.connect.contains_key(&i) {
                let _x = activated[self.connect[&i]].clone();

                match self.accumulation {
                    feedback::Accumulation::Add => {
                        x.add_inplace(&_x);
                    }
                    feedback::Accumulation::Subtract => {
                        x.sub_inplace(&_x);
                    }
                    feedback::Accumulation::Multiply => {
                        x.mul_inplace(&_x);
                    }
                    feedback::Accumulation::Overwrite => {
                        x = _x;
                    }
                    feedback::Accumulation::Mean => {
                        x.mean_inplace(&_x);
                    }
                    #[allow(unreachable_patterns)]
                    _ => unimplemented!("Accumulation method not implemented."),
                }
            }

            // Perform the forward pass of the current layer.
            let (mut pre, mut post, mut max, fbs) = self._forward(&x, i, i + 1);

            // Store the outputs of the current layer.
            preactivated.append(&mut pre);
            activated.append(&mut post);
            maxpools.append(&mut max);
            feedbacks.extend(fbs);

            // Check if the layer output should be fed back to a previous layer.
            if self.loopbacks.contains_key(&i) {
                let mut current: tensor::Tensor = activated.last().unwrap().clone();

                // Reshaping the last activated tensor in cases of flattened output.
                current = current.reshape(match self.layers[self.loopbacks[&i]] {
                    Layer::Dense(ref layer) => layer.inputs.clone(),
                    Layer::Convolution(ref layer) => layer.inputs.clone(),
                    Layer::Deconvolution(ref layer) => layer.inputs.clone(),
                    Layer::Maxpool(ref layer) => layer.inputs.clone(),
                    _ => panic!("Feedback not implemented for this layer type."),
                });

                // Add the original input of the fed-back layer to the latent representation.
                match self.accumulation {
                    feedback::Accumulation::Add => {
                        current.add_inplace(&activated[self.loopbacks[&i]]);
                    }
                    feedback::Accumulation::Subtract => {
                        current.sub_inplace(&activated[self.loopbacks[&i]]);
                    }
                    feedback::Accumulation::Multiply => {
                        current.mul_inplace(&activated[self.loopbacks[&i]]);
                    }
                    feedback::Accumulation::Overwrite => {
                        ();
                    }
                    feedback::Accumulation::Mean => {
                        current.mean_inplace(&activated[self.loopbacks[&i]])
                    }
                }

                // Perform the forward pass of the feedback loop.
                // TODO: Handle feedback blocks inside loopbacks. Or; panic?
                let (fpre, fpost, fmax, _) = self._forward(&current, self.loopbacks[&i], i + 1);

                // Store the outputs of the loopback layers.
                for (idx, j) in (self.loopbacks[&i]..i + 1).enumerate() {
                    match self.accumulation {
                        feedback::Accumulation::Add => {
                            preactivated[j].add_inplace(&fpre[idx]);
                            activated[j + 1].add_inplace(&fpost[idx]);

                            // Extend the maxpool indices.
                            if let Some(Some(max)) = maxpools.get_mut(j) {
                                if let Some(fmax) = &fmax[idx] {
                                    max.extend(&fmax);
                                } else {
                                    panic!("Maxpool indices are missing.");
                                }
                            }
                        }
                        feedback::Accumulation::Subtract => {
                            preactivated[j].sub_inplace(&fpre[idx]);
                            activated[j + 1].sub_inplace(&fpost[idx]);

                            // Extend the maxpool indices.
                            if let Some(Some(max)) = maxpools.get_mut(j) {
                                if let Some(fmax) = &fmax[idx] {
                                    max.extend(&fmax);
                                } else {
                                    panic!("Maxpool indices are missing.");
                                }
                            }
                        }
                        feedback::Accumulation::Multiply => {
                            preactivated[j].mul_inplace(&fpre[idx]);
                            activated[j + 1].mul_inplace(&fpost[idx]);

                            // Extend the maxpool indices.
                            if let Some(Some(max)) = maxpools.get_mut(j) {
                                if let Some(fmax) = &fmax[idx] {
                                    max.extend(&fmax);
                                } else {
                                    panic!("Maxpool indices are missing.");
                                }
                            }
                        }
                        feedback::Accumulation::Overwrite => {
                            preactivated[j] = fpre[idx].to_owned();
                            activated[j + 1] = fpost[idx].to_owned();

                            // Overwrite the maxpool indices.
                            if let Some(Some(max)) = maxpools.get_mut(j) {
                                if let Some(fmax) = &fmax[idx] {
                                    *max = fmax.clone();
                                } else {
                                    panic!("Maxpool indices are missing.");
                                }
                            }
                        }
                        feedback::Accumulation::Mean => {
                            preactivated[j].mean_inplace(&fpre[idx]);
                            activated[j + 1].mean_inplace(&fpost[idx]);

                            // Extend the maxpool indices.
                            if let Some(Some(max)) = maxpools.get_mut(j) {
                                if let Some(fmax) = &fmax[idx] {
                                    max.extend(&fmax);
                                } else {
                                    panic!("Maxpool indices are missing.");
                                }
                            }
                        }
                        #[allow(unreachable_patterns)]
                        _ => unimplemented!("Accumulation method not implemented."),
                    }
                }
            }
        }

        (preactivated, activated, maxpools, feedbacks)
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
    /// * A vector of preactivated tensors.
    /// * A vector of activated tensors.
    /// * A vector of maxpool tensors.
    /// * A nested vector of intermediate feedback block tensors.
    fn _forward(
        &self,
        input: &tensor::Tensor,
        from: usize,
        to: usize,
    ) -> (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        Vec<Option<tensor::Tensor>>,
        Vec<Vec<tensor::Tensor>>,
    ) {
        let mut preactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];
        let mut maxpools: Vec<Option<tensor::Tensor>> = Vec::new();
        let mut feedbacks: Vec<Vec<tensor::Tensor>> = Vec::new();

        for layer in &self.layers[from..to] {
            let x = activated.last().unwrap();
            match layer {
                Layer::Dense(layer) => {
                    assert_eq_shape!(layer.inputs, x.shape);
                    let (pre, post) = layer.forward(x);
                    preactivated.push(pre);
                    activated.push(post);
                    maxpools.push(None);
                }
                Layer::Convolution(layer) => {
                    let (pre, post) = layer.forward(x);
                    preactivated.push(pre);
                    activated.push(post);
                    maxpools.push(None);
                }
                Layer::Deconvolution(layer) => {
                    let (pre, post) = layer.forward(x);
                    preactivated.push(pre);
                    activated.push(post);
                    maxpools.push(None);
                }
                Layer::Maxpool(layer) => {
                    assert_eq_shape!(layer.inputs, x.shape);
                    let (pre, post, max) = layer.forward(x);
                    preactivated.push(pre);
                    activated.push(post);
                    maxpools.push(Some(max));
                }
                Layer::Feedback(block) => {
                    assert_eq_shape!(block.inputs, x.shape);
                    let (pre, post, max, fbpre, fbpost) = block.forward(x);
                    preactivated.push(pre);
                    activated.push(post);
                    maxpools.push(Some(max));
                    feedbacks.push(vec![fbpre, fbpost]);
                }
            };
        }

        // Removing the input clone from the activated vector.
        // As this is present in the `forward` function.
        activated.remove(0);

        (preactivated, activated, maxpools, feedbacks)
    }

    /// Compute the backward pass of the network for the given output gradient.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient of the output.
    /// * `preactivated` - The pre-activation values of each layer.
    /// * `activated` - The post-activation values of each layer.
    /// * `maxpools` - The maxpool indices of each maxpool-layer.
    ///
    /// # Returns
    ///
    /// A tuple containing the weight and bias gradients of each layer.
    fn backward(
        &self,
        gradient: tensor::Tensor,
        preactivated: &Vec<tensor::Tensor>,
        activated: &Vec<tensor::Tensor>,
        maxpools: &Vec<Option<tensor::Tensor>>,
        feedbacks: &Vec<Vec<tensor::Tensor>>,
    ) -> (Vec<tensor::Tensor>, Vec<Option<tensor::Tensor>>) {
        let mut gradients: Vec<tensor::Tensor> = vec![gradient];
        let mut weight_gradient: Vec<tensor::Tensor> = Vec::new();
        let mut bias_gradient: Vec<Option<tensor::Tensor>> = Vec::new();

        let mut connect = HashMap::new();
        for (key, value) in self.connect.iter() {
            // {to: from} -> {from: to}
            connect.insert(value, key);
        }

        self.layers.iter().rev().enumerate().for_each(|(i, layer)| {
            let idx = self.layers.len() - i - 1;

            let input: &tensor::Tensor = &activated[idx];
            let output: &tensor::Tensor = &preactivated[idx];

            // Check for skip connections.
            // Add the gradient of the skip connection to the current gradient.
            if connect.contains_key(&idx) {
                let gradient = gradients[self.layers.len() - connect[&idx]].clone();
                gradients.last_mut().unwrap().add_inplace(&gradient);
                // TODO: Handle accumulation methods.
            }

            let (gradient, wg, bg) = match layer {
                Layer::Dense(layer) => layer.backward(&gradients.last().unwrap(), input, output),
                Layer::Convolution(layer) => {
                    layer.backward(&gradients.last().unwrap(), input, output)
                }
                Layer::Deconvolution(layer) => {
                    layer.backward(&gradients.last().unwrap(), input, output)
                }
                Layer::Maxpool(layer) => (
                    layer.backward(
                        &gradients.last().unwrap(),
                        if let Some(max) = &maxpools[idx] {
                            max
                        } else {
                            panic!("Maxpool indices are missing.")
                        },
                    ),
                    tensor::Tensor::single(vec![0.0; 0]),
                    None,
                ),
                Layer::Feedback(block) => {
                    block.backward(&gradients.last().unwrap(), feedbacks.last().unwrap())
                }
            };

            gradients.push(gradient);
            weight_gradient.push(wg);
            bias_gradient.push(bg);
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
    fn update(
        &mut self,
        stepnr: i32,
        mut weight_gradients: Vec<tensor::Tensor>,
        mut bias_gradients: Vec<Option<tensor::Tensor>>,
    ) {
        self.layers
            .iter_mut()
            .rev()
            .enumerate()
            .for_each(|(i, layer)| match layer {
                Layer::Dense(layer) => {
                    self.optimizer.update(
                        i,
                        0,
                        false,
                        stepnr,
                        &mut layer.weights,
                        &mut weight_gradients[i],
                    );

                    if let Some(bias) = &mut layer.bias {
                        self.optimizer.update(
                            i,
                            0,
                            true,
                            stepnr,
                            bias,
                            &mut bias_gradients[i].as_mut().unwrap(),
                        )
                    }
                }
                Layer::Convolution(layer) => {
                    for (f, (filter, gradient)) in layer
                        .kernels
                        .iter_mut()
                        .zip(weight_gradients[i].quadruple_to_vec_triple().iter_mut())
                        .enumerate()
                    {
                        self.optimizer.update(i, f, false, stepnr, filter, gradient);
                        // TODO: Add bias term here.
                    }
                }
                Layer::Deconvolution(layer) => {
                    for (f, (filter, gradient)) in layer
                        .kernels
                        .iter_mut()
                        .zip(weight_gradients[i].quadruple_to_vec_triple().iter_mut())
                        .enumerate()
                    {
                        self.optimizer.update(i, f, false, stepnr, filter, gradient);
                        // TODO: Add bias term here.
                    }
                }
                Layer::Maxpool(_) => {}
                Layer::Feedback(block) => block.update(
                    stepnr,
                    &mut weight_gradients[i],
                    &mut bias_gradients[i].as_mut().unwrap(),
                ),
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
    /// * `tol` - The tolerance for the accuracy.
    ///
    /// # Returns
    ///
    /// A tuple containing the total loss and accuracy of the network for the given `inputs` and `targets`.
    pub fn validate(
        &mut self,
        inputs: &[&tensor::Tensor],
        targets: &[&tensor::Tensor],
        tol: f32,
    ) -> (f32, f32) {
        let mut training: bool = false;
        for layer in &mut self.layers {
            match layer {
                Layer::Dense(layer) => {
                    if layer.training && !training {
                        training = true;
                    } else {
                        break;
                    }
                    layer.training = false
                }
                Layer::Convolution(layer) => layer.training = false,
                Layer::Deconvolution(layer) => layer.training = false,
                Layer::Feedback(feedback) => feedback.training(false),
                _ => (),
            }
        }

        let results: Vec<_> = inputs
            .par_iter()
            .zip(targets.par_iter())
            .map(|(input, target)| {
                let prediction = self.predict(input);
                let (loss, _) = self.objective.loss(&prediction, target);

                let acc = match self.layers.last().unwrap() {
                    Layer::Dense(layer) => match layer.activation {
                        activation::Function::Softmax(_) => {
                            if target.argmax() == prediction.argmax() {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => {
                            let target = target.get_flat();
                            let prediction = prediction.get_flat();

                            if target.len() == 1 {
                                if (prediction[0] - target[0]).abs() < tol {
                                    1.0
                                } else {
                                    0.0
                                }
                            } else {
                                target
                                    .iter()
                                    .zip(prediction.iter())
                                    .map(|(t, p)| if (t - p).abs() < tol { 1.0 } else { 0.0 })
                                    .sum::<f32>()
                                    / target.len() as f32
                            }
                        }
                    },
                    Layer::Convolution(_) => {
                        unimplemented!("Image output (target) not supported.")
                    }
                    Layer::Deconvolution(_) => {
                        unimplemented!("Image output (target) not supported.")
                    }
                    Layer::Maxpool(_) => {
                        unimplemented!("Image output (target) not supported.")
                    }
                    _ => unimplemented!("Feedback blocks not yet implemented."),
                };

                (loss, acc)
            })
            .collect();

        if training {
            for layer in &mut self.layers {
                match layer {
                    Layer::Dense(layer) => layer.training = true,
                    Layer::Convolution(layer) => layer.training = true,
                    Layer::Deconvolution(layer) => layer.training = true,
                    Layer::Feedback(feedback) => feedback.training(true),
                    _ => (),
                }
            }
        }

        let mut loss: Vec<f32> = Vec::new();
        let mut acc: Vec<f32> = Vec::new();
        for (_loss, _acc) in results {
            loss.push(_loss);
            acc.push(_acc);
        }

        (
            loss.iter().sum::<f32>() / loss.len() as f32,
            acc.iter().sum::<f32>() / acc.len() as f32,
        )
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
                Layer::Deconvolution(layer) => {
                    let (_, out) = layer.forward(&output);
                    output = out;
                }
                Layer::Maxpool(layer) => {
                    let (_, out, _) = layer.forward(&output);
                    output = out;
                }
                Layer::Feedback(block) => {
                    let (_, out, _, _, _) = block.forward(&output);
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
    use super::*;
    use crate::assert_eq_data;

    #[test]
    fn test_forward() {
        let mut network = Network::new(tensor::Shape::Triple(1, 3, 3));

        network.convolution(
            1,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            activation::Activation::Linear,
            None,
        );

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                conv.kernels[0] = tensor::Tensor::triple(vec![vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                ]]);
            }
            _ => (),
        }

        let input = tensor::Tensor::triple(vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]]);

        let (pre, post, _, _) = network.forward(&input);

        assert_eq_data!(pre.last().unwrap().data, input.data);
        assert_eq_data!(post.last().unwrap().data, input.data);
    }

    #[test]
    fn test_backward() {
        // See Python file `documentation/validation/test_network_backward.py` for the reference implementation.

        let mut network = Network::new(tensor::Shape::Triple(2, 4, 4));

        network.convolution(
            3,
            (2, 2),
            (1, 1),
            (0, 0),
            (1, 1),
            activation::Activation::ReLU,
            None,
        );
        network.dense(5, activation::Activation::ReLU, true, None);

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                conv.kernels[0] = tensor::Tensor::triple(vec![
                    vec![vec![1.0, 1.0], vec![2.0, 2.0]],
                    vec![vec![1.0, 2.0], vec![1.0, 2.0]],
                ]);
                conv.kernels[1] = tensor::Tensor::triple(vec![
                    vec![vec![2.0, 2.0], vec![1.0, 1.0]],
                    vec![vec![2.0, 1.0], vec![2.0, 1.0]],
                ]);
                conv.kernels[2] = tensor::Tensor::triple(vec![
                    vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                    vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                ]);
            }
            _ => (),
        }
        match network.layers[1] {
            Layer::Dense(ref mut dense) => {
                dense.weights = tensor::Tensor::double(vec![
                    vec![2.5; 27],
                    vec![-1.2; 27],
                    vec![0.5; 27],
                    vec![3.5; 27],
                    vec![5.2; 27],
                ]);
                dense.bias = Some(tensor::Tensor::single(vec![3.0, 4.0, 5.0, 6.0, 7.0]));
            }
            _ => (),
        }

        let input = tensor::Tensor::triple(vec![
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
            tensor::Tensor::triple(vec![
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
            tensor::Tensor::single(vec![
                10.0, 16.0, 7.0, 19.0, 31.0, 14.0, 7.0, 11.0, 5.0, 5.0, 14.0, 8.0, 11.0, 29.0,
                16.0, 8.0, 19.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
            tensor::Tensor::single(vec![603.0, -284.00003, 125.0, 846.0, 1255.0]),
            tensor::Tensor::single(vec![603.0, 0.0, 125.0, 846.0, 1255.0]),
        ];

        let (pre, post, max, _) = network.forward(&input);

        assert_eq_data!(pre[0].data, output[0].data);
        assert_eq_data!(post[1].data, output[1].data);

        assert_eq_data!(pre[1].data, output[2].data);
        assert_eq_data!(post[2].data, output[3].data);

        // let gradient = tensor::Tensor::from(vec![vec![vec![1.0; 3]; 3]; 3]);
        let gradient = tensor::Tensor::single(vec![1.0; 5]);

        let (weight_gradient, bias_gradient) =
            network.backward(gradient, &pre, &post, &max, &Vec::new());

        let _weight_gradient = vec![
            tensor::Tensor::quadruple(vec![
                vec![vec![vec![117., 117.]; 2]; 27],
                vec![vec![vec![117., 117.]; 2]; 27],
                vec![vec![vec![0.0, 0.0]; 2]; 27],
            ]),
            tensor::Tensor::double(vec![
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
            ]),
            tensor::Tensor::single(vec![1., 0., 1., 1., 1.]),
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
        // See Python file `documentation/validation/test_network_update.py` for the reference implementation.

        let mut network = Network::new(tensor::Shape::Triple(2, 4, 4));

        network.convolution(
            3,
            (2, 2),
            (1, 1),
            (0, 0),
            (1, 1),
            activation::Activation::ReLU,
            None,
        );
        network.dense(5, activation::Activation::ReLU, true, None);

        network.set_optimizer(optimizer::SGD::create(0.1, None));

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                conv.kernels[0] = tensor::Tensor::triple(vec![
                    vec![vec![1.0, 1.0], vec![2.0, 2.0]],
                    vec![vec![1.0, 2.0], vec![1.0, 2.0]],
                ]);
                conv.kernels[1] = tensor::Tensor::triple(vec![
                    vec![vec![2.0, 2.0], vec![1.0, 1.0]],
                    vec![vec![2.0, 1.0], vec![2.0, 1.0]],
                ]);
                conv.kernels[2] = tensor::Tensor::triple(vec![
                    vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                    vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                ]);
            }
            _ => (),
        }
        match network.layers[1] {
            Layer::Dense(ref mut dense) => {
                dense.weights = tensor::Tensor::double(vec![
                    vec![2.5; 27],
                    vec![-1.2; 27],
                    vec![0.5; 27],
                    vec![3.5; 27],
                    vec![5.2; 27],
                ]);
                dense.bias = Some(tensor::Tensor::single(vec![3.0, 4.0, 5.0, 6.0, 7.0]));
            }
            _ => (),
        }

        let input = tensor::Tensor::triple(vec![
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

        let (pre, post, max, _) = network.forward(&input);

        let gradient = tensor::Tensor::single(vec![1.0; 5]);

        let (weight_gradients, bias_gradients) =
            network.backward(gradient, &pre, &post, &max, &Vec::new());

        network.update(0, weight_gradients, bias_gradients);

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                assert_eq_data!(
                    conv.kernels[0].data,
                    tensor::Tensor::triple(vec![
                        vec![vec![-10.7, -10.7], vec![-9.7000, -9.7000]],
                        vec![vec![-10.7, -9.7000], vec![-10.7, -9.7000]],
                    ])
                    .data
                );
                assert_eq_data!(
                    conv.kernels[1].data,
                    tensor::Tensor::triple(vec![
                        vec![vec![-9.7000, -9.7000], vec![-10.7, -10.7]],
                        vec![vec![-9.7000, -10.7], vec![-9.7000, -10.7]],
                    ])
                    .data
                );
                assert_eq_data!(
                    conv.kernels[2].data,
                    tensor::Tensor::triple(vec![
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
                    dense.weights.data,
                    tensor::Data::Double(vec![
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
                    ])
                );
                if let Some(bias) = &dense.bias {
                    assert_eq_data!(
                        bias.data,
                        tensor::Data::Single(vec![2.9000, 4.0000, 4.9000, 5.9000, 6.9000])
                    );
                }
            }
            _ => (),
        }
    }
}
