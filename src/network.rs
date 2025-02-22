// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::{
    activation, assert_eq_shape, convolution, deconvolution, dense, feedback, maxpool, objective,
    optimizer, tensor,
};

use rayon::prelude::*;

use std::collections::HashMap;
use std::sync::Arc;

// Number of chunks for parallel processing of validation and batched prediction.
const _CHUNKS: usize = 64;

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

    /// Extracts the input shape of the layer.
    fn inputs(&self) -> &tensor::Shape {
        match &self {
            Layer::Dense(layer) => &layer.inputs,
            Layer::Convolution(layer) => &layer.inputs,
            Layer::Deconvolution(layer) => &layer.inputs,
            Layer::Feedback(layer) => &layer.inputs,
            Layer::Maxpool(layer) => &layer.inputs,
        }
    }

    /// Extracts the output shape of the layer.
    fn outputs(&self) -> &tensor::Shape {
        match &self {
            Layer::Dense(layer) => &layer.outputs,
            Layer::Convolution(layer) => &layer.outputs,
            Layer::Deconvolution(layer) => &layer.outputs,
            Layer::Feedback(layer) => &layer.outputs,
            Layer::Maxpool(layer) => &layer.outputs,
        }
    }
}

/// A feedforward neural network.
///
/// # Attributes
///
/// * `input` - The input `tensor::Shape` of the network.
/// * `layers` - The `Layer`s of the network.
/// * `loopbacks` - The looped connections of the network. from: (to, iterations)
/// * `loopaccumulation` - The accumulation type of the network for looped connections.
/// * `connect` - The skip connections of the network.
/// * `skipaccumulation` - The accumulation type of the network for skip connections.
/// * `optimizer` - The `optimizer::Optimizer` function of the network.
/// * `objective` - The `objective::Function` of the network.
pub struct Network {
    input: tensor::Shape,

    pub layers: Vec<Layer>,

    pub loopbacks: HashMap<usize, (usize, usize, bool)>,
    loopaccumulation: feedback::Accumulation,

    pub connect: HashMap<usize, usize>,
    skipaccumulation: feedback::Accumulation,

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
            write!(f, "\t\taccumulation: {}\n", self.skipaccumulation)?;

            let mut entries: Vec<(&usize, &usize)> = self.connect.iter().collect();
            entries.sort_by_key(|&(to, _)| to);
            for (to, from) in entries.iter() {
                write!(f, "\t\t{}.input -> {}.input\n", from, to)?;
            }
            write!(f, "\t)\n")?;
        }
        if !self.loopbacks.is_empty() {
            write!(f, "\tloops: (\n")?;
            write!(f, "\t\taccumulation: {}\n", self.loopaccumulation)?;

            let mut entries: Vec<(&usize, &(usize, usize, bool))> = self.loopbacks.iter().collect();
            entries.sort_by_key(|&(to, _)| to);
            for (from, (to, iterations, inskips)) in entries.iter() {
                write!(
                    f,
                    "\t\t{}.output -> {}.input (x {})\n",
                    from, to, iterations
                )?;
                if *inskips {
                    write!(f, "\t\twith input-to-input skips\n")?;
                }
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
            loopaccumulation: feedback::Accumulation::Mean,
            connect: HashMap::new(),
            skipaccumulation: feedback::Accumulation::Add,
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
    /// * `inskips` - Whether to use input-to-input skip connections inside the feedback block.
    /// * `outskips` - Whether to use output-to-input skip connections inside the feedback block.
    /// * `accumulation` - The accumulation method of the feedback block.
    ///  - `feedback::Accumulation::Mean` is assumed to be the best choice.
    ///
    /// # Notes
    ///
    /// * The feedback block must have at least one layer.
    /// * The input and output shapes of the feedback block must match.
    ///   - To allow for loops.
    pub fn feedback(
        &mut self,
        layers: Vec<feedback::Layer>,
        loops: usize,
        inskips: bool,
        outskips: bool,
        accumulation: feedback::Accumulation,
    ) {
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
            inskips,
            outskips,
            accumulation,
        )));
    }

    /// Add a loop connection between two layers.
    /// Only supports loop connections for identical shapes.
    ///
    /// # Arguments
    ///
    /// * `outof` - The index of the layer to connect from (output).
    /// * `into` - The index of the layer to connect to (input).
    /// * `iterations` - The number of iterations in the loop connection.
    /// * `scale` - The scaling function of the loop connection wrt. gradients.
    /// * `inskips` - Whether to use input-to-input skip connections inside the loop connection.
    pub fn loopback(
        &mut self,
        outof: usize,
        into: usize,
        iterations: usize,
        scale: tensor::Scale,
        inskips: bool,
    ) {
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
                    layer.loops += iterations as f32
                }
                Layer::Convolution(layer) => {
                    layer.scale = Arc::clone(&scale);
                    layer.loops += iterations as f32
                }
                Layer::Deconvolution(layer) => {
                    layer.scale = Arc::clone(&scale);
                    layer.loops += iterations as f32
                }
                Layer::Maxpool(layer) => layer.loops += iterations as f32,
                Layer::Feedback(_) => panic!("Loop connection includes feedback block."),
            }
        }

        // Store the loop connection for use in the forward pass.
        self.loopbacks.insert(outof, (into, iterations, inskips));
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
            Layer::Dense(layer) => match &layer.inputs {
                tensor::Shape::Single(nodes) => nodes,
                _ => panic!("Unknown shape!"),
            },
            Layer::Convolution(layer) => match &layer.inputs {
                tensor::Shape::Triple(c, h, w) => &(*c * *h * *w),
                _ => panic!("Unknown shape!"),
            },
            Layer::Deconvolution(layer) => match &layer.inputs {
                tensor::Shape::Triple(c, h, w) => &(*c * *h * *w),
                _ => panic!("Unknown shape!"),
            },
            Layer::Maxpool(_) => panic!("Unknown shape!"),
            Layer::Feedback(feedback) => match &feedback.inputs {
                tensor::Shape::Single(nodes) => &nodes,
                tensor::Shape::Triple(c, h, w) => &(*c * *h * *w),
                _ => panic!("Unknown shape!"),
            },
        };
        let to = match &self.layers[into] {
            Layer::Dense(layer) => match &layer.inputs {
                tensor::Shape::Single(nodes) => nodes,
                _ => panic!("Unknown shape!"),
            },
            Layer::Convolution(layer) => match &layer.inputs {
                tensor::Shape::Triple(c, h, w) => &(*c * *h * *w),
                _ => panic!("Unknown shape!"),
            },
            Layer::Deconvolution(layer) => match &layer.inputs {
                tensor::Shape::Triple(c, h, w) => &(*c * *h * *w),
                _ => panic!("Unknown shape!"),
            },
            Layer::Maxpool(layer) => match &layer.inputs {
                tensor::Shape::Single(nodes) => &nodes,
                tensor::Shape::Triple(c, h, w) => &(*c * *h * *w),
                _ => panic!("Unknown shape!"),
            },
            Layer::Feedback(feedback) => match &feedback.inputs {
                tensor::Shape::Single(nodes) => &nodes,
                tensor::Shape::Triple(c, h, w) => &(*c * *h * *w),
                _ => panic!("Unknown shape!"),
            },
        };
        assert_eq!(from, to);

        // Store the skip connection for use in the propagation.
        self.connect.insert(into, infrom);
    }

    /// Extract the total number of parameters in the network.
    fn parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameters()).sum()
    }

    /// Set the `feedback::Accumulation` function of the network.
    ///
    /// # Arguments
    /// * `skipaccumulation` - The `feedback::Accumulation` function for skip connections.
    /// * `loopaccumulation` - The `feedback::Accumulation` function for loop connections.
    pub fn set_accumulation(
        &mut self,
        skipaccumulation: feedback::Accumulation,
        loopaccumulation: feedback::Accumulation,
    ) {
        self.skipaccumulation = skipaccumulation;
        self.loopaccumulation = loopaccumulation;
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
                            feedbacks,
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
                // Extracting the tensor from the layer with the corresponding index.
                // Reshaping it in case the connection is across shape types.
                let mut _x = activated[self.connect[&i]].clone();
                if _x.shape != x.shape {
                    _x = _x.reshape(x.shape.clone());
                }

                match self.skipaccumulation {
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
                        x.mean_inplace(&vec![&_x]);
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
                let (into, iterations, inskips) = self.loopbacks[&i];

                // Perform the forward pass of the feedback loop.
                // TODO: Handle feedback blocks inside loopbacks. Or; panic?
                let mut fpres: Vec<Vec<tensor::Tensor>> = Vec::new();
                let mut fposts: Vec<Vec<tensor::Tensor>> =
                    vec![vec![activated.last().unwrap().clone()]];
                let mut fmaxs: Vec<Vec<Option<tensor::Tensor>>> = Vec::new();
                for _ in 0..iterations {
                    let mut current: tensor::Tensor =
                        fposts.last().unwrap().last().unwrap().clone();

                    // Reshaping the last activated tensor in cases of flattened output.
                    let inputs = self.layers[into].inputs();
                    if inputs != self.layers[i].outputs() {
                        current = current.reshape(inputs.clone());
                    }

                    if inskips {
                        current.add_inplace(&activated[into]);
                        // TODO: Handle other accumulation methods.
                    }

                    let (fpre, fpost, fmax, _) = self._forward(&current, into, i + 1);
                    fpres.push(fpre);
                    fposts.push(fpost);
                    fmaxs.push(fmax);
                }

                fposts.remove(0);

                // Store the outputs of the loopback layers.
                // TODO: Only combine these if the network is training.
                // TODO: ALWAYS combine the outputs, `i`, as these are used for subsequent layers.
                for (idx, j) in (into..i + 1).enumerate() {
                    match self.loopaccumulation {
                        feedback::Accumulation::Add => {
                            for iteration in 0..iterations {
                                preactivated[j].add_inplace(&fpres[iteration][idx]);
                                activated[j + 1].add_inplace(&fposts[iteration][idx]);

                                // Extend the maxpool indices.
                                if let Some(Some(max)) = maxpools.get_mut(j) {
                                    if let Some(fmax) = &fmaxs[iteration][idx] {
                                        max.extend(&fmax);
                                    } else {
                                        panic!("Maxpool indices are missing.");
                                    }
                                }
                            }
                        }
                        feedback::Accumulation::Subtract => {
                            for iteration in 0..iterations {
                                preactivated[j].sub_inplace(&fpres[iteration][idx]);
                                activated[j + 1].sub_inplace(&fposts[iteration][idx]);

                                // Extend the maxpool indices.
                                if let Some(Some(max)) = maxpools.get_mut(j) {
                                    if let Some(fmax) = &fmaxs[iteration][idx] {
                                        max.extend(&fmax);
                                    } else {
                                        panic!("Maxpool indices are missing.");
                                    }
                                }
                            }
                        }
                        feedback::Accumulation::Multiply => {
                            for iteration in 0..iterations {
                                preactivated[j].mul_inplace(&fpres[iteration][idx]);
                                activated[j + 1].mul_inplace(&fposts[iteration][idx]);

                                // Extend the maxpool indices.
                                if let Some(Some(max)) = maxpools.get_mut(j) {
                                    if let Some(fmax) = &fmaxs[iteration][idx] {
                                        max.extend(&fmax);
                                    } else {
                                        panic!("Maxpool indices are missing.");
                                    }
                                }
                            }
                        }
                        feedback::Accumulation::Overwrite => {
                            for iteration in 0..iterations {
                                preactivated[j] = fpres[iteration][idx].to_owned();
                                activated[j + 1] = fposts[iteration][idx].to_owned();

                                // Overwrite the maxpool indices.
                                if let Some(Some(max)) = maxpools.get_mut(j) {
                                    if let Some(fmax) = &fmaxs[iteration][idx] {
                                        *max = fmax.clone();
                                    } else {
                                        panic!("Maxpool indices are missing.");
                                    }
                                }
                            }
                        }
                        feedback::Accumulation::Mean => {
                            let fpre: Vec<&tensor::Tensor> =
                                fpres.iter().map(|x| &x[idx]).collect();
                            let fpost: Vec<&tensor::Tensor> =
                                fposts.iter().map(|x| &x[idx]).collect();
                            let fmax: Vec<&Option<tensor::Tensor>> =
                                fmaxs.iter().map(|x| &x[idx]).collect();

                            preactivated[j].mean_inplace(&fpre);
                            activated[j + 1].mean_inplace(&fpost);

                            // Extend the maxpool indices.
                            if let Some(Some(max)) = maxpools.get_mut(j) {
                                for fmax in fmax {
                                    if let Some(fmax) = fmax {
                                        max.extend(&fmax);
                                    } else {
                                        panic!("Maxpool indices are missing.");
                                    }
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
                    let (pre, post, max) = layer.forward(x);
                    preactivated.push(pre);
                    activated.push(post);
                    maxpools.push(Some(max));
                }
                Layer::Feedback(block) => {
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
    /// * `feedbacks` - The feedback block tensors of each feedback-layer.
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
        mut feedbacks: Vec<Vec<tensor::Tensor>>,
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

            let (mut gradient, wg, bg) = match layer {
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
                    block.backward(&gradients.last().unwrap(), &feedbacks.pop().unwrap())
                }
            };

            // Check for skip- or loopback connections.
            // Add the gradient of the skip connection to the current gradient.
            if connect.contains_key(&idx) {
                let gradient2 = gradients[self.layers.len() - *connect[&idx]].clone();
                gradient.add_inplace(&gradient2.reshape(gradient.shape.clone()));
                // TODO: Handle accumulation methods?
            }

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
            .par_chunks(_CHUNKS)
            .zip(targets.par_chunks(_CHUNKS))
            .flat_map(|(inputs, targets)| {
                inputs
                    .iter()
                    .zip(targets.iter())
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
                                            .map(
                                                |(t, p)| {
                                                    if (t - p).abs() < tol {
                                                        1.0
                                                    } else {
                                                        0.0
                                                    }
                                                },
                                            )
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
                    .collect::<Vec<_>>()
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

        let (loss, acc): (Vec<_>, Vec<_>) = results.into_iter().unzip();

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
        let (_, outputs, _, _) = self.forward(input);
        outputs.last().unwrap().clone()
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
        inputs
            .par_chunks(_CHUNKS)
            .flat_map(|batch| {
                batch
                    .iter()
                    .map(|input| self.predict(input))
                    .collect::<Vec<_>>()
            })
            .collect()
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
            network.backward(gradient, &pre, &post, &max, Vec::new());

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
                    vec![vec![0.0, 5.0], vec![0.0, 0.0]],
                    vec![vec![0.0, 0.0], vec![0.0, 10.0]],
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
            network.backward(gradient, &pre, &post, &max, Vec::new());

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
                        vec![vec![-4.68, -6.7], vec![-4.68, -11.7]],
                        vec![vec![-7.02, -11.7], vec![-7.02, -1.7]],
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
                            0.6000, 1.5000, -1.5000, -0.5000, 2.5000, 0.0000, 0.5000, 2.5000,
                            1.0000, 0.5000, 2.5000
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
                            -1.4000, -0.5000, -3.5000, -2.5000, 0.5000, -2.0000, -1.5000, 0.5000,
                            -1.0000, -1.5000, 0.5000
                        ],
                        vec![
                            2.5000, 1.9000, 2.8000, 1.6000, 0.4000, 2.1000, 2.8000, 2.4000, 3.0000,
                            3.0000, 2.1000, 2.7000, 2.4000, 0.6000, 1.9000, 2.7000, 1.6000, 2.5000,
                            -0.5000, 0.5000, 3.5000, 1.0000, 1.5000, 3.5000, 2.0000, 1.5000,
                            3.5000
                        ],
                        vec![
                            4.2000, 3.6000, 4.5000, 3.3000, 2.1000, 3.8000, 4.5000, 4.1000, 4.7000,
                            4.7000, 3.8000, 4.4000, 4.1000, 2.3000, 3.6000, 4.4000, 3.3000, 4.2000,
                            1.2000, 2.2000, 5.2000, 2.7000, 3.2000, 5.2000, 3.7000, 3.2000, 5.2000
                        ]
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

    #[test]
    fn test_learn_sgd() {
        // See Python file `documentation/validation/test_network_learn.py` for the reference implementation.

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
                    vec![vec![0.0, 5.0], vec![0.0, 3.0]],
                    vec![vec![2.0, 0.0], vec![0.0, 10.0]],
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

        let target = tensor::Tensor::single(vec![1.0, 0.0, 0.0, 0.0, 0.0]);

        network.set_objective(objective::Objective::MSE, None);
        network.set_optimizer(optimizer::SGD::create(0.001, None));

        network.learn(&vec![&input], &vec![&target], None, 1, 1, None);

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                let weights: Vec<Vec<Vec<Vec<f32>>>> = conv
                    .kernels
                    .iter()
                    .map(|kernel| kernel.get_triple(&kernel.shape))
                    .collect();

                let expected: Vec<Vec<Vec<Vec<f32>>>> = vec![
                    vec![
                        vec![vec![-79.8500, -79.8500], vec![-78.8500, -78.8500]],
                        vec![vec![-79.8500, -78.8500], vec![-79.8500, -78.8500]],
                    ],
                    vec![
                        vec![vec![-78.8500, -78.8500], vec![-79.8500, -79.8500]],
                        vec![vec![-78.8500, -79.8500], vec![-78.8500, -79.8500]],
                    ],
                    vec![
                        vec![vec![-80.8500, -75.8500], vec![-64.6800, -77.8500]],
                        vec![vec![-78.8500, -80.8500], vec![-56.5950, -70.8500]],
                    ],
                ];

                for (weight, expect) in weights.iter().zip(expected.iter()) {
                    for (w1, e1) in weight.iter().zip(expect.iter()) {
                        for (w2, e2) in w1.iter().zip(e1.iter()) {
                            for (w3, e3) in w2.iter().zip(e2.iter()) {
                                assert!((w3 - e3).abs() < 1e-5);
                            }
                        }
                    }
                }
            }
            _ => (),
        }
        match network.layers[1] {
            Layer::Dense(ref mut dense) => {
                assert_eq_data!(
                    dense.weights.data,
                    tensor::Data::Double(vec![
                        vec![
                            -1.9080, -4.5528, -0.5856, -5.8752, -11.1648, -3.6712, -0.5856,
                            -2.3488, 0.2960, 0.2960, -3.6712, -1.0264, -2.3488, -10.2832, -4.5528,
                            -1.0264, -5.8752, -1.9080, -16.4544, -13.3688, 2.5000, -12.4872,
                            -15.1320, -0.1448, -4.1120, -8.0792, 1.6184
                        ],
                        vec![
                            -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000,
                            -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000,
                            -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000,
                            -1.2000, -1.2000, -1.2000
                        ],
                        vec![
                            -0.4000, -0.9400, -0.1300, -1.2100, -2.2900, -0.7600, -0.1300, -0.4900,
                            0.0500, 0.0500, -0.7600, -0.2200, -0.4900, -2.1100, -0.9400, -0.2200,
                            -1.2100, -0.4000, -3.3700, -2.7400, 0.5000, -2.5600, -3.1000, -0.0400,
                            -0.8500, -1.6600, 0.3200
                        ],
                        vec![
                            -2.6840, -6.3944, -0.8288, -8.2496, -15.6704, -5.1576, -0.8288,
                            -3.3024, 0.4080, 0.4080, -5.1576, -1.4472, -3.3024, -14.4336, -6.3944,
                            -1.4472, -8.2496, -2.6840, -23.0912, -18.7624, 3.5000, -17.5256,
                            -21.2360, -0.2104, -5.7760, -11.3416, 2.2632
                        ],
                        vec![
                            -3.9800, -9.4880, -1.2260, -12.2420, -23.2580, -7.6520, -1.2260,
                            -4.8980, 0.6100, 0.6100, -7.6520, -2.1440, -4.8980, -21.4220, -9.4880,
                            -2.1440, -12.2420, -3.9800, -34.2740, -27.8480, 5.2000, -26.0120,
                            -31.5200, -0.3080, -8.5700, -16.8320, 3.3640
                        ]
                    ])
                );
                if let Some(bias) = &dense.bias {
                    assert_eq_data!(
                        bias.data,
                        tensor::Data::Single(vec![2.5592, 4.0000, 4.9100, 5.3816, 6.0820])
                    );
                }
            }
            _ => (),
        }
    }

    #[test]
    fn test_learn_adam() {
        // See Python file `documentation/validation/test_network_learn.py` for the reference implementation.

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
                    vec![vec![0.0, 5.0], vec![0.0, 3.0]],
                    vec![vec![2.0, 0.0], vec![0.0, 10.0]],
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

        let target = tensor::Tensor::single(vec![1.0, 0.0, 0.0, 0.0, 0.0]);

        network.set_objective(objective::Objective::MSE, None);
        network.set_optimizer(optimizer::Adam::create(0.001, 0.9, 0.999, 1e-8, None));

        network.learn(&vec![&input], &vec![&target], None, 1, 2, None);

        match network.layers[0] {
            Layer::Convolution(ref mut conv) => {
                let weights: Vec<Vec<Vec<Vec<f32>>>> = conv
                    .kernels
                    .iter()
                    .map(|kernel| kernel.get_triple(&kernel.shape))
                    .collect();

                let expected: Vec<Vec<Vec<Vec<f32>>>> = vec![
                    vec![
                        vec![vec![9.9800e-01, 9.9800e-01], vec![1.9980e+00, 1.9980e+00]],
                        vec![vec![9.9800e-01, 1.9980e+00], vec![9.9800e-01, 1.9980e+00]],
                    ],
                    vec![
                        vec![vec![1.9980e+00, 1.9980e+00], vec![9.9800e-01, 9.9800e-01]],
                        vec![vec![1.9980e+00, 9.9800e-01], vec![1.9980e+00, 9.9800e-01]],
                    ],
                    vec![
                        vec![vec![-2.0000e-03, 4.9980e+00], vec![-2.0000e-03, 2.9980e+00]],
                        vec![vec![1.9980e+00, -2.0000e-03], vec![-2.0000e-03, 9.9980e+00]],
                    ],
                ];

                for (weight, expect) in weights.iter().zip(expected.iter()) {
                    for (w1, e1) in weight.iter().zip(expect.iter()) {
                        for (w2, e2) in w1.iter().zip(e1.iter()) {
                            for (w3, e3) in w2.iter().zip(e2.iter()) {
                                assert!((w3 - e3).abs() < 1e-6);
                            }
                        }
                    }
                }
            }
            _ => (),
        }
        match network.layers[1] {
            Layer::Dense(ref mut dense) => {
                let weights: Vec<Vec<f32>> = match dense.weights.data {
                    tensor::Data::Double(ref weights) => weights.clone(),
                    _ => panic!("Invalid weight type"),
                };

                let expected: Vec<Vec<f32>> = vec![
                    vec![
                        2.4980, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980,
                        2.4980, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980,
                        2.4980, 2.4980, 2.5000, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980, 2.4980,
                    ],
                    vec![
                        -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000,
                        -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000,
                        -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000, -1.2000,
                        -1.2000, -1.2000, -1.2000,
                    ],
                    vec![
                        0.4980, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980,
                        0.4980, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980,
                        0.4980, 0.4980, 0.5000, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980, 0.4980,
                    ],
                    vec![
                        3.4980, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980,
                        3.4980, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980,
                        3.4980, 3.4980, 3.5000, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980, 3.4980,
                    ],
                    vec![
                        5.1980, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980,
                        5.1980, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980,
                        5.1980, 5.1980, 5.2000, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980, 5.1980,
                    ],
                ];

                for (weight, expect) in weights.iter().zip(expected.iter()) {
                    for (w, e) in weight.iter().zip(expect.iter()) {
                        assert!((w - e).abs() < 1e-6);
                    }
                }

                if let Some(bias) = &dense.bias {
                    let weight = bias.get_flat();
                    let expect = vec![2.9980, 4.0000, 4.9980, 5.9980, 6.9980];

                    for (w, e) in weight.iter().zip(expect.iter()) {
                        assert!((w - e).abs() < 1e-6);
                    }
                }
            }
            _ => (),
        }
    }

    #[test]
    fn test_learn_bigger() {
        // See Python file `documentation/validation/test_network_learn_bigger.py` for the reference implementation.

        let input = tensor::Tensor::triple(vec![vec![vec![0.1; 32]; 32]; 3]);
        let target = tensor::Tensor::one_hot(4, 10);

        let mut network = Network::new(tensor::Shape::Triple(3, 32, 32));
        network.convolution(
            32,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            activation::Activation::ReLU,
            None,
        );
        network.maxpool((2, 2), (2, 2));
        network.convolution(
            32,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            activation::Activation::ReLU,
            None,
        );
        network.maxpool((2, 2), (2, 2));
        network.convolution(
            32,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            activation::Activation::ReLU,
            None,
        );
        network.maxpool((2, 2), (2, 2));
        network.dense(256, activation::Activation::ReLU, true, None);
        network.dense(10, activation::Activation::Softmax, true, None);

        for layer in network.layers.iter_mut() {
            match layer {
                Layer::Convolution(ref mut conv) => {
                    conv.kernels = vec![
                        tensor::Tensor::ones(conv.kernels[0].shape.clone());
                        conv.kernels.len()
                    ];
                }
                Layer::Dense(ref mut dense) => {
                    dense.weights = tensor::Tensor::ones(dense.weights.shape.clone());
                    if let Some(ref mut bias) = dense.bias {
                        *bias = tensor::Tensor::ones(bias.shape.clone());
                    }
                }
                _ => (),
            }
        }

        network.set_objective(objective::Objective::CrossEntropy, None);
        network.set_optimizer(optimizer::Adam::create(0.0001, 0.9, 0.999, 1e-8, None));

        network.learn(&vec![&input], &vec![&target], None, 1, 5, None);

        for layer in network.layers.iter() {
            match layer {
                Layer::Convolution(conv) => {
                    let weights: Vec<Vec<Vec<Vec<f32>>>> = conv
                        .kernels
                        .iter()
                        .map(|kernel| kernel.get_triple(&kernel.shape))
                        .collect();

                    let expected: Vec<Vec<Vec<Vec<f32>>>> =
                        vec![vec![vec![vec![1.0; 3]; 3]; weights[0].len()]; weights.len()];

                    for (weight, expect) in weights.iter().zip(expected.iter()) {
                        for (w1, e1) in weight.iter().zip(expect.iter()) {
                            for (w2, e2) in w1.iter().zip(e1.iter()) {
                                for (w3, e3) in w2.iter().zip(e2.iter()) {
                                    assert!((w3 - e3).abs() < 1e-3);
                                }
                            }
                        }
                    }
                }
                _ => (),
            }
        }

        match &network.layers[network.layers.len() - 2] {
            Layer::Dense(dense) => {
                let weights: Vec<Vec<f32>> = match dense.weights.data {
                    tensor::Data::Double(ref weights) => weights.clone(),
                    _ => panic!("Invalid weight type"),
                };

                let expected: Vec<Vec<f32>> = vec![vec![1.0; 512]; 256];

                for (weight, expect) in weights.iter().zip(expected.iter()) {
                    for (w, e) in weight.iter().zip(expect.iter()) {
                        assert!((w - e).abs() < 1e-3);
                    }
                }

                if let Some(bias) = &dense.bias {
                    let weight = bias.get_flat();
                    let expect = vec![1.0; 256];

                    for (w, e) in weight.iter().zip(expect.iter()) {
                        assert!((w - e).abs() < 1e-3);
                    }
                }
            }
            _ => (),
        }

        match &network.layers[network.layers.len() - 1] {
            Layer::Dense(dense) => {
                let weights: Vec<Vec<f32>> = match dense.weights.data {
                    tensor::Data::Double(ref weights) => weights.clone(),
                    _ => panic!("Invalid weight type"),
                };

                let expected: Vec<Vec<f32>> = vec![
                    vec![0.9997029; 256],
                    vec![0.9997029; 256],
                    vec![0.9997029; 256],
                    vec![0.9997029; 256],
                    vec![1.0002971; 256],
                    vec![0.9997029; 256],
                    vec![0.9997029; 256],
                    vec![0.9997029; 256],
                    vec![0.9997029; 256],
                    vec![0.9997029; 256],
                ];

                for (weight, expect) in weights.iter().zip(expected.iter()) {
                    for (w, e) in weight.iter().zip(expect.iter()) {
                        assert!((w - e).abs() < 1e-6);
                    }
                }

                if let Some(bias) = &dense.bias {
                    let weight = bias.get_flat();
                    let expect = vec![
                        0.9997029, 0.9997029, 0.9997029, 0.9997029, 1.0002971, 0.9997029,
                        0.9997029, 0.9997029, 0.9997029, 0.9997029,
                    ];

                    for (w, e) in weight.iter().zip(expect.iter()) {
                        assert!((w - e).abs() < 1e-6);
                    }
                }
            }
            _ => (),
        }
    }
}
