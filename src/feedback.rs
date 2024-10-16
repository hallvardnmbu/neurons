// Copyright (C) 2024 Hallvard Høyland Lavik

use std::collections::HashMap;

use crate::{activation, assert_eq_shape, network, optimizer, tensor};

#[derive(Clone)]
pub enum Accumulation {
    Add,
    Subtract,
    Multiply,
    Overwrite,
    Mean,
    // TODO: Expand?
}

impl std::fmt::Display for Accumulation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Accumulation::Add => write!(f, "additive"),
            Accumulation::Subtract => write!(f, "subtractive"),
            Accumulation::Multiply => write!(f, "multiplicative"),
            Accumulation::Overwrite => write!(f, "overwrite"),
            Accumulation::Mean => write!(f, "mean"),
            #[allow(unreachable_patterns)]
            _ => unimplemented!("Accumulation method not implemented."),
        }
    }
}

/// A simplified layer definition used for defining feedback blocks.
///
/// # Dense
///
/// * `nodes` - The number of nodes in the layer.
/// * `activation` - The activation function of the layer.
/// * `bias` - Whether the layer should include a bias.
/// * `dropout` - The dropout rate of the layer.
///
/// # Convolution
///
/// * `filters` - The number of filters in the layer.
/// * `activation` - The activation function of the layer.
/// * `kernel` - The kernel size of the layer.
/// * `stride` - The stride of the layer.
/// * `padding` - The padding of the layer.
/// * `dilation` - The dilation of the layer.
/// * `dropout` - The dropout rate of the layer.
///
/// # Maxpool
///
/// * `kernel` - The pool size of the layer.
/// * `stride` - The stride of the layer.
pub enum Layer {
    Dense(usize, activation::Activation, bool, Option<f32>),
    Convolution(
        usize,
        activation::Activation,
        (usize, usize),
        (usize, usize),
        (usize, usize),
        (usize, usize),
        Option<f32>,
    ),
    Deconvolution(
        usize,
        activation::Activation,
        (usize, usize),
        (usize, usize),
        (usize, usize),
        Option<f32>,
    ),
    Maxpool((usize, usize), (usize, usize)),
}

/// A feedback block.
///
/// # Attributes
///
/// * `inputs` - The number of inputs to the block.
/// * `outputs` - The number of outputs from the block.
/// * `optimizer` - The optimizer used for training the block.
/// * `flatten` - Whether the block should flatten the output.
/// * `layers` - The layers of the block.
/// * `connect` - The (skip) connections between layers.
/// * `coupled` - The coupled layers of the block.
///
/// # Notes
///
/// * The `inputs` should match the `outputs`, to allow for feedback looping.
/// * TODO: Add support for differing input and output shapes, projecting differences internally.
#[derive(Clone)]
pub struct Feedback {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,
    pub(crate) optimizer: optimizer::Optimizer,
    pub(crate) flatten: bool,
    layers: Vec<network::Layer>,
    connect: HashMap<usize, usize>,
    pub(crate) accumulation: Accumulation,
    coupled: Vec<Vec<usize>>,
}

impl std::fmt::Display for Feedback {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Feedback (\n")?;
        write!(f, "\t\t\t{} -> {}\n", self.inputs, self.outputs)?;

        // let optimizer: String = self
        //     .optimizer
        //     .to_string()
        //     .lines()
        //     .map(|line| format!("\t\t{}", line))
        //     .collect::<Vec<String>>()
        //     .join("\n");
        // write!(f, "\t\t\toptimizer: (\n{}\n", optimizer)?;

        write!(f, "\t\t\tlayers: (\n")?;
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                network::Layer::Dense(layer) => {
                    write!(
                        f,
                        "\t\t\t\t{}: Dense{} ({} -> {})\n",
                        i, layer.activation, layer.inputs, layer.outputs
                    )?;
                }
                network::Layer::Convolution(layer) => {
                    write!(
                        f,
                        "\t\t\t\t{}: Convolution{} ({} -> {})\n",
                        i, layer.activation, layer.inputs, layer.outputs
                    )?;
                }
                network::Layer::Deconvolution(layer) => {
                    write!(
                        f,
                        "\t\t\t\t{}: Decovolution{} ({} -> {})\n",
                        i, layer.activation, layer.inputs, layer.outputs
                    )?;
                }
                network::Layer::Maxpool(layer) => {
                    write!(
                        f,
                        "\t\t\t\t{}: Maxpool ({} -> {})\n",
                        i, layer.inputs, layer.outputs
                    )?;
                }
                network::Layer::Feedback(_) => panic!("Nested feedback blocks are not supported."),
            }
        }
        write!(f, "\t\t\t)\n")?;
        if !self.coupled.is_empty() {
            write!(f, "\t\t\tcoupled: (\n")?;
            for coupled in self.coupled.iter() {
                write!(f, "\t\t\t\t{:?}\n", coupled)?;
            }
            write!(f, "\t\t\t\taccumulation: {}\n", self.accumulation)?;
            write!(f, "\t\t\t)\n")?;
        }
        if !self.connect.is_empty() {
            write!(f, "\t\t\tconnections: (\n")?;
            write!(f, "\t\t\t\taccumulation: {}\n", self.accumulation)?;

            let mut entries: Vec<(&usize, &usize)> = self.connect.iter().collect();
            entries.sort_by_key(|&(to, _)| to);
            for (to, from) in entries.iter() {
                write!(f, "\t\t\t\t{}.input -> {}.input\n", from, to)?;
            }
            write!(f, "\t\t\t)\n")?;
        }
        write!(f, "\t\t\tflatten: {}\n", self.flatten)?;
        write!(f, "\t\t)")?;
        Ok(())
    }
}

impl Feedback {
    /// Create a new feedback block.
    ///
    /// # Arguments
    ///
    /// * `layers` - The layers of the block.
    /// * `loops` - The number of loops the block should perform.
    /// * `skips` - Whether the block should include skip connections.
    /// * `accumulation` - The accumulation method of the block.
    pub fn create(
        mut layers: Vec<network::Layer>,
        loops: usize,
        skips: bool,
        accumulation: Accumulation,
    ) -> Self {
        assert!(loops > 0, "Feedback block should loop at least once.");
        let inputs = match layers.first().unwrap() {
            network::Layer::Dense(dense) => dense.inputs.clone(),
            network::Layer::Convolution(convolution) => convolution.inputs.clone(),
            network::Layer::Deconvolution(deconvolution) => deconvolution.inputs.clone(),
            network::Layer::Maxpool(maxpool) => maxpool.inputs.clone(),
            network::Layer::Feedback(_) => panic!("Nested feedback blocks are not supported."),
        };
        let outputs = match layers.last().unwrap() {
            network::Layer::Dense(dense) => dense.outputs.clone(),
            network::Layer::Convolution(convolution) => convolution.outputs.clone(),
            network::Layer::Deconvolution(deconvolution) => deconvolution.outputs.clone(),
            network::Layer::Maxpool(maxpool) => maxpool.outputs.clone(),
            network::Layer::Feedback(_) => panic!("Nested feedback blocks are not supported."),
        };
        assert_eq_shape!(inputs, outputs);

        let length = layers.len();

        // Extend the layers `loops` times.
        let _layers = layers.clone();
        for _ in 1..loops {
            layers.extend(_layers.clone());
        }

        // Define the coupled layers.
        let mut coupled: Vec<Vec<usize>> = Vec::new();
        for layer in 0..length {
            let mut coupling = Vec::new();
            for i in 0..loops {
                coupling.push(layer + i * length);
            }
            coupled.push(coupling);
        }

        // Define the skip connections.
        let mut connect: HashMap<usize, usize> = HashMap::new();
        if skips {
            for i in 1..loops {
                // {to: from}
                connect.insert(i * length, 0);
            }
        }

        Feedback {
            inputs,
            outputs,
            optimizer: optimizer::Adam::create(0.001, 0.9, 0.999, 1e-8, None),
            flatten: false,
            layers,
            connect,
            accumulation,
            coupled,
        }
    }

    /// Set the `optimizer::Optimizer` function of the network.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The reference to the network optimizer, to copy the values from.
    pub fn copy_optimizer(&mut self, mut optimizer: optimizer::Optimizer) {
        let mut vectors: Vec<Vec<Vec<tensor::Tensor>>> = Vec::new();
        for layer in self.layers.iter().rev() {
            match layer {
                network::Layer::Dense(layer) => {
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
                network::Layer::Convolution(layer) => {
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
                network::Layer::Deconvolution(layer) => {
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
                network::Layer::Maxpool(_) => {
                    vectors.push(vec![vec![tensor::Tensor::single(vec![0.0; 0])]])
                }
                _ => unimplemented!("Feedback blocks not yet implemented."),
            }
        }

        // Validate the optimizers' parameters.
        // Override to default values if wrongly set.
        optimizer.validate(vectors);

        self.optimizer = optimizer;
    }

    /// Count the number of parameters.
    /// Only counts the parameters of the first loop, as the rest are identical (coupled).
    pub fn parameters(&self) -> usize {
        let mut parameters = 0;
        for idx in 0..self.coupled.len() {
            parameters += match &self.layers[idx] {
                network::Layer::Dense(dense) => dense.parameters(),
                network::Layer::Convolution(convolution) => convolution.parameters(),
                network::Layer::Deconvolution(deconvolution) => deconvolution.parameters(),
                network::Layer::Maxpool(_) => 0,
                network::Layer::Feedback(_) => panic!("Nested feedback blocks are not supported."),
            };
        }
        parameters
    }

    pub fn training(&mut self, train: bool) {
        self.layers.iter_mut().for_each(|layer| match layer {
            network::Layer::Dense(layer) => layer.training = train,
            network::Layer::Convolution(layer) => layer.training = train,
            network::Layer::Deconvolution(layer) => layer.training = train,
            network::Layer::Maxpool(_) => {}
            network::Layer::Feedback(_) => panic!("Nested feedback blocks are not supported."),
        });
    }

    /// Compute the forward pass of the feedback block for the given input, including all
    /// intermediate pre- and post-activation values.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data (x).
    ///
    /// # Returns
    ///
    /// * Unactivated tensor to be used for neighbouring layers when backpropagating.
    /// * Activated tensor to be used for neighbouring layers when backpropagating.
    /// * Maxpool tensor to be used for neighbouring layers when backpropagating.
    /// * Intermediate unactivated tensors (nested).
    /// * Intermediate activated tensors (nested).
    pub fn forward(
        &self,
        input: &tensor::Tensor,
    ) -> (
        tensor::Tensor,
        tensor::Tensor,
        tensor::Tensor,
        tensor::Tensor,
        tensor::Tensor,
    ) {
        let mut unactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];
        let mut maxpools: Vec<Option<tensor::Tensor>> = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let mut x = activated.last().unwrap().clone();

            // Check if the layer should account for a skip connection.
            if self.connect.contains_key(&i) {
                let _x = activated[self.connect[&i]].clone();

                match self.accumulation {
                    Accumulation::Add => {
                        x.add_inplace(&_x);
                    }
                    Accumulation::Subtract => {
                        x.sub_inplace(&_x);
                    }
                    Accumulation::Multiply => {
                        x.mul_inplace(&_x);
                    }
                    Accumulation::Overwrite => {
                        x = _x;
                    }
                    Accumulation::Mean => {
                        x.mean_inplace(&_x);
                    }
                    #[allow(unreachable_patterns)]
                    _ => unimplemented!("Accumulation method not implemented."),
                }
            }

            match layer {
                network::Layer::Dense(layer) => {
                    assert_eq_shape!(layer.inputs, x.shape);
                    let (pre, post) = layer.forward(&x);
                    unactivated.push(pre);
                    activated.push(post);
                    maxpools.push(None);
                }
                network::Layer::Convolution(layer) => {
                    assert_eq_shape!(layer.inputs, x.shape);
                    let (pre, post) = layer.forward(&x);
                    unactivated.push(pre);
                    activated.push(post);
                    maxpools.push(None);
                }
                network::Layer::Deconvolution(layer) => {
                    assert_eq_shape!(layer.inputs, x.shape);
                    let (pre, post) = layer.forward(&x);
                    unactivated.push(pre);
                    activated.push(post);
                    maxpools.push(None);
                }
                network::Layer::Maxpool(layer) => {
                    assert_eq_shape!(layer.inputs, x.shape);
                    let (pre, post, max) = layer.forward(&x);
                    unactivated.push(pre);
                    activated.push(post);
                    maxpools.push(Some(max));
                }
                network::Layer::Feedback(_) => panic!("Nested feedback blocks are not supported."),
            };
        }

        // Flattening the last output if specified.
        if self.flatten {
            let last = activated.last().unwrap().clone();
            activated.pop();
            activated.push(last.flatten());
        }

        (
            unactivated[0].clone(),
            activated[activated.len() - 1].clone(),
            tensor::Tensor::nestedoptional(maxpools),
            tensor::Tensor::nested(unactivated),
            tensor::Tensor::nested(activated),
        )
    }

    /// Applies the backward pass of the layer to the gradient vector.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient tensor::Tensor to the layer.
    /// * `inbetween` - The intermediate tensors of the forward pass.
    ///
    /// # Returns
    ///
    /// The input-, weight- and bias gradient of the layer.
    pub fn backward(
        &self,
        gradient: &tensor::Tensor,
        inbetween: &Vec<tensor::Tensor>,
    ) -> (tensor::Tensor, tensor::Tensor, Option<tensor::Tensor>) {
        // We need to un-nest the input and output tensors (see `forward`).
        let unactivated = inbetween[0].unnested();
        let activated = inbetween[1].unnested();

        let mut gradients: Vec<tensor::Tensor> = vec![gradient.clone()];
        let mut weight_gradients: Vec<tensor::Tensor> = Vec::new();
        let mut bias_gradients: Vec<Option<tensor::Tensor>> = Vec::new();

        let mut connect: HashMap<usize, Vec<usize>> = HashMap::new();
        for (key, value) in self.connect.iter() {
            // {to: from} -> {from: [to1, ...]}
            if connect.contains_key(value) {
                connect.get_mut(value).unwrap().push(*key);
            } else {
                connect.insert(*value, vec![*key]);
            }
        }

        self.layers.iter().rev().enumerate().for_each(|(i, layer)| {
            let idx = self.layers.len() - i - 1;

            let input: &tensor::Tensor = &activated[idx];
            let output: &tensor::Tensor = &unactivated[idx];

            // Check for skip connections.
            // Add the gradient of the skip connection to the current gradient.
            if connect.contains_key(&idx) {
                for j in connect[&idx].iter() {
                    let gradient = gradients[self.layers.len() - j - 1].clone();
                    gradients.last_mut().unwrap().add_inplace(&gradient);
                }
                // TODO: Handle accumulation methods.
            }

            let (gradient, wg, bg) = match layer {
                network::Layer::Dense(layer) => {
                    layer.backward(&gradients.last().unwrap(), input, output)
                }
                network::Layer::Convolution(layer) => {
                    layer.backward(&gradients.last().unwrap(), input, output)
                }
                network::Layer::Deconvolution(layer) => {
                    layer.backward(&gradients.last().unwrap(), input, output)
                }
                _ => panic!("Unsupported layer type."),
            };

            gradients.push(gradient);
            weight_gradients.push(wg);
            bias_gradients.push(bg);
        });

        return (
            gradients.last().unwrap().clone(),
            tensor::Tensor::nested(weight_gradients),
            Some(tensor::Tensor::nestedoptional(bias_gradients)),
        );
    }

    pub fn update(
        &mut self,
        stepnr: i32,
        weight_gradients: &mut tensor::Tensor,
        bias_gradients: &mut tensor::Tensor,
    ) {
        let mut weight_gradients = weight_gradients.unnested();
        let mut bias_gradients = bias_gradients.unnestedoptional();

        // Update the weights and biases of the layers.
        self.layers
            .iter_mut()
            .rev()
            .enumerate()
            .for_each(|(i, layer)| match layer {
                network::Layer::Dense(layer) => {
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
                network::Layer::Convolution(layer) => {
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
                network::Layer::Deconvolution(layer) => {
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
                network::Layer::Maxpool(_) => {}
                network::Layer::Feedback(_) => panic!("Feedback layers are not supported."),
            });

        // Couple respective layers.
        // Iterates through `self.coupled` and updates the weights and biases to match.
        for couple in self.coupled.iter() {
            let mut count: f32 = 0.0;
            let mut weights: Vec<tensor::Tensor> = Vec::new();
            let mut biases: Vec<tensor::Tensor> = Vec::new();

            // Add the weights and biases of the coupled layers.
            for idx in couple.iter() {
                match &self.layers[*idx] {
                    network::Layer::Dense(layer) => {
                        weights.push(layer.weights.clone());
                        if let Some(bias) = &layer.bias {
                            biases.push(bias.clone());
                        }
                    }
                    network::Layer::Convolution(layer) => {
                        weights.push(tensor::Tensor::nested(layer.kernels.clone()));
                    }
                    network::Layer::Deconvolution(layer) => {
                        weights.push(tensor::Tensor::nested(layer.kernels.clone()));
                    }
                    _ => continue,
                }
                count += 1.0;
            }

            let mut weight: tensor::Tensor = weights.remove(0);
            let mut bias: Option<tensor::Tensor> = if biases.is_empty() {
                None
            } else {
                Some(biases.remove(0))
            };
            match self.accumulation {
                Accumulation::Add => {
                    for w in weights.iter() {
                        weight.add_inplace(w);
                    }
                    if let Some(bias) = &mut bias {
                        for b in biases.iter() {
                            bias.add_inplace(b);
                        }
                    }
                }
                Accumulation::Multiply => {
                    for w in weights.iter() {
                        weight.mul_inplace(w);
                    }
                    if let Some(bias) = &mut bias {
                        for b in biases.iter() {
                            bias.mul_inplace(b);
                        }
                    }
                }
                Accumulation::Subtract => {
                    for w in weights.iter() {
                        weight.sub_inplace(w);
                    }
                    if let Some(bias) = &mut bias {
                        for b in biases.iter() {
                            bias.sub_inplace(b);
                        }
                    }
                }
                Accumulation::Mean => {
                    for w in weights.iter() {
                        weight.add_inplace(w);
                    }
                    if let Some(bias) = &mut bias {
                        for b in biases.iter() {
                            bias.add_inplace(b);
                        }
                    }
                    weight.div_scalar_inplace(count);
                    if let Some(b) = &mut bias {
                        b.div_scalar_inplace(count);
                    }
                }
                Accumulation::Overwrite => {
                    // Do nothing?
                    unimplemented!("Overwrite accumulation is not implemented.")
                }
            }

            // Update the weights and biases of the coupled layers.
            for i in couple.iter() {
                match &mut self.layers[*i] {
                    network::Layer::Dense(layer) => {
                        layer.weights = weight.clone();
                        if let Some(b) = &mut layer.bias {
                            *b = bias.clone().unwrap();
                        }
                    }
                    network::Layer::Convolution(layer) => {
                        layer.kernels = weight.unnested();
                    }
                    network::Layer::Deconvolution(layer) => {
                        layer.kernels = weight.unnested();
                    }
                    _ => continue,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{activation, assert_eq_data, assert_eq_shape, dense, network, optimizer, tensor};

    #[test]
    fn test_feedback_create() {
        let layers = vec![
            network::Layer::Dense(dense::Dense::create(
                tensor::Shape::Single(2),
                tensor::Shape::Single(2),
                &activation::Activation::ReLU,
                false,
                None,
            )),
            network::Layer::Dense(dense::Dense::create(
                tensor::Shape::Single(2),
                tensor::Shape::Single(2),
                &activation::Activation::ReLU,
                false,
                None,
            )),
        ];
        let feedback = Feedback::create(layers.clone(), 2, true, Accumulation::Add);

        assert_eq!(feedback.inputs, tensor::Shape::Single(2));
        assert_eq!(feedback.outputs, tensor::Shape::Single(2));
        assert_eq!(feedback.layers.len(), 4); // 2 loops of 2 layers each
        assert_eq!(feedback.coupled.len(), 2);
        assert_eq!(feedback.connect.len(), 1);
    }

    // #[test]
    // fn test_feedback_copy_optimizer() {
    //     let layers = vec![network::Layer::Dense(dense::Dense::create(
    //         tensor::Shape::Single(2),
    //         tensor::Shape::Single(2),
    //         &activation::Activation::ReLU,
    //         false,
    //         None,
    //     ))];
    //     let mut feedback = Feedback::create(layers.clone(), 1, false, Accumulation::Add);
    //     let optimizer = optimizer::SGD::create(0.1, None);
    //     feedback.copy_optimizer(optimizer.clone());

    //     assert_eq!(feedback.optimizer, optimizer);
    // }

    #[test]
    fn test_feedback_parameters() {
        let layers = vec![network::Layer::Dense(dense::Dense::create(
            tensor::Shape::Single(3),
            tensor::Shape::Single(3),
            &activation::Activation::ReLU,
            true,
            None,
        ))];
        let feedback = Feedback::create(layers.clone(), 1, false, Accumulation::Add);

        assert_eq!(feedback.parameters(), 12); // 9 weights + 3 biases
    }

    #[test]
    fn test_feedback_training() {
        let layers = vec![network::Layer::Dense(dense::Dense::create(
            tensor::Shape::Single(3),
            tensor::Shape::Single(3),
            &activation::Activation::ReLU,
            true,
            None,
        ))];
        let mut feedback = Feedback::create(layers.clone(), 1, false, Accumulation::Add);
        feedback.training(true);

        for layer in feedback.layers.iter() {
            if let network::Layer::Dense(layer) = layer {
                assert!(layer.training);
            }
        }
    }

    #[test]
    fn test_feedback_forward() {
        let mut layer = dense::Dense::create(
            tensor::Shape::Single(3),
            tensor::Shape::Single(3),
            &activation::Activation::ReLU,
            true,
            None,
        );
        layer.weights = tensor::Tensor::double(vec![vec![1.0; 3]; 3]);
        layer.bias = Some(tensor::Tensor::single(vec![0.0; 3]));
        let layers = vec![network::Layer::Dense(layer)];
        let feedback = Feedback::create(layers.clone(), 1, false, Accumulation::Add);
        let input = tensor::Tensor::single(vec![-1.0, 2.0, 3.0]);

        let (unactivated, activated, maxpool, intermediate_unactivated, intermediate_activated) =
            feedback.forward(&input);

        assert_eq_shape!(unactivated.shape, tensor::Shape::Single(3));
        assert_eq_shape!(activated.shape, tensor::Shape::Single(3));
        assert_eq_shape!(maxpool.shape, tensor::Shape::Nested(1));
        assert_eq_shape!(
            intermediate_unactivated.shape,
            tensor::Tensor::nested(vec![tensor::Tensor::single(vec![1.0; 3]),]).shape
        );
        assert_eq_shape!(
            intermediate_activated.shape,
            tensor::Tensor::nested(vec![
                tensor::Tensor::single(vec![1.0; 3]),
                tensor::Tensor::single(vec![1.0; 3]),
            ])
            .shape
        );

        // Check actual values
        let expected_unactivated = tensor::Tensor::single(vec![4.0; 3]);
        let expected_activated = tensor::Tensor::single(vec![4.0; 3]);
        assert_eq_data!(unactivated.data, expected_unactivated.data);
        assert_eq_data!(activated.data, expected_activated.data);
    }

    #[test]
    fn test_feedback_backward() {
        let mut layer = dense::Dense::create(
            tensor::Shape::Single(3),
            tensor::Shape::Single(3),
            &activation::Activation::ReLU,
            true,
            None,
        );
        layer.weights = tensor::Tensor::double(vec![vec![1.0; 3]; 3]);
        layer.bias = Some(tensor::Tensor::single(vec![0.0; 3]));
        let layers = vec![network::Layer::Dense(layer)];
        let feedback = Feedback::create(layers.clone(), 1, false, Accumulation::Add);
        let input = tensor::Tensor::single(vec![1.0, 2.0, 3.0]);
        let (_, _, _, intermediate_unactivated, intermediate_activated) = feedback.forward(&input);
        let gradient = tensor::Tensor::single(vec![0.1, 0.2, 0.3]);

        let (input_gradient, weight_gradient, bias_gradient) = feedback.backward(
            &gradient,
            &vec![intermediate_unactivated, intermediate_activated],
        );

        assert_eq_shape!(input_gradient.shape, tensor::Shape::Single(3));
        assert_eq!(
            weight_gradient.shape,
            tensor::Tensor::nested(vec![tensor::Tensor::double(vec![vec![1.0; 3]; 2]),]).shape
        );
        assert_eq!(
            bias_gradient.clone().unwrap().shape,
            tensor::Tensor::nested(vec![tensor::Tensor::single(vec![1.0; 3]),]).shape
        );

        // Check actual values
        let expected_input_gradient = tensor::Tensor::single(vec![0.6, 0.6, 0.6]);
        let expected_weight_gradient = tensor::Tensor::nested(vec![tensor::Tensor::double(vec![
            vec![0.1 * 1.0, 0.1 * 2.0, 0.1 * 3.0],
            vec![0.2 * 1.0, 0.2 * 2.0, 0.2 * 3.0],
            vec![0.3 * 1.0, 0.3 * 2.0, 0.3 * 3.0],
        ])]);
        let expected_bias_gradient = tensor::Tensor::single(vec![0.1, 0.2, 0.3]);

        assert_eq_data!(input_gradient.data, expected_input_gradient.data);
        assert_eq_data!(
            weight_gradient.unnested()[0].data,
            expected_weight_gradient.unnested()[0].data
        );
        assert_eq_data!(
            bias_gradient.clone().unwrap().unnestedoptional()[0]
                .clone()
                .unwrap()
                .data,
            expected_bias_gradient.data
        );
    }

    #[test]
    fn test_feedback_update() {
        let layers = vec![network::Layer::Dense(dense::Dense::create(
            tensor::Shape::Single(3),
            tensor::Shape::Single(3),
            &activation::Activation::ReLU,
            true,
            None,
        ))];
        let mut weight_gradient = tensor::Tensor::nested(vec![
            tensor::Tensor::double(vec![
                vec![0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6],
                vec![0.7, 0.8, 0.9],
            ]),
            tensor::Tensor::double(vec![
                vec![0.1, 0.2, 0.3],
                vec![0.7, 0.8, 0.9],
                vec![0.4, 0.5, 0.6],
            ]),
            tensor::Tensor::double(vec![
                vec![0.7, 0.8, 0.9],
                vec![0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6],
            ]),
        ]);
        let mut bias_gradient = tensor::Tensor::nestedoptional(vec![
            Some(tensor::Tensor::single(vec![0.1, 0.2, 0.3])),
            Some(tensor::Tensor::single(vec![0.5, 0.7, 1.0])),
            Some(tensor::Tensor::single(vec![1.1, 1.2, 0.3])),
        ]);

        for accumulation in vec![
            Accumulation::Add,
            Accumulation::Subtract,
            Accumulation::Multiply,
            // Accumulation::Overwrite,
            Accumulation::Mean,
        ] {
            let mut feedback = Feedback::create(layers.clone(), 3, false, accumulation.clone());
            feedback.update(1, &mut weight_gradient, &mut bias_gradient);

            let (weight, bias) = match &feedback.layers[0] {
                network::Layer::Dense(layer) => (layer.weights.clone(), layer.bias.clone()),
                _ => panic!("Invalid layer type"),
            };

            // Check if weights and biases have been updated
            for i in 0..3 {
                match &feedback.layers[i] {
                    network::Layer::Dense(layer) => {
                        assert_eq_data!(layer.weights.data, weight.data);
                        if let Some(bias) = &bias {
                            assert_eq_data!(layer.bias.clone().unwrap().data, bias.data);
                        } else {
                            panic!("Should have bias!");
                        }
                    }
                    _ => panic!("Invalid layer type"),
                }
            }
        }
    }
}
