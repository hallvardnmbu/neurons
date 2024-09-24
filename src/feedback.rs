// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::{assert_eq_shape, network, tensor};

pub enum Accumulation {
    Add,
    Sub,
    Multiply,
    Overwrite,
    Mean,
    // TODO: Expand?
}

impl std::fmt::Display for Accumulation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Accumulation::Add => write!(f, "additive"),
            Accumulation::Sub => write!(f, "subtractive"),
            Accumulation::Multiply => write!(f, "multiplicative"),
            Accumulation::Overwrite => write!(f, "overwrite"),
            Accumulation::Mean => write!(f, "mean"),
            #[allow(unreachable_patterns)]
            _ => unimplemented!("Accumulation method not implemented."),
        }
    }
}

/// A feedback block.
///
/// # Attributes
///
/// * `inputs` - The number of inputs to the block.
/// * `outputs` - The number of outputs from the block.
/// * `flatten` - Whether the block should flatten the output.
/// * `layers` - The layers of the block.
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
    pub(crate) flatten: bool,
    pub(crate) layers: Vec<network::Layer>,
    pub(crate) coupled: Vec<Vec<usize>>,
}

impl std::fmt::Display for Feedback {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Feedback(\n")?;
        write!(f, "\t\t\t{} -> {}\n", self.inputs, self.outputs)?;
        write!(f, "\t\t\tlayers: (\n")?;
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                network::Layer::Dense(layer) => {
                    write!(
                        f,
                        "\t\t\t\t{}: Dense{}({} -> {})\n",
                        i, layer.activation, layer.inputs, layer.outputs
                    )?;
                }
                network::Layer::Convolution(layer) => {
                    write!(
                        f,
                        "\t\t\t\t{}: Convolution{}({} -> {})\n",
                        i, layer.activation, layer.inputs, layer.outputs
                    )?;
                }
                network::Layer::Maxpool(layer) => {
                    write!(
                        f,
                        "\t\t\t\t{}: Maxpool({} -> {})\n",
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
            write!(f, "\t\t\t)\n")?;
        }
        write!(f, "\t\t\tflatten: {}\n", self.flatten)?;
        write!(f, "\t\t)")?;
        Ok(())
    }
}

impl Feedback {
    pub fn create(mut layers: Vec<network::Layer>, loops: usize) -> Self {
        assert!(loops > 0, "Feedback block should loop at least once.");
        let inputs = match layers.first().unwrap() {
            network::Layer::Dense(dense) => dense.inputs.clone(),
            network::Layer::Convolution(convolution) => convolution.inputs.clone(),
            network::Layer::Maxpool(maxpool) => maxpool.inputs.clone(),
            network::Layer::Feedback(_) => panic!("Nested feedback blocks are not supported."),
        };
        let outputs = match layers.last().unwrap() {
            network::Layer::Dense(dense) => dense.outputs.clone(),
            network::Layer::Convolution(convolution) => convolution.outputs.clone(),
            network::Layer::Maxpool(maxpool) => maxpool.outputs.clone(),
            network::Layer::Feedback(_) => panic!("Nested feedback blocks are not supported."),
        };
        assert_eq_shape!(inputs, outputs);

        let length = layers.len();

        // Extend the layers `loops` times.
        for _ in 1..loops {
            layers.extend(layers.clone());
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

        Feedback {
            inputs,
            outputs,
            flatten: false,
            layers,
            coupled,
        }
    }

    // Count the number of parameters.
    // Only counts the parameters of the first loop, as the rest are identical.
    pub fn parameters(&self) -> usize {
        let mut parameters = 0;
        for idx in 0..self.coupled.len() {
            parameters += match &self.layers[idx] {
                network::Layer::Dense(dense) => dense.parameters(),
                network::Layer::Convolution(convolution) => convolution.parameters(),
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
            network::Layer::Maxpool(_) => (),
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
    /// A tuple containing the pre- and post-activation values and the maxpool indices (if any) of
    /// each layer.
    pub fn forward(
        &self,
        input: &tensor::Tensor,
    ) -> (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        Vec<Option<tensor::Tensor>>,
    ) {
        let mut unactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];
        let mut maxpools: Vec<Option<tensor::Tensor>> = Vec::new();

        for layer in self.layers.iter() {
            match layer {
                network::Layer::Dense(layer) => {
                    let (pre, post) = layer.forward(activated.last().unwrap());
                    unactivated.push(pre);
                    activated.push(post);
                    maxpools.push(None);
                }
                network::Layer::Convolution(layer) => {
                    let (pre, post) = layer.forward(activated.last().unwrap());
                    unactivated.push(pre);
                    activated.push(post);
                    maxpools.push(None);
                }
                network::Layer::Maxpool(layer) => {
                    let (pre, post, max) = layer.forward(activated.last().unwrap());
                    unactivated.push(pre);
                    activated.push(post);
                    maxpools.push(Some(max))
                }
                network::Layer::Feedback(_) => panic!("Nested feedback blocks are not supported."),
            };
        }

        (unactivated, activated, maxpools)
    }

    pub fn backward(&self, _input: &tensor::Tensor, _output: &tensor::Tensor) -> tensor::Tensor {
        panic!("TODO!")
    }
}
